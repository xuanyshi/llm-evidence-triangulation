"""Regression tests use synthetic abstracts and a local fake model service."""
import argparse
from contextlib import redirect_stdout
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_batch as batch

QUESTION = {'exposure': 'Exposure X', 'outcome': 'Event Y'}
ABSTRACT = 'Adults received exposure X or control. Event Y HR 1.5 (95% CI 1.2 to 1.8).'


def result():
    effect = {key: None for key in json.loads((batch.SKILL_ROOT/'references/prompts.json').read_text())['schema']['properties']['effects']['items']['required']}
    effect.update(exposure_direction='not found', effect_direction='increase', significance='positive',
                  significance_basis='95% CI', significance_quote=ABSTRACT, population='Adults',
                  intervention='Exposure X', comparator='Control', outcome='Event Y', effect_type='HR',
                  effect_value=1.5, ci_lower=1.2, ci_upper=1.8, source_quote=ABSTRACT,
                  study_design='Not reported', participants_basis='not_reported', participants_quote='',
                  population_condition_quote='', sample_sizes=[])
    return {'picos': {k: {'text': 'Reported study fact', 'source_quote': ABSTRACT} for k in 'PICOS'}, 'effects': [effect]}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        body = json.loads(request['messages'][1]['content'])
        stage = 'map' if 'extracted' in body else 'extract'
        title = body['article']['title']
        with self.server.lock:
            self.server.calls.append((title, stage))
            self.server.active += 1
            self.server.maximum = max(self.server.maximum, self.server.active)
            limited = self.server.limited
            self.server.limited = False
        time.sleep(.03)
        if limited:
            self.send_response(429); self.send_header('Retry-After', '0'); self.end_headers()
        elif self.server.fail_mapping and title == 'Study 1' and stage == 'map':
            self.send_response(503); self.send_header('Retry-After', '0'); self.end_headers()
        else:
            value = result() if stage == 'extract' else {'entities': [{'concept_id': None, 'label': 'Event Y', 'mention': 'Event Y', 'role': 'O', 'source_quote': ABSTRACT}]}
            output = json.dumps({'model': 'test-model', 'choices': [{'message': {'content': json.dumps(value)}}], 'usage': {'total_tokens': 12}}).encode()
            self.send_response(200); self.send_header('Content-Length', str(len(output))); self.end_headers(); self.wfile.write(output)
        with self.server.lock:
            self.server.active -= 1


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.calls, self.server.active, self.server.maximum = [], 0, 0
        self.server.lock = threading.Lock()
        self.server.fail_mapping = False
        self.server.limited = False
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.articles = [{'id': str(i), 'title': f'Study {i}', 'abstract': ABSTRACT} for i in range(3)]
        self.args = argparse.Namespace(output=Path(self.temp.name), requests_per_minute=600000,
            model='test-model', base_url=f'http://127.0.0.1:{self.server.server_port}', stage='both',
            workers=2, max_attempts=3, timeout=2, max_tokens=2000, disable_thinking=False, api_key_env='EVIDENCE_TEST_KEY', resume=False, retry_failed=False,
            retry_mappings=False, terminology_requests_per_minute=120, terminology_timeout=2, terminology_cache_days=30)
        self.env = patch.dict(os.environ, {'EVIDENCE_TEST_KEY': 'local-fake-test-key'}); self.env.start()
        self.terms = patch.object(batch.TerminologyResolver, 'fetch', return_value={
            'query': 'Event Y', 'checked_at': 'test', 'requests': [],
            **{v: None for v in ('mesh', 'icd', 'umls')},
            'checks': {v: {'status': 'not_found', 'candidates': [], 'reason': None} for v in ('mesh', 'icd', 'umls')}})
        self.terms.start()

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.terms.stop(); self.env.stop(); self.temp.cleanup()

    def execute(self, question=None):
        runner = batch.BatchRunner(self.args, question or QUESTION, self.articles, None)
        try:
            runner.initialize()
            with redirect_stdout(io.StringIO()):
                code = runner.run()
            return code
        finally:
            runner.close()

    def test_parallel_resume_and_429(self):
        self.server.limited = True
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.server.calls), 7)
        self.assertEqual(self.server.maximum, 2)
        self.args.resume = True
        self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.server.calls), 7)
        records = [json.loads(line) for line in (self.args.output/'results.jsonl').read_text().splitlines()]
        self.assertEqual([r['id'] for r in records], ['0','1','2'])
        self.assertTrue(all(r['status'] == 'succeeded' for r in records))
        self.assertEqual(records[0]['entities'][0]['mapping_status'], 'needs_review')
        self.assertEqual(len((self.args.output/'mapping_review.jsonl').read_text().splitlines()), 9)
        with (self.args.output/'evidence.csv').open(encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['Effect type'], 'HR')
        self.assertEqual(rows[0]['Significance'], 'Significant')
        self.assertEqual(rows[0]['Mapping review'], 'Needs review')
        self.assertEqual(rows[0]['Relation ID'], records[0]['nc_relations'][0]['relation_id'])
        self.assertEqual(rows[0]['Participants basis'], 'not_reported')
        self.assertEqual(rows[0]['Number of participants'], '')
        self.assertEqual(rows[0]['Exposure direction'], 'not found')
        self.assertIn('P value', rows[0])
        self.assertEqual(records[0]['nc_relations'][0]['statistical_value'], float(rows[0]['Effect value']))

    def test_mapping_upgrade_never_repeats_model_calls(self):
        self.assertEqual(self.execute(), 0)
        before = len(self.server.calls)
        with sqlite3.connect(self.args.output/'run.sqlite3') as db:
            manifest = json.loads(db.execute('SELECT manifest FROM metadata WHERE id=1').fetchone()[0])
            manifest.pop('terminology')
            manifest['implementation_hash'] = 'historical-runner'
            db.execute('UPDATE metadata SET manifest=? WHERE id=1', (json.dumps(manifest),))
        self.args.retry_mappings = True
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.execute(), 0)
        self.assertEqual(len(self.server.calls), before)
        with sqlite3.connect(self.args.output/'run.sqlite3') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM mapping_migrations').fetchone()[0], 1)

    def test_failed_mapping_retries_only_that_stage(self):
        self.server.fail_mapping = True
        self.args.max_attempts = 1
        self.assertEqual(self.execute(), 1)
        before = len(self.server.calls)
        self.server.fail_mapping = False
        self.args.retry_failed = True
        self.assertEqual(self.execute(), 0)
        self.assertEqual(self.server.calls[before:], [('Study 1','map')])

    def test_changed_question_and_concurrent_owner_rejected(self):
        runner = batch.BatchRunner(self.args, QUESTION, self.articles, None)
        runner.initialize()
        other = batch.BatchRunner(self.args, QUESTION, self.articles, None)
        try:
            with self.assertRaises((ValueError, sqlite3.OperationalError)):
                other.initialize()
        finally:
            other.close(); runner.close()
        self.args.resume = True
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.execute({'exposure': 'Exposure X', 'outcome': 'Different endpoint'})

    def test_interrupted_mapping_retains_extraction(self):
        runner = batch.BatchRunner(self.args, QUESTION, self.articles, None)
        runner.initialize()
        runner.save('0', status='running', extraction=batch.packed(result()))
        runner.close()
        self.args.resume = True
        self.assertEqual(self.execute(), 0)
        self.assertNotIn(('Study 0','extract'), self.server.calls)
        self.assertIn(('Study 0','map'), self.server.calls)

    def test_input_and_grounding_validation(self):
        path = self.args.output/'input.jsonl'
        path.write_text('\n'.join(json.dumps(a) for a in self.articles))
        self.assertEqual(len(batch.read_articles(path)), 3)
        path.write_text(json.dumps([self.articles[0], self.articles[0]]))
        with self.assertRaises(ValueError): batch.read_articles(path)
        data = result(); data['effects'][0]['source_quote'] = 'Invented result'
        schema = json.loads((batch.SKILL_ROOT/'references/prompts.json').read_text())['schema']
        with self.assertRaises(ValueError): batch.validate_extraction(data, self.articles[0], schema)
        data = result(); data['effects'][0]['effect_direction'] = 'decrease'
        with self.assertRaises(ValueError): batch.validate_extraction(data, self.articles[0], schema)
        data = result(); data['effects'][0]['significance'] = 'negative'
        with self.assertRaises(ValueError): batch.validate_extraction(data, self.articles[0], schema)

    def test_rate_limiter_spaces_requests(self):
        stop = threading.Event()
        limiter = batch.RateLimiter(60, stop)
        with patch.object(batch.time, 'monotonic', return_value=100), patch.object(stop, 'wait', return_value=False) as waiter:
            limiter.acquire(); limiter.acquire(); limiter.acquire()
            self.assertEqual([c.args[0] for c in waiter.call_args_list], [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
