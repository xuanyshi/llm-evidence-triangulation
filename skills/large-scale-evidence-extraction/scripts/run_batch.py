"""Run resumable, bounded-concurrency abstract extraction on an OpenAI-compatible API."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import jsonschema
from prepare_messages import SKILL_ROOT, build_messages
from terminology import TerminologyResolver, RESOLVER_VERSION


def packed(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def read_articles(path):
    with path.open(encoding='utf-8-sig', newline='') as source:
        if path.suffix.lower() == '.csv':
            records = list(csv.DictReader(source))
        elif path.suffix.lower() == '.jsonl':
            records = [json.loads(line) for line in source if line.strip()]
        else:
            records = json.load(source)
    if isinstance(records, dict):
        records = [records]
    if not isinstance(records, list) or not records:
        raise ValueError('Input must contain at least one article.')
    found = set()
    for article in records:
        if not isinstance(article, dict):
            raise ValueError('Each article must be an object.')
        identifier = str(article.get('id') or article.get('pmid') or '').strip()
        if not identifier or identifier in found:
            raise ValueError('Each article needs a unique nonempty id or pmid; use relation_id plus PMID for repeated studies.')
        found.add(identifier)
        article['id'] = identifier
    return records


class StageError(Exception):
    pass


class RateLimiter:
    def __init__(self, rpm, stop):
        self.interval = 60 / rpm
        self.next_start = 0.0
        self.lock = threading.Lock()
        self.stop = stop

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        if self.stop.wait(max(0, start - now)):
            raise StageError('Run interrupted before request.')


def retry_delay(value, attempt):
    try:
        seconds = float(value)
    except (ValueError, TypeError):
        try:
            seconds = parsedate_to_datetime(value).timestamp() - time.time()
        except (ValueError, TypeError, OverflowError):
            seconds = 2 ** attempt + random.random()
    if not math.isfinite(seconds):
        seconds = 2 ** attempt
    return max(0, min(seconds, 300))


def significance_resolution(effect):
    reported, inferred = effect['significance_reported'], effect['significance_inferred']
    conflict = reported != 'not found' and inferred != 'not found' and reported != inferred
    if conflict:
        return 'not found', 'Not found', '', True
    if reported != 'not found':
        return reported, 'Reported statement', effect['significance_reported_quote'], False
    return inferred, effect['significance_inferred_basis'], effect['significance_inferred_quote'], False


def p_significance(raw, alpha):
    match = re.fullmatch(r'\s*(?:[pP]\s*)?(<=|>=|<|>|=|≤|≥)?\s*((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*', raw or '')
    if not match:
        return 'not found'
    operator, value = match[1] or '=', float(match[2])
    if not 0 <= value <= 1:
        return 'not found'
    if (operator == '=' and value < alpha) or (operator == '<' and value <= alpha) or (operator in ('<=', '≤') and value < alpha):
        return 'positive'
    if (operator == '=' and value > alpha) or (operator == '>' and value >= alpha) or (operator in ('>=', '≥') and value > alpha):
        return 'negative'
    return 'not found'


def validate_significance(effect, article):
    for name in ('reported', 'inferred'):
        value, quote = effect['significance_' + name], effect['significance_' + name + '_quote']
        if value == 'not found':
            if quote:
                raise ValueError(f'Unknown {name} significance requires an empty quote.')
        elif not quote or quote not in article['abstract']:
            raise ValueError(f'{name} significance requires its own literal abstract quote.')
    inferred, basis, alpha = (effect['significance_inferred'], effect['significance_inferred_basis'], effect['significance_inferred_alpha'])
    if inferred == 'not found':
        if basis != 'Not found' or alpha is not None:
            raise ValueError('Unknown inferred significance requires Not found basis and null alpha.')
    elif basis == '95% CI':
        if effect['ci_lower'] is None or effect['ci_upper'] is None or alpha is not None:
            raise ValueError('CI inference requires both 95% bounds and null P-test alpha.')
    elif basis == 'P value':
        if alpha is None or inferred != p_significance(effect['p_value'], alpha):
            raise ValueError('P inference requires a reported threshold and a P value/bound that resolves significance at that threshold.')
    else:
        raise ValueError('Inferred significance requires a numeric basis.')
    resolved, resolved_basis, quote, _ = significance_resolution(effect)
    if (effect['significance'], effect['significance_basis'], effect['significance_quote']) != (resolved, resolved_basis, quote):
        raise ValueError('Legacy significance must follow the two-channel resolution: conflict -> not found; otherwise reported first, then inferred. Preserve both original channels.')


def validate_extraction(data, article, schema, question=None):
    jsonschema.validate(data, schema)
    for field, item in data['picos'].items():
        quote = item['source_quote']
        if not quote and item['text'] == 'Not reported':
            continue
        if not quote or not any(quote in article[k] for k in ('title', 'abstract')):
            raise ValueError(f'PICOS {field} source_quote {quote!r} is not a literal title/abstract passage. Copy a contiguous passage without dropping intervening words; use Not reported with empty quote only if absent.')
    for effect in data['effects']:
        validate_participants(effect, article)
        validate_significance(effect, article)
        value = effect['effect_value']
        if value is not None and not math.isfinite(value):
            raise ValueError('Reported effect values must be finite.')
        if value is None and effect['effect_direction'] == 'not found' and effect['significance_reported'] == effect['significance_inferred'] == 'not found':
            raise ValueError('A qualitative row requires a reported direction or significance.')
        quote = effect['source_quote']
        if not quote or quote not in article['abstract']:
            raise ValueError('Each effect needs an exact nonempty abstract quote.')
        significance = effect['significance_quote']
        if significance and significance not in article['abstract']:
            raise ValueError('Significance quote must occur verbatim in the abstract.')
        if effect['significance'] != 'not found' and not significance:
            raise ValueError('A significance classification requires its source passage.')
        lo, hi = effect['ci_lower'], effect['ci_upper']
        if (lo is None) != (hi is None):
            raise ValueError('Supply both stated 95% CI bounds or leave both null.')
        if lo is not None and (not all(math.isfinite(x) for x in (lo, hi)) or lo > hi or (value is not None and not lo <= value <= hi)):
            raise ValueError('Check the point estimate and its exact 95% interval.')
        metric = effect['effect_type'].strip().lower()
        average_comparator = re.search(r'(?:average|mean).*?(?:risk|population)|(?:population|overall).*?(?:average|mean)', effect['comparator'], re.I)
        contrast = ' '.join((question or {}).get(k, '') for k in ('contrast', 'comparator'))
        extreme_contrast = re.search(r'\b(?:high|highest|higher|top)\b.*\b(?:low|lowest|lower|bottom)\b', contrast, re.I)
        if average_comparator and extreme_contrast:
            raise ValueError('The question requires higher/highest versus lower/lowest groups. A population-average reference is ineligible. Omit that estimate; use an explicitly reported eligible comparison such as paired cumulative incidences if available. Do not relabel an HR or borrow a trend/global test as pairwise significance.')
        population_reference = re.search(r'(?:hazard ratios?|\bHRs?\b)[^.!?]{0,200}relative to (?:the )?average risk in the whole study population', article['abstract'], re.I)
        if metric in ('hr', 'hazard ratio') and population_reference and not re.search(r'(?:average|mean).*?(?:risk|population)|(?:population|overall).*?(?:average|mean)', effect['comparator'], re.I):
            raise ValueError('Methods explicitly define HRs relative to average risk in the whole study population. Do not label those HRs as high versus low. Use the stated population-average comparator only if eligible for the question; otherwise omit these HRs and retain an eligible directly reported comparison such as paired cumulative incidences, without borrowing HR/trend significance.')
        null = None
        if metric in ('hr', 'or', 'rr', 'hazard ratio', 'odds ratio', 'risk ratio', 'relative risk'):
            null = 1
        elif metric in ('rd', 'risk difference', 'mean difference', 'standardized mean difference', 'log odds ratio', 'log hazard ratio', 'log risk ratio', 'correlation'):
            null = 0
        if null is not None:
            if value is not None:
                delta = value - null
                direction = 'no_change' if delta == 0 else 'increase' if delta > 0 else 'decrease'
                if effect['effect_direction'] != direction:
                    raise ValueError('Direction disagrees with the reported metric and point estimate.')
            if lo is not None and effect['significance_inferred_basis'] == '95% CI':
                expected = 'negative' if lo <= null <= hi else 'positive'
                if effect['significance_inferred'] != expected:
                    raise ValueError('Significance disagrees with this estimate\'s 95% interval.')
        if effect['comparator_estimate'] is not None:
            if value is None:
                raise ValueError('Paired incidence requires both group values.')
            if metric != 'cumulative incidence' or effect['unit'] != '%':
                raise ValueError('Paired comparator values are reserved for cumulative incidence percentages.')
            if not 0 <= value <= 100 or not 0 <= effect['comparator_estimate'] <= 100:
                raise ValueError('Cumulative incidence percentages must lie between 0 and 100.')
    return data


def validate_participants(effect, article):
    def source(quote):
        if not quote or not any(quote in article[k] for k in ('title', 'abstract')):
            raise ValueError('Population and count fields require exact source quotes.')

    def stated(number, quote):
        source(quote)
        # Recognize digit counts, including thousands separators, without accepting
        # 96 inside 2096 or treating an exponent as an expanded participant count.
        tokens = re.findall(r'(?<![\w.])\d{1,3}(?:[,\s\u00a0]\d{3})+(?:\.\d+)?(?!\w)|(?<![\w.])\d+(?:\.\d+)?(?!\w)', quote)
        if not math.isfinite(number) or not any(float(re.sub(r'[,\s\u00a0]', '', token)) == number for token in tokens):
            raise ValueError('Count must be explicitly stated as digits in its source quote.')

    condition = effect['population_main_condition']
    if condition is not None:
        if not condition.strip():
            raise ValueError('Use null for an unreported baseline condition.')
        source(effect['population_condition_quote'])
    elif effect['population_condition_quote']:
        raise ValueError('An unreported baseline condition has an empty quote.')
    n = effect['number_of_participants']
    basis = effect['participants_basis']
    if n is None:
        if basis != 'not_reported' or effect['participants_quote']:
            raise ValueError('Unknown participant N requires not_reported and an empty quote.')
    else:
        if basis == 'not_reported':
            raise ValueError('Participant N needs an analysis or enrolled population basis.')
        stated(n, effect['participants_quote'])
        role = 'analysis_population' if basis == 'analysis_population' else 'cohort_total'
        if not any(item['role'] == role and item['unit'] == 'participants' and item['number'] == n and item['source_quote'] == effect['participants_quote'] for item in effect['sample_sizes']):
            raise ValueError('Participant N must match its scoped sample_sizes item and quote.')
    for item in effect['sample_sizes']:
        stated(item['number'], item['source_quote'])
        if not item['population'].strip():
            raise ValueError('Each count needs its population or analysis scope.')
        if item['unit'] in ('participants', 'events', 'clusters') and int(item['number']) != item['number']:
            raise ValueError('Participant, event and cluster counts must be integers.')
        if item['role'] == 'events' and item['unit'] != 'events':
            raise ValueError('Event counts cannot supply participant N.')
        if item['role'] in ('cohort_total', 'analysis_population', 'intervention_group', 'comparator_group', 'cases', 'controls', 'exposure_gwas', 'outcome_gwas') and item['unit'] != 'participants':
            raise ValueError('Participant-group counts require participants units; use other for clusters/person-years.')
    included = effect['included_studies']
    if included is not None:
        if effect['study_design'] not in ('META', 'SR', 'REVIEW'):
            raise ValueError('included_studies must be null for primary studies.')
        source(included['source_quote'])
        if included['total'] is None and not included['by_design']:
            raise ValueError('Use null when no included-study count is reported.')
        if included['total'] is not None:
            stated(included['total'], included['source_quote'])
        for item in included['by_design']:
            stated(item['count'], included['source_quote'])
            if included['total'] is not None and item['count'] > included['total']:
                raise ValueError('Design-specific study count exceeds the reported total.')


def nc_relation(identifier, position, effect):
    """Lossless export projection; never overwrite raw model direction or P text."""
    raw_p = effect.get('p_value')
    match = re.fullmatch(r'\s*(?:[pP]\s*)?(<=|>=|<|>|=|≤|≥)?\s*((?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*', raw_p or '')
    operator = (match[1] or '=') if match else None
    p_number = float(match[2]) if match else None
    if p_number is not None and not 0 <= p_number <= 1:
        p_number, operator = None, None
    direction = 'no_change' if effect['significance'] == 'negative' else effect['effect_direction'] if effect['significance'] == 'positive' else 'not found'
    included = effect.get('included_studies')
    included_text = None
    if included:
        parts = [f"{i['count']} {i['study_design']}" for i in included['by_design']]
        included_text = '; '.join(([f"{included['total']} total"] if included['total'] is not None else []) + parts)
    return {
        'relation_id': f'{identifier}:relation:{position:04d}',
        'exposure': effect['intervention'], 'exposure_direction': effect['exposure_direction'],
        'outcome': effect['outcome'], 'significance': effect['significance'], 'direction': direction,
        **{k: effect.get(k) for k in ('significance_reported', 'significance_reported_quote', 'significance_inferred', 'significance_inferred_basis', 'significance_inferred_quote', 'significance_inferred_alpha')},
        'significance_conflict': significance_resolution(effect)[3] if 'significance_reported' in effect else None,
        'population_main_condition': effect.get('population_main_condition'),
        'condition': effect.get('population_main_condition'), 'condition_quote': effect.get('population_condition_quote'),
        'comparator': effect['comparator'],
        'study_design': effect.get('study_design'), 'included_studies': included_text,
        'number_of_participants': effect.get('number_of_participants'),
        'study_design_detail': effect.get('study_design_detail'), 'study_design_category': effect.get('study_design_category'),
        'statistical_parameter': None if effect['effect_type'] == 'Not reported' else effect['effect_type'],
        'statistical_value': effect['effect_value'], 'ci_lower': effect['ci_lower'], 'ci_upper': effect['ci_upper'],
        'p_value': p_number if operator == '=' else None, 'p_value_raw': raw_p,
        'p_value_operator': operator, 'p_value_bound': p_number if operator and operator != '=' else None,
        'effect_direction': effect['effect_direction'], 'result_kind': 'quantitative' if effect['effect_value'] is not None else 'qualitative',
        'participants_basis': effect.get('participants_basis'), 'sample_sizes': effect.get('sample_sizes'),
        'participants_quote': effect.get('participants_quote'), 'included_studies_detail': included,
        'source_quote': effect['source_quote'], 'direction_rule': 'NC: negative significance -> no_change; unknown significance -> not found; otherwise raw effect direction',
    }


def validate_entities(data, article, index):
    if not isinstance(data, dict) or not isinstance(data.get('entities'), list):
        raise ValueError('Return an entities array.')
    catalog = {item['id']: item for item in (index or {}).get('concepts', [])}
    entities, seen = [], set()
    required = {'concept_id', 'label', 'mention', 'role', 'source_quote'}
    for item in data['entities']:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError('Each entity must contain concept_id, label, mention, role and source_quote.')
        if not all(isinstance(item[k], str) and item[k].strip() for k in required - {'concept_id'}):
            raise ValueError('Entity label, role, mention and source_quote must be nonempty strings.')
        if item['role'] not in 'PICOS' or len(item['role']) != 1:
            raise ValueError('Use P, I, C, O or S for an entity role.')
        quote = item['source_quote']
        if item['mention'] not in quote or not any(quote in article[k] for k in ('title', 'abstract')):
            raise ValueError('Each entity quote must be literal and contain its mention.')
        concept = item['concept_id']
        if concept is not None and (not isinstance(concept, str) or concept not in catalog):
            raise ValueError('Use a supplied concept_id or null for an unindexed concept.')
        key = (item['role'], concept or item['label'].casefold())
        if key in seen:
            continue
        seen.add(key)
        term = catalog.get(concept, {})
        entities.append({**item, 'label': term.get('label', item['label']), **{v: term.get(v) for v in ('mesh', 'icd', 'umls')},
                         'mapping_status': 'index' if term else 'unresolved'})
    return entities


class BatchRunner:
    def __init__(self, args, question, articles, index):
        self.args, self.question, self.articles, self.index = args, question, articles, index
        self.stop = threading.Event()
        self.limiter = RateLimiter(args.requests_per_minute, self.stop)
        self.db_path = args.output / 'run.sqlite3'
        self.prompts = json.loads((SKILL_ROOT / 'references/prompts.json').read_text())
        # A separate SQLite write transaction holds a process-wide output lock.
        # The operating system releases this lock after a crash.
        self.owner = None
        self.resolver = None

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def initialize(self):
        self.args.output.mkdir(parents=True, exist_ok=True)
        self.owner = sqlite3.connect(self.args.output / 'owner.sqlite3', timeout=0)
        self.owner.execute('CREATE TABLE IF NOT EXISTS owner (id INTEGER PRIMARY KEY)')
        try:
            self.owner.execute('BEGIN IMMEDIATE')
        except sqlite3.OperationalError:
            self.owner.close()
            self.owner = None
            raise ValueError('Another process owns this output directory.') from None
        manifest = {
            'question': self.question, 'model': self.args.model, 'base_url': self.args.base_url,
            'stage': self.args.stage, 'temperature': 0, 'max_tokens': self.args.max_tokens,
            'disable_thinking': self.args.disable_thinking,
            'prompt_hash': digest(self.prompts), 'index_hash': digest(self.index),
            'input_hash': digest(self.articles),
            'terminology': {'resolver_version': RESOLVER_VERSION, 'online_icd_system': 'ICD-10-CM'},
            'implementation_hash': hashlib.sha256(Path(__file__).read_bytes() + (SKILL_ROOT / 'scripts/prepare_messages.py').read_bytes() + (SKILL_ROOT / 'scripts/terminology.py').read_bytes()).hexdigest(),
        }
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, manifest TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, position INTEGER, article TEXT, status TEXT, extraction TEXT, entities TEXT, error TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, study TEXT, stage TEXT, attempt INTEGER, started REAL, elapsed REAL, status TEXT, usage TEXT, response TEXT)')
            previous = db.execute('SELECT manifest FROM metadata WHERE id=1').fetchone()
            if previous:
                old_manifest = json.loads(previous['manifest'])
                if old_manifest != manifest:
                    omit = {'implementation_hash', 'terminology'}
                    compatible = {k: v for k, v in old_manifest.items() if k not in omit} == {k: v for k, v in manifest.items() if k not in omit}
                    if not self.args.retry_mappings or not compatible:
                        raise ValueError('Question, input, model, prompts, index or implementation changed. Use a new output directory; --retry-mappings permits terminology-only upgrades.')
                    db.execute('CREATE TABLE IF NOT EXISTS mapping_migrations (id INTEGER PRIMARY KEY, previous_manifest TEXT, updated REAL)')
                    db.execute('INSERT INTO mapping_migrations (previous_manifest,updated) VALUES (?,?)', (previous['manifest'], time.time()))
                    db.execute('UPDATE metadata SET manifest=? WHERE id=1', (packed(manifest),))
                if not (self.args.resume or self.args.retry_failed or self.args.retry_mappings):
                    raise ValueError('Output already contains a run. Use --resume, --retry-failed or --retry-mappings.')
            else:
                if self.args.retry_mappings:
                    raise ValueError('--retry-mappings requires an existing run with saved entities.')
                db.execute('INSERT INTO metadata VALUES (1, ?)', (packed(manifest),))
                for position, article in enumerate(self.articles):
                    db.execute('INSERT INTO jobs VALUES (?,?,?, ?,NULL,NULL,NULL)',
                               (article['id'], position, packed(article), 'pending'))
            db.execute("UPDATE jobs SET status='pending' WHERE status='running'")
        self.resolver = TerminologyResolver(self.args.output / 'terminology.sqlite3',
            rpm=self.args.terminology_requests_per_minute, timeout=self.args.terminology_timeout,
            cache_days=self.args.terminology_cache_days, stop=self.stop)
        self.export()

    def close(self):
        if self.owner is not None:
            self.owner.rollback()
            self.owner.close()
            self.owner = None

    def save(self, identifier, **fields):
        # Column names come only from the runner, never from model data.
        with self.connect() as db:
            db.execute('UPDATE jobs SET ' + ','.join(k + '=?' for k in fields) + ' WHERE id=?',
                       (*fields.values(), identifier))

    def log_request(self, identifier, stage, attempt, started, status, response='', usage=None):
        with self.connect() as db:
            db.execute('INSERT INTO requests (study,stage,attempt,started,elapsed,status,usage,response) VALUES (?,?,?,?,?,?,?,?)',
                       (identifier, stage, attempt, started, time.time()-started, status, packed(usage), response))

    def model_call(self, article, stage, extraction=None):
        messages = build_messages(self.question, article, stage, extraction, self.index if stage == 'map' else None)
        validator = (lambda d: validate_extraction(d, article, self.prompts['schema'], self.question)) if stage == 'extract' else (lambda d: validate_entities(d, article, self.index))
        last_error = 'Model stage did not complete.'
        for attempt in range(1, self.args.max_attempts + 1):
            self.limiter.acquire()
            started = time.time()
            payload = {'model': self.args.model, 'messages': messages, 'temperature': 0,
                       'max_tokens': self.args.max_tokens, 'response_format': {'type': 'json_object'}}
            if self.args.disable_thinking:
                payload['thinking'] = {'type': 'disabled'}
            request = Request(self.args.base_url.rstrip('/') + '/chat/completions',
                              data=packed(payload).encode(), method='POST',
                              headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ[self.args.api_key_env]})
            try:
                with urlopen(request, timeout=self.args.timeout) as response:
                    envelope = json.load(response)
                content = envelope['choices'][0]['message']['content']
                if not isinstance(content, str):
                    raise ValueError('Provider returned no textual JSON content.')
                self.log_request(article['id'], stage, attempt, started, 'received', content,
                                 {'reported_model': envelope.get('model'), 'tokens': envelope.get('usage')})
            except HTTPError as error:
                self.log_request(article['id'], stage, attempt, started, f'http_{error.code}')
                if error.code in (401, 403):
                    self.stop.set()
                    raise StageError(f'Provider authorization failed (HTTP {error.code}); check configured credentials.') from None
                if error.code not in (408, 409, 429) and error.code < 500:
                    raise StageError(f'Provider rejected the request (HTTP {error.code}).') from None
                last_error = f'Provider returned HTTP {error.code}.'
                delay = retry_delay(error.headers.get('Retry-After'), attempt)
                if attempt < self.args.max_attempts:
                    self.stop.wait(delay)
                continue
            except (URLError, TimeoutError, OSError, HTTPException, ValueError, KeyError, IndexError, TypeError) as error:
                self.log_request(article['id'], stage, attempt, started, 'transport_or_response_error:' + type(error).__name__)
                last_error = 'Network timeout or invalid provider response.'
                if attempt < self.args.max_attempts:
                    self.stop.wait(retry_delay(None, attempt))
                continue
            try:
                return validator(json.loads(content))
            except (ValueError, TypeError, jsonschema.ValidationError) as error:
                last_error = str(error).split('\n')[0][:500]
                messages.extend([{'role': 'assistant', 'content': content},
                                 {'role': 'user', 'content': 'Correct the JSON. ' + last_error}])
        raise StageError(last_error)

    def one(self, identifier):
        with self.connect() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (identifier,)).fetchone()
        article = json.loads(row['article'])
        self.save(identifier, status='running', error=None)
        extraction = json.loads(row['extraction']) if row['extraction'] else None
        try:
            if extraction is None:
                extraction = self.model_call(article, 'extract')
                self.save(identifier, extraction=packed(extraction))
            if self.args.stage == 'both' and not row['entities']:
                entities = self.model_call(article, 'map', extraction)
                self.save(identifier, entities=packed(entities))
            elif row['entities']:
                entities = json.loads(row['entities'])
            if self.args.stage == 'both':
                self.save(identifier, entities=packed(self.resolver.enrich(entities)))
            if self.stop.is_set():
                self.save(identifier, status='pending', error='Interrupted during terminology lookup.')
                return
            self.save(identifier, status='succeeded' if extraction['effects'] else 'no_eligible_result')
        except StageError as error:
            status = 'mapping_failed' if extraction is not None and self.args.stage == 'both' else 'failed'
            if self.stop.is_set():
                status = 'pending'
            self.save(identifier, status=status, error=str(error))
        except Exception:
            self.save(identifier, status='mapping_failed' if extraction is not None else 'failed', error='Unexpected processing error; inspect inputs and runner.')

    def run(self):
        with self.connect() as db:
            if self.args.retry_mappings:
                selected = db.execute('SELECT id FROM jobs WHERE entities IS NOT NULL ORDER BY position')
            else:
                statuses = ('failed', 'mapping_failed') if self.args.retry_failed else ('pending',)
                sql = 'SELECT id FROM jobs WHERE status IN (' + ','.join('?' for _ in statuses) + ') ORDER BY position'
                selected = db.execute(sql, statuses)
            identifiers = iter([r['id'] for r in selected])
        operation = self.refresh_mappings if self.args.retry_mappings else self.one
        pool = ThreadPoolExecutor(max_workers=self.args.workers)
        pending = set()
        try:
            for _ in range(self.args.workers):
                identifier = next(identifiers, None)
                if identifier is not None:
                    pending.add(pool.submit(operation, identifier))
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    if not self.stop.is_set():
                        identifier = next(identifiers, None)
                        if identifier is not None:
                            pending.add(pool.submit(operation, identifier))
                if done:
                    print(packed(self.counts()), flush=True)
        except KeyboardInterrupt:
            self.stop.set()
            print('Stopping after active requests finish; completed stages are saved.', file=sys.stderr)
        finally:
            pool.shutdown(wait=True)
            self.export()
        counts = self.counts()
        return 1 if self.stop.is_set() or any(counts.get(s, 0) for s in ('failed', 'mapping_failed', 'pending', 'running')) else 0

    def refresh_mappings(self, identifier):
        with self.connect() as db:
            row = db.execute('SELECT entities FROM jobs WHERE id=?', (identifier,)).fetchone()
        entities = self.resolver.enrich(json.loads(row['entities']), refresh=True)
        self.save(identifier, entities=packed(entities))

    def counts(self):
        with self.connect() as db:
            return {r['status']: r['n'] for r in db.execute('SELECT status,COUNT(*) AS n FROM jobs GROUP BY status')}

    def export(self):
        with self.connect() as db:
            rows = db.execute('SELECT * FROM jobs ORDER BY position').fetchall()
            manifest = json.loads(db.execute('SELECT manifest FROM metadata WHERE id=1').fetchone()['manifest'])
        json_path = self.args.output / 'results.jsonl'
        csv_path = self.args.output / 'evidence.csv'
        columns = ['Study', 'Relation ID', 'PMID', 'Year', 'Status', 'Mapping review', 'Study design',
                   'Study design code', 'Study design category', 'Study design detail', 'P', 'Condition',
                   'Number of participants', 'Participants basis', 'Participants quote', 'Sample sizes', 'Included studies',
                   'I', 'Exposure direction', 'C', 'O', 'Result kind', 'Effect type', 'Effect value',
                   'Comparator value', 'Unit', 'Effect direction', 'NC direction', 'CI', 'Significance reported', 'Significance inferred', 'Significance inferred basis',
                   'Significance conflict', 'Significance reported quote', 'Significance inferred quote', 'Inference alpha', 'P value',
                   'Adjustment', 'Timepoint', 'Source quote', 'Entities']
        with json_path.with_suffix('.jsonl.tmp').open('w', encoding='utf-8') as out, csv_path.with_suffix('.csv.tmp').open('w', encoding='utf-8-sig', newline='') as table:
            review_items = []
            writer = csv.DictWriter(table, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                article = json.loads(row['article'])
                data = json.loads(row['extraction']) if row['extraction'] else None
                entities = json.loads(row['entities']) if row['entities'] else []
                for entity in entities:
                    for vocab in entity.get('review_vocabularies', []):
                        review_items.append({'study': row['id'], 'role': entity['role'], 'label': entity['label'], 'mention': entity['mention'],
                            'vocabulary': vocab, **entity['mapping_checks'][vocab], 'source_quote': entity['source_quote']})
                relations = [nc_relation(row['id'], i, e) for i, e in enumerate(data['effects'], 1)] if data else []
                out.write(packed({'id': row['id'], 'status': row['status'], 'question': self.question, 'article': article,
                                  'data': data, 'nc_relations': relations, 'entities': entities, 'error': row['error']}) + '\n')
                for position, effect in enumerate((data['effects'] or [None]) if data else [None]):
                    line = {'Study': row['id'], 'Status': row['status'], 'Entities': packed(entities),
                            'PMID': article.get('pmid', ''), 'Year': article.get('year', ''),
                            'Mapping review': 'Needs review' if any(e.get('review_vocabularies') for e in entities) else ''}
                    if data:
                        line.update({'Study design': data['picos']['S']['text'], **{k: data['picos'][k]['text'] for k in 'PICO'}})
                    if effect:
                        relation = relations[position]
                        line.update({'Relation ID': relation['relation_id'], 'Study design code': effect.get('study_design'),
                            'Study design category': effect.get('study_design_category'), 'Study design detail': effect.get('study_design_detail'),
                            'Condition': effect.get('population_main_condition'),
                            'Number of participants': effect.get('number_of_participants'), 'Participants basis': effect.get('participants_basis'),
                            'Participants quote': effect.get('participants_quote'), 'Sample sizes': packed(effect.get('sample_sizes')),
                            'Included studies': packed(effect.get('included_studies')) if effect.get('included_studies') else '',
                            'Exposure direction': effect['exposure_direction'], 'NC direction': relation['direction'],
                            'P value': effect.get('p_value'), 'Result kind': relation['result_kind']})
                        line.update({k: effect[v] for k, v in {'P':'population','I':'intervention','C':'comparator','O':'outcome','Effect type':'effect_type','Effect value':'effect_value','Comparator value':'comparator_estimate','Unit':'unit','Effect direction':'effect_direction','Adjustment':'adjustment','Timepoint':'timepoint','Source quote':'source_quote'}.items()})
                        line['CI'] = '' if effect['ci_lower'] is None else f"{effect['ci_lower']}–{effect['ci_upper']}"
                        labels = {'positive': 'Significant', 'negative': 'Not significant', 'not found': ''}
                        for channel in ('reported', 'inferred'):
                            line['Significance ' + channel] = labels.get(effect.get('significance_' + channel), '')
                            line['Significance ' + channel + ' quote'] = effect.get('significance_' + channel + '_quote', '')
                        line['Significance inferred basis'] = effect.get('significance_inferred_basis', '')
                        line['Inference alpha'] = effect.get('significance_inferred_alpha')
                        line['Significance conflict'] = 'Yes' if relation['significance_conflict'] else ''
                    for role, field in [('S', 'Study design'), ('P', 'P'), ('I', 'I'), ('C', 'C'), ('O', 'O')]:
                        mapped = []
                        for entity in entities:
                            if entity['role'] != role:
                                continue
                            codes = [f"{entity[v].get('code_system', v.upper())} {entity[v]['code']}" for v in ('mesh', 'icd', 'umls') if entity.get(v)]
                            codes.extend(v.upper() + ' needs review' for v in entity.get('review_vocabularies', []))
                            if codes:
                                mapped.append(entity['label'] + ' [' + ', '.join(codes) + ']')
                        if mapped:
                            line[field] = str(line.get(field, '')) + ' | ' + '; '.join(mapped)
                    # Avoid spreadsheet formula evaluation of supplied text.
                    for field, value in line.items():
                        if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')):
                            line[field] = "'" + value
                    writer.writerow(line)
        os.replace(json_path.with_suffix('.jsonl.tmp'), json_path)
        os.replace(csv_path.with_suffix('.csv.tmp'), csv_path)
        review_path = self.args.output / 'mapping_review.jsonl'
        review_tmp = review_path.with_suffix('.jsonl.tmp')
        review_tmp.write_text(''.join(packed(item) + '\n' for item in review_items))
        os.replace(review_tmp, review_path)
        summary = {'counts': self.counts(), 'model': self.args.model, 'manifest': manifest,
                   'updated_at': datetime.now(timezone.utc).isoformat(),
                   'terminology': 'Index plus NLM MeSH, NCI Metathesaurus and NLM ICD-10-CM lookup; unresolved or ambiguous codes require review.',
                   'mapping_review_items': len(review_items),
                   'validation': 'Schema, source spans and selected numeric consistency checks; semantic attribution requires review.'}
        target = self.args.output / 'summary.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--question', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--base-url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--api-key-env', default='EVIDENCE_API_KEY')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--requests-per-minute', type=float, default=60)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--timeout', type=float, default=90)
    parser.add_argument('--max-tokens', type=int, default=6000)
    parser.add_argument('--disable-thinking', action='store_true', help='Send thinking.type=disabled on providers that support this setting.')
    parser.add_argument('--terminology-requests-per-minute', type=float, default=120)
    parser.add_argument('--terminology-timeout', type=float, default=15)
    parser.add_argument('--terminology-cache-days', type=float, default=30)
    parser.add_argument('--stage', choices=('extract', 'both'), default='both')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--retry-failed', action='store_true')
    mode.add_argument('--retry-mappings', action='store_true', help='Refresh missing codes on saved entities without any model calls.')
    args = parser.parse_args()
    runner = None
    try:
        if not 1 <= args.workers <= 64 or not 1 <= args.max_attempts <= 3:
            raise ValueError('Use 1–64 workers and 1–3 attempts per stage.')
        if not all(math.isfinite(v) and v > 0 for v in (args.requests_per_minute, args.timeout, args.max_tokens, args.terminology_requests_per_minute, args.terminology_timeout, args.terminology_cache_days)):
            raise ValueError('Rate, timeout and token limit must be positive finite values.')
        url = urlparse(args.base_url)
        if url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('localhost', '127.0.0.1', '::1')):
            raise ValueError('Use HTTPS for the provider; HTTP is allowed only for localhost tests.')
        if url.username or url.password or url.query or url.fragment or not url.netloc:
            raise ValueError('Base URL must not contain credentials, query parameters or fragments.')
        if not args.retry_mappings and not os.environ.get(args.api_key_env):
            raise ValueError('Set the API key in the named environment variable before running.')
        articles = read_articles(args.input)
        question = json.loads(args.question.read_text())
        index = json.loads(args.index.read_text()) if args.index else None
        for article in articles:
            build_messages(question, article, 'extract')
        if index is not None:
            build_messages(question, articles[0], 'map', {'picos': {}, 'effects': []}, index)
        runner = BatchRunner(args, question, articles, index)
        runner.initialize()
        return runner.run()
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.exit(2, str(error) + '\n')
    finally:
        if runner:
            runner.close()


if __name__ == '__main__':
    sys.exit(main())
