"""Terminology source, ambiguity, caching and recovery regression tests."""
from concurrent.futures import ThreadPoolExecutor
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import terminology as terms

LABEL = 'Hepatocellular carcinoma'
CONCEPT = {'code': 'C0019204', 'name': 'Hepatocellular Carcinoma', 'version': 'test-release', 'active': True,
           'synonyms': [
               {'source': 'MSH', 'code': 'D006528', 'name': LABEL},
               {'source': 'ICD10CM', 'code': 'C22.0', 'name': LABEL},
               {'source': 'ICD10CM', 'code': 'C22.0', 'name': 'Liver cell carcinoma'}]}


def fixture(url, params, requests):
    requests.append({'url': url, 'status': 'ok'})
    if url == terms.MESH + 'descriptor':
        data = []
    elif url == terms.MESH + 'label':
        data = ['Carcinoma, Hepatocellular']
    elif url == terms.NCIM + 'search':
        data = {'total': 1, 'concepts': [copy.deepcopy(CONCEPT)]}
    elif params['sf'] == 'code':
        data = [1, ['C22.0'], None, [['C22.0', 'Liver cell carcinoma']]]
    else:
        data = [0, [], None, []]
    return data, url


class TerminologyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'terms.sqlite3'
        self.resolver = terms.TerminologyResolver(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_crosswalk_requires_source_synonym_and_target_confirmation(self):
        with patch.object(self.resolver, 'get', side_effect=fixture):
            result = self.resolver.lookup(LABEL)
        self.assertEqual([result[v]['code'] for v in terms.VOCABS], ['D006528','C22.0','C0019204'])
        self.assertEqual(result['icd']['code_system'], 'ICD-10-CM')
        self.assertEqual(result['mesh']['match_method'], 'verified_source_synonym')
        self.assertEqual(result['umls']['version_scope'], 'NCI Metathesaurus release')
        self.assertTrue(all(x['status']=='matched' for x in result['checks'].values()))

    def test_cache_deduplicates_concurrency_and_survives_restart(self):
        with patch.object(self.resolver, 'get', side_effect=fixture) as get:
            with ThreadPoolExecutor(max_workers=6) as pool:
                results = list(pool.map(self.resolver.lookup, [LABEL]*6))
            self.assertEqual(get.call_count, 5)
            self.assertEqual(len(results), 6)
        second = terms.TerminologyResolver(self.path)
        with patch.object(second, 'get', side_effect=AssertionError('Cache miss')):
            self.assertEqual(second.lookup(LABEL)['icd']['code'], 'C22.0')

    def test_source_failure_does_not_suppress_other_services(self):
        def get(url, params, requests):
            if url == terms.NCIM+'search': raise OSError('Unavailable')
            if url == terms.MESH+'descriptor': return [{'resource':'http://id.nlm.nih.gov/mesh/D006528','label':LABEL}],url
            return [1,['C22.0'],None,[['C22.0',LABEL]]],url
        with patch.object(self.resolver,'get',side_effect=get):result=self.resolver.lookup(LABEL)
        self.assertEqual(result['checks']['umls']['status'],'lookup_error')
        self.assertEqual(result['mesh']['code'],'D006528')
        self.assertEqual(result['icd']['code'],'C22.0')

    def test_zero_ncim_results_are_not_a_service_error(self):
        def get(url, params, requests):
            if url == terms.NCIM+'search':return {'total':0},url
            if url == terms.MESH+'descriptor':return [],url
            return [0,[],None,[]],url
        with patch.object(self.resolver,'get',side_effect=get):result=self.resolver.lookup('Synthetic absent term')
        self.assertTrue(all(x['status']=='not_found' for x in result['checks'].values()))

    def test_multiple_exact_cuis_do_not_choose_first(self):
        def get(url, params, requests):
            if url == terms.NCIM+'search':
                return {'total':2,'concepts':[CONCEPT,{**CONCEPT,'code':'C9999999'}]},url
            return fixture(url,params,requests)
        with patch.object(self.resolver,'get',side_effect=get):result=self.resolver.lookup(LABEL)
        self.assertIsNone(result['umls']);self.assertIsNone(result['icd'])
        self.assertEqual(result['checks']['umls']['status'],'ambiguous')
        self.assertEqual(len(result['checks']['umls']['candidates']),2)

    def test_target_code_must_be_verified_and_no_broader_code_invented(self):
        def get(url, params, requests):
            if url == terms.ICD and params['sf']=='code':return [1,['C22.1'],None,[['C22.1','Different cancer']]],url
            return fixture(url,params,requests)
        with patch.object(self.resolver,'get',side_effect=get):result=self.resolver.lookup(LABEL)
        self.assertIsNone(result['icd'])
        self.assertEqual(result['checks']['icd']['status'],'ambiguous')
        self.assertEqual(result['checks']['icd']['candidates'],[{'code':'C22.0'}])

    def test_partial_index_preserved_and_review_attached_per_vocabulary(self):
        entity={'label':LABEL,'role':'O','mention':LABEL,'source_quote':LABEL,'concept_id':'local',
                'mesh':None,'umls':None,'icd':{'code':'C22','label':'Existing broader category','relation':'Broader','version':'ICD-10 2019'},'mapping_status':'index'}
        with patch.object(self.resolver,'get',side_effect=fixture):mapped=self.resolver.enrich([entity])[0]
        self.assertEqual(mapped['icd'],entity['icd']);self.assertEqual(mapped['mapping_checks']['icd']['status'],'index')
        self.assertEqual(mapped['mapping_status'],'mapped')
        self.assertEqual(mapped['review_vocabularies'],[])

    def test_normalization_keeps_polarity_and_distinct_labels(self):
        self.assertNotEqual(terms.normalize('HER2+'),terms.normalize('HER2-'))
        self.assertNotEqual(terms.normalize('Type 1 diabetes'),terms.normalize('Type 2 diabetes'))
        self.assertEqual(terms.normalize('  Liver  Cancer '),'liver cancer')

    def test_force_refresh_is_deduplicated_within_run(self):
        with patch.object(self.resolver,'get',side_effect=fixture):self.resolver.lookup(LABEL)
        second=terms.TerminologyResolver(self.path)
        with patch.object(second,'get',side_effect=fixture) as get:
            with ThreadPoolExecutor(max_workers=5) as pool:
                list(pool.map(lambda _:second.lookup(LABEL,refresh=True),range(5)))
            self.assertEqual(get.call_count,5)

    def test_http_429_retries_and_records_each_attempt(self):
        requests = []
        limited = HTTPError(terms.ICD, 429, 'Rate limit', {'Retry-After': '0'}, None)
        with patch.object(self.resolver, 'acquire'), patch.object(terms, 'urlopen', side_effect=[limited, io.StringIO('[]')]):
            data, url = self.resolver.get(terms.ICD, {'terms': 'Example condition'}, requests)
        self.assertEqual(data, [])
        self.assertEqual([x['status'] for x in requests], ['HTTP 429', 'ok'])
        self.assertIn('Example+condition', url)

    def test_error_cache_expires_and_can_recover(self):
        failed = {'query': LABEL, 'checked_at': 'test', 'requests': [],
                  **{v: None for v in terms.VOCABS},
                  'checks': {v: terms.decision('lookup_error') for v in terms.VOCABS}}
        with patch.object(self.resolver, 'fetch', return_value=failed), patch.object(terms.time, 'time', return_value=1000):
            self.resolver.lookup(LABEL)
        second = terms.TerminologyResolver(self.path)
        with patch.object(second, 'fetch', return_value=failed) as fetch, patch.object(terms.time, 'time', return_value=1061):
            second.lookup(LABEL)
            self.assertEqual(fetch.call_count, 1)


if __name__=='__main__':unittest.main()
