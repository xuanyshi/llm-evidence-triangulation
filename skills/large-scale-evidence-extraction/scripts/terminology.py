"""Source-grounded online terminology lookup with persistent, shared caching."""
from contextlib import closing
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
from http.client import HTTPException
import json
import math
import re
import sqlite3
import threading
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MESH = 'https://id.nlm.nih.gov/mesh/lookup/'
NCIM = 'https://api-evsrest.nci.nih.gov/api/v1/concept/ncim/'
ICD = 'https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search'
VOCABS = ('mesh', 'icd', 'umls')
RESOLVER_VERSION = '1'


def normalize(label):
    # Preserve biomedical punctuation, polarity, numbers and accents.
    return ' '.join(unicodedata.normalize('NFKC', label).casefold().split())


def decision(status, candidates=None, reason=None):
    return {'status': status, 'candidates': candidates or [], 'reason': reason}


def mapping(code, label, system, source, url, version=None, method='exact_label', evidence_url=None):
    return {'code': code, 'label': label, 'code_system': system, 'relation': 'Exact',
            'source': source, 'source_url': url, 'version': version,
            'retrieved_at': datetime.now(timezone.utc).isoformat(),
            'match_method': method, 'evidence_url': evidence_url or url}


def exact_ncim(data, label):
    key = normalize(label)
    candidates = {}
    for concept in data['concepts']:
        if not re.fullmatch(r'C\d{7}', concept.get('code', '')) or concept.get('active') is False:
            continue
        names = [concept.get('name', '')] + [s.get('name', '') for s in concept.get('synonyms', [])]
        if any(normalize(name) == key for name in names):
            candidates[concept['code']] = concept
    return list(candidates.values())


def source_codes(concept, source, label, pattern):
    # Require the query itself to be a synonym in the target vocabulary.
    return sorted({s['code'] for s in concept.get('synonyms', [])
                   if s.get('source') == source and normalize(s.get('name', '')) == normalize(label)
                   and re.fullmatch(pattern, s.get('code', ''))})


class TerminologyResolver:
    def __init__(self, cache_path, rpm=120, timeout=15, attempts=2, cache_days=30, stop=None):
        self.cache_path = cache_path
        self.timeout, self.attempts, self.cache_days = timeout, attempts, cache_days
        self.stop = stop or threading.Event()
        self.interval = 60 / rpm
        self.next_start = 0
        self.rate_lock = threading.Lock()
        self.locks = [threading.Lock() for _ in range(64)]
        self.refreshed = set()
        with closing(sqlite3.connect(cache_path, timeout=30)) as db, db:
            db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, saved REAL, result TEXT)')

    def acquire(self):
        with self.rate_lock:
            now = time.monotonic()
            start = max(now, self.next_start)
            self.next_start = start + self.interval
        if self.stop.wait(max(0, start - now)):
            raise InterruptedError('Terminology lookup interrupted.')

    def get(self, url, params, requests):
        address = url + '?' + urlencode(params)
        for attempt in range(self.attempts):
            self.acquire()
            entry = {'url': address, 'attempt': attempt + 1}
            requests.append(entry)
            try:
                request = Request(address, headers={'Accept': 'application/json', 'User-Agent': 'EvidenceExtraction/1.0'})
                with urlopen(request, timeout=self.timeout) as response:
                    data = json.load(response)
                entry.update(status='ok', response_sha256=hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest())
                return data, address
            except HTTPError as error:
                entry['status'] = f'HTTP {error.code}'
                if error.code not in (408, 429) and error.code < 500:
                    raise
                retry = error.headers.get('Retry-After')
                try:
                    delay = float(retry)
                except (ValueError, TypeError):
                    try:
                        delay = parsedate_to_datetime(retry).timestamp() - time.time()
                    except (ValueError, TypeError, OverflowError):
                        delay = 2 ** attempt
                if not math.isfinite(delay):
                    delay = 2 ** attempt
            except (URLError, OSError, HTTPException, ValueError) as error:
                entry['status'] = type(error).__name__
                delay = 2 ** attempt
            if attempt + 1 < self.attempts:
                self.stop.wait(max(0, min(delay, 60)))
        raise OSError('Terminology service unavailable after retries.')

    def lookup(self, label, refresh=False):
        key = RESOLVER_VERSION + '|' + normalize(label)
        lock = self.locks[int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % len(self.locks)]
        with lock:
            refresh = refresh and key not in self.refreshed
            with closing(sqlite3.connect(self.cache_path, timeout=30)) as db:
                row = db.execute('SELECT saved,result FROM cache WHERE key=?', (key,)).fetchone()
            if row:
                result = json.loads(row[1])
                has_error = any(v['status'] == 'lookup_error' for v in result['checks'].values())
                ttl = 60 if has_error else self.cache_days * 86400
                if not refresh and time.time() - row[0] < ttl:
                    return result
            result = self.fetch(label)
            with closing(sqlite3.connect(self.cache_path, timeout=30)) as db, db:
                db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)', (key, time.time(), json.dumps(result)))
            self.refreshed.add(key)
            return result

    def fetch(self, label):
        result = {v: None for v in VOCABS}
        result.update(query=label, checked_at=datetime.now(timezone.utc).isoformat(), requests=[],
                      checks={v: decision('not_found') for v in VOCABS})
        requests = result['requests']
        # Each source is independent. Failure in one never suppresses another.
        try:
            data, url = self.get(MESH + 'descriptor', {'label': label, 'match': 'exact', 'limit': 20}, requests)
            if not isinstance(data, list):
                raise ValueError('Malformed MeSH response.')
            matches = {c['resource'].split('/')[-1]: c for c in data
                       if normalize(c.get('label', '')) == normalize(label)
                       and re.fullmatch(r'D\d+', c.get('resource', '').split('/')[-1])}
            if len(matches) == 1 and len(data) < 20:
                code, concept = next(iter(matches.items()))
                result['mesh'] = mapping(code, concept['label'], 'MeSH', 'NLM MeSH RDF', MESH.replace('lookup/', '') + code, evidence_url=url)
                result['checks']['mesh'] = decision('matched')
            elif data:
                result['checks']['mesh'] = decision('ambiguous', [{'code': k, 'label': v['label']} for k, v in matches.items()], 'No unique exact descriptor.')
        except (OSError, HTTPException, ValueError, KeyError, TypeError, AttributeError) as error:
            result['checks']['mesh'] = decision('lookup_error', reason=type(error).__name__)

        concept = None
        ncim_url = None
        try:
            data, ncim_url = self.get(NCIM + 'search', {'term': label, 'type': 'match', 'include': 'full', 'pageSize': 50}, requests)
            if isinstance(data, dict) and data.get('total') == 0 and not data.get('concepts'):
                data['concepts'] = []
            if not isinstance(data, dict) or not isinstance(data.get('concepts'), list):
                raise ValueError('Malformed NCIm response.')
            matches = exact_ncim(data, label)
            candidates = [{'code': c['code'], 'label': c['name']} for c in matches]
            if len(matches) == 1 and data.get('total', len(data['concepts'])) <= len(data['concepts']):
                concept = matches[0]
                result['umls'] = mapping(concept['code'], concept['name'], 'UMLS CUI', 'NCI Metathesaurus',
                                         NCIM + concept['code'] + '?include=full', concept.get('version'),
                                         'exact_name_or_source_synonym', ncim_url)
                result['umls']['version_scope'] = 'NCI Metathesaurus release'
                result['checks']['umls'] = decision('matched')
            elif matches or data.get('total', 0) > len(data['concepts']):
                result['checks']['umls'] = decision('ambiguous', candidates, 'Multiple exact concepts or truncated search.')
        except (OSError, HTTPException, ValueError, KeyError, TypeError, AttributeError) as error:
            result['checks']['umls'] = decision('lookup_error', reason=type(error).__name__)

        try:
            data, url = self.get(ICD, {'terms': label, 'sf': 'name', 'df': 'code,name', 'maxList': 50}, requests)
            if not isinstance(data, list) or len(data) < 4 or not isinstance(data[3], list):
                raise ValueError('Malformed ICD-10-CM response.')
            matches = {row[0]: row[1] for row in data[3] if normalize(row[1]) == normalize(label)}
            if len(matches) == 1 and data[0] <= len(data[3]):
                code, name = next(iter(matches.items()))
                result['icd'] = mapping(code, name, 'ICD-10-CM', 'NLM Clinical Tables / CMS', url)
                result['checks']['icd'] = decision('matched')
            elif data[0]:
                result['checks']['icd'] = decision('ambiguous', [{'code': row[0], 'label': row[1]} for row in data[3]][:20], 'No unique exact diagnosis name.')
        except (OSError, HTTPException, ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            result['checks']['icd'] = decision('lookup_error', reason=type(error).__name__)

        if concept:
            self.crosswalk(result, concept, ncim_url)
        return result

    def crosswalk(self, result, concept, ncim_url):
        # Source-specific synonym codes are checked against the target service.
        for vocab, source, pattern in [('mesh', 'MSH', r'[DC]\d+'), ('icd', 'ICD10CM', r'[A-Z]\d[A-Z0-9](?:\.[A-Z0-9]{1,4})?')]:
            if result[vocab]:
                continue
            codes = source_codes(concept, source, result['query'], pattern)
            if len(codes) > 1:
                result['checks'][vocab] = decision('ambiguous', [{'code': c} for c in codes], 'Multiple target codes for the exact source synonym.')
                continue
            if not codes:
                continue
            code = codes[0]
            try:
                if vocab == 'mesh':
                    data, url = self.get(MESH + 'label', {'resource': 'http://id.nlm.nih.gov/mesh/' + code}, result['requests'])
                    if not isinstance(data, list) or len(data) != 1:
                        raise ValueError('Descriptor label not uniquely confirmed.')
                    name = data[0] if isinstance(data[0], str) else data[0]['label']
                    mapped = mapping(code, name, 'MeSH', 'NLM MeSH RDF with NCIm source synonym',
                                     'https://id.nlm.nih.gov/mesh/' + code, method='verified_source_synonym', evidence_url=ncim_url)
                else:
                    data, url = self.get(ICD, {'terms': code, 'sf': 'code', 'df': 'code,name', 'maxList': 50}, result['requests'])
                    exact = [row for row in data[3] if row[0] == code]
                    if len(exact) != 1:
                        raise ValueError('ICD-10-CM code not uniquely confirmed.')
                    name = exact[0][1]
                    # Confirm the target's current label is also in this NCIm concept.
                    aliases = [s.get('name', '') for s in concept.get('synonyms', []) if s.get('source') == source and s.get('code') == code]
                    if normalize(name) not in {normalize(x) for x in aliases}:
                        raise ValueError('Target label and source synonym crosswalk disagree.')
                    mapped = mapping(code, name, 'ICD-10-CM', 'NLM Clinical Tables / CMS with NCIm source synonym',
                                     url, method='verified_source_synonym', evidence_url=ncim_url)
                mapped['target_verification_url'] = url
                mapped['crosswalk_version'] = concept.get('version')
                result[vocab] = mapped
                result['checks'][vocab] = decision('matched')
            except ValueError as error:
                result['checks'][vocab] = decision('ambiguous', [{'code': code}], str(error))
            except (OSError, HTTPException, KeyError, IndexError, TypeError, AttributeError) as error:
                result['checks'][vocab] = decision('lookup_error', [{'code': code}], type(error).__name__)

    def enrich(self, entities, refresh=False):
        enriched = []
        refreshed = set()
        for original in entities:
            entity = dict(original)
            missing = [v for v in VOCABS if not entity.get(v)]
            checks = {v: decision('index' if original.get('mapping_status') == 'index' else 'matched') for v in VOCABS if entity.get(v)}
            if missing:
                key = normalize(entity['label'])
                result = self.lookup(entity['label'], refresh=refresh and key not in refreshed)
                refreshed.add(key)
                conflict = entity.get('umls') and result['umls'] and entity['umls']['code'] != result['umls']['code']
                for vocab in missing:
                    if conflict:
                        checks[vocab] = decision('ambiguous', reason='Existing CUI conflicts with the label lookup.')
                    else:
                        entity[vocab] = result[vocab]
                        checks[vocab] = result['checks'][vocab]
                entity['lookup_provenance'] = {k: result[k] for k in ('query', 'checked_at', 'requests')}
            entity['mapping_checks'] = checks
            entity['review_vocabularies'] = [v for v in VOCABS if not entity.get(v)]
            entity['mapping_status'] = 'needs_review' if entity['review_vocabularies'] else 'mapped'
            enriched.append(entity)
        return enriched
