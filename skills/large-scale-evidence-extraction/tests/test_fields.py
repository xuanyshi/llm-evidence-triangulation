"""NC field regressions using synthetic, explicitly scoped source passages."""
import copy
import json
import unittest
from test_run_batch import ABSTRACT, result
import run_batch as batch


class FieldTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((batch.SKILL_ROOT/'references/prompts.json').read_text())['schema']
        self.data = result()
        self.effect = self.data['effects'][0]
        self.article = {'title': 'Synthetic study', 'abstract': ABSTRACT}

    def check(self):
        return batch.validate_extraction(self.data, self.article, self.schema)

    def sample(self, role, number, quote, unit='participants'):
        self.article['abstract'] += ' ' + quote
        item = {'role': role, 'number': number, 'unit': unit, 'population': role, 'source_quote': quote}
        self.effect['sample_sizes'].append(item)
        return item

    def test_analysis_n_distinct_from_cohort(self):
        self.sample('cohort_total', 3681, 'We enrolled 3,681 people.')
        item = self.sample('analysis_population', 2096, 'The incident-outcome analysis included 2096 people.')
        self.effect.update(number_of_participants=2096, participants_basis='analysis_population', participants_quote=item['source_quote'])
        self.check()
        self.effect['number_of_participants'] = 96
        with self.assertRaisesRegex(ValueError, 'explicitly stated'): self.check()

    def test_missing_n_not_zero_and_counts_not_summed(self):
        self.sample('cases', 100, 'There were 100 cases.')
        self.sample('controls', 200, 'There were 200 controls.')
        self.check()
        self.assertIsNone(self.effect['number_of_participants'])
        self.effect.update(number_of_participants=300, participants_basis='analysis_population', participants_quote='There were 200 controls.')
        with self.assertRaisesRegex(ValueError, 'explicitly stated'): self.check()

    def test_events_and_clusters_not_participants(self):
        item = self.sample('events', 40, 'We observed 40 events.', 'events')
        self.sample('other', 10, 'There were 10 clusters.', 'clusters')
        self.check()
        self.effect.update(number_of_participants=40, participants_basis='analysis_population', participants_quote=item['source_quote'])
        with self.assertRaisesRegex(ValueError, 'scoped'): self.check()

    def test_mr_keeps_gwas_counts_separate(self):
        self.effect['study_design'] = 'MR'
        self.sample('exposure_gwas', 10000, 'Exposure GWAS included 10,000 participants.')
        self.sample('outcome_gwas', 25000, 'Outcome GWAS included 25,000 participants.')
        self.check()
        self.assertIsNone(self.effect['number_of_participants'])

    def test_meta_study_counts_and_primary_null(self):
        quote = 'We included 12 studies: 4 RCTs and 8 cohort studies.'
        self.article['abstract'] += ' ' + quote
        self.effect.update(study_design='META', included_studies={'total':12,'by_design':[{'study_design':'RCT','count':4},{'study_design':'OS','count':8}],'source_quote':quote})
        self.check()
        projection = batch.nc_relation('x',1,self.effect)
        self.assertEqual(projection['included_studies'], '12 total; 4 RCT; 8 OS')
        self.effect['study_design'] = 'RCT'
        with self.assertRaisesRegex(ValueError, 'primary'): self.check()

    def test_qualitative_relationship_retained(self):
        quote = 'There was no significant association between exposure X and event Y.'
        self.article['abstract'] += ' ' + quote
        self.effect.update(effect_type='Not reported', effect_value=None, ci_lower=None, ci_upper=None,
            effect_direction='no_change', significance='negative', significance_basis='Reported statement',
            significance_quote=quote, source_quote=quote)
        self.check()
        projection = batch.nc_relation('x',1,self.effect)
        self.assertEqual(projection['result_kind'], 'qualitative')
        self.assertIsNone(projection['statistical_value'])
        self.assertEqual(projection['direction'], 'no_change')

    def test_nc_direction_and_p_bounds_preserve_raw(self):
        quote = 'Event Y HR 1.5 (95% CI 0.8 to 1.8), P=0.2.'
        self.article['abstract'] += ' ' + quote
        self.effect.update(ci_lower=.8, significance='negative', significance_quote=quote, source_quote=quote, p_value='P=0.2')
        self.check()
        original = copy.deepcopy(self.effect)
        relation = batch.nc_relation('x',1,self.effect)
        self.assertEqual(relation['effect_direction'], 'increase')
        self.assertEqual(relation['direction'], 'no_change')
        self.assertEqual(relation['p_value'], .2)
        self.assertEqual(self.effect, original)
        self.effect['p_value'] = '<0.001'
        relation = batch.nc_relation('x',1,self.effect)
        self.assertIsNone(relation['p_value'])
        self.assertEqual(relation['p_value_bound'], .001)
        self.assertEqual(relation['p_value_operator'], '<')
        self.effect['significance'] = 'not found'
        self.assertEqual(batch.nc_relation('x',1,self.effect)['direction'], 'not found')

    def test_explicit_population_reference_cannot_be_low_group(self):
        self.article['abstract'] += ' Hazard ratios express the risk in groups relative to average risk in the whole study population.'
        self.effect['comparator'] = 'Low group'
        with self.assertRaisesRegex(ValueError, 'average risk'): self.check()
        self.effect['comparator'] = 'Average risk in the whole study population'
        self.check()
        with self.assertRaisesRegex(ValueError, 'ineligible'):
            batch.validate_extraction(self.data, self.article, self.schema, {'contrast':'Highest versus lowest'})

    def test_required_field_omission_rejected(self):
        del self.effect['number_of_participants']
        with self.assertRaises(batch.jsonschema.ValidationError): self.check()

    def test_condition_requires_source(self):
        self.effect.update(population_main_condition='Cancer', population_condition_quote='Adults had cancer.')
        with self.assertRaisesRegex(ValueError, 'exact source'): self.check()


if __name__ == '__main__':
    unittest.main()
