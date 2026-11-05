"""Keep author statements separate from numeric inference and NC resolution."""
import json
import unittest
from test_run_batch import result, ABSTRACT
import run_batch as batch


class SignificanceTests(unittest.TestCase):
    def setUp(self):
        self.data = result()
        self.e = self.data['effects'][0]
        self.article = {'title':'Synthetic study','abstract':ABSTRACT}
        self.schema = json.loads((batch.SKILL_ROOT/'references/prompts.json').read_text())['schema']

    def validate(self):
        return batch.validate_extraction(self.data, self.article, self.schema)

    def resolve(self):
        value, basis, quote, _ = batch.significance_resolution(self.e)
        self.e.update(significance=value, significance_basis=basis, significance_quote=quote)

    def test_ci_alone_does_not_fill_reported(self):
        self.validate()
        self.assertEqual(self.e['significance_reported'], 'not found')
        self.assertEqual(self.e['significance_inferred'], 'positive')

    def test_conflict_preserves_both_and_blocks_nc_direction(self):
        quote = 'After multiplicity correction, the association was not statistically significant.'
        self.article['abstract'] += ' ' + quote
        self.e.update(significance_reported='negative', significance_reported_quote=quote)
        self.resolve()
        self.validate()
        n = batch.nc_relation('x', 1, self.e)
        self.assertTrue(n['significance_conflict'])
        self.assertEqual(n['direction'], 'not found')
        self.assertEqual(n['significance_reported'], 'negative')
        self.assertEqual(n['significance_inferred'], 'positive')
        self.assertEqual(n['effect_direction'], 'increase')

    def test_agreement_retains_separate_quotes(self):
        quote = 'This association was statistically significant.'
        self.article['abstract'] += ' ' + quote
        self.e.update(significance_reported='positive', significance_reported_quote=quote)
        self.resolve()
        self.validate()
        self.assertEqual(self.e['significance_basis'], 'Reported statement')
        self.assertFalse(batch.significance_resolution(self.e)[3])
        self.assertNotEqual(self.e['significance_reported_quote'], self.e['significance_inferred_quote'])

    def test_p_requires_reported_alpha_and_resolved_bound(self):
        quote = 'P<0.01; the significance threshold was 0.05.'
        self.article['abstract'] += ' ' + quote
        self.e.update(ci_lower=None, ci_upper=None, p_value='<0.01', significance_inferred_basis='P value',
                      significance_inferred_quote=quote, significance_inferred_alpha=.05)
        self.resolve()
        self.validate()
        self.e['significance_inferred_alpha'] = None
        with self.assertRaisesRegex(ValueError, 'reported threshold'): self.validate()
        self.e['significance_inferred_alpha'] = .05
        self.e['p_value'] = '<0.1'
        with self.assertRaisesRegex(ValueError, 'resolves significance'): self.validate()

    def test_p_boundaries(self):
        for raw, expected in [('<0.05','positive'),('<=0.05','not found'),('0.05','not found'),
                              ('>0.05','negative'),('>=0.05','not found'),('0.2','negative'),('<0.1','not found')]:
            self.assertEqual(batch.p_significance(raw,.05), expected, raw)

    def test_unknown_channels_have_empty_quotes(self):
        self.e.update(significance_reported_quote='Invented quote')
        with self.assertRaisesRegex(ValueError, 'empty quote'): self.validate()


if __name__ == '__main__':
    unittest.main()
