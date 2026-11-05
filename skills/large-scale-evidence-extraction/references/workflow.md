# Research question, messages and validation

## Research question object

`exposure` and `outcome` are required nonempty strings. Optional fields are nonempty strings describing restrictions supplied by the user. Do not add implicit restrictions.

```json
{
  "exposure": "The exposure or intervention requested by the user",
  "outcome": "The measured endpoint requested by the user"
}
```

Optional keys are `comparator`, `population`, `outcome_type`, `contrast`, `follow_up`, and `study_designs`. Examples of a contrast are a per-unit increment, current versus never exposure, or an intervention versus placebo. Outcome type can distinguish incidence, recurrence, mortality, survival duration, prevalence or continuous change. Study-design restrictions can be expressed as text such as `Randomized trials only`. Keep each distinct research question attached to its own results.

## Model messages

For extraction, the system message is `prompts.json.extract`. The user message is JSON containing `research_question`, `article` (title and abstract), and `schema`. For entity recognition, the system message is `prompts.json.map`; the user JSON contains the same question and article, `extracted` (the populated validated first-call object), and `terminology_index` (ID/label entries, or an empty list).

Build either message pair with the standard-library helper. Paths below are relative to the skill folder. Generated messages go to standard output and can be supplied to the selected model interface.

```sh
python3 scripts/prepare_messages.py --stage extract --question question.json --article article.json
python3 scripts/prepare_messages.py --stage map --question question.json --article article.json --extraction extraction.json
```

The optional `--index path/to/index.json` on the mapping stage accepts an object with a `concepts` array of `{id,label,...}` objects. The helper sends only IDs and labels to the model. Keep the complete index for subsequent code lookup. The helper checks input shape and required question fields; it does not execute or validate an extraction. For a batch, prepare and process each article separately with the same question and retain input identifiers outside the model payload.

## Comparison selection

Keep numerical estimates and directly reported qualitative relationships matching the current question and the actual reference groups. Restrict to extreme exposure categories only when requested. Preserve continuous exposure increments, doses, units and transformations. Keep categorical contrasts without imposing an exposure ordering. Preserve relevant distinct methods, adjustment models, follow-up times and within-subgroup comparisons. Exclude a joint exposure contrast only when it changes an unrelated factor outside the requested question.

Never transfer estimates, population sizes or exposure means across subcohorts or outcomes. A whole-population-average reference remains that reference. A range spanning several intervention arms is not an arm-specific estimate. When the requested endpoint is a composite, include individual components only if the user permits them.

Deduplicate a comparative estimate and descriptive group rates only when they express the same population, contrast, endpoint and timepoint. Do not discard distinct comparisons because some other result in the article has a ratio. Preserve every eligible point estimate as originally reported, including a positive reduction magnitude, an explicitly signed coefficient, or a logarithmic estimate. Do not invert, exponentiate or convert units.

The schema represents numerical effects, paired cumulative incidence percentages and explicit qualitative relationships. Qualitative rows have null effect_value and supported direction or significance. Diagnostic accuracy, prevalence pairs, descriptive group means or unassigned ranges may require an extension for a particular request. Explain that specific limitation and make an explicit schema extension when needed; do not mislabel them as incidence or manufacture an estimate. Keep extensions documented with that run.

## Terminology resolution

Use an index entry only when its named concept matches the entity. Preserve its MeSH/ICD/UMLS mapping objects, exact or broader relation, version and source URL. Unindexed entities remain eligible for extraction and mapping. The example's vocabulary is optional and does not define allowable concepts or outcomes.

For unindexed terms and missing vocabulary fields in indexed entities, the runner automatically performs independent exact label searches with URL-encoded parameters. [terminology-mapping.md](terminology-mapping.md) describes caching, source verification and review states.

- MeSH descriptor lookup at `https://id.nlm.nih.gov/mesh/lookup/descriptor`, with `label`, `match=exact`, and `limit=5`. Accept an unambiguous returned descriptor and retain its identifier and label.
- NCI Metathesaurus lookup at `https://api-evsrest.nci.nih.gov/api/v1/concept/ncim/search`, with `term`, `type=match`, `include=full`, and `pageSize=8`. Prefer a unique preferred-name match or an unambiguous supported synonym. Record that the CUI was obtained via NCI Metathesaurus.
- For new ICD mappings, query the NLM Clinical Tables ICD-10-CM service. Accept a unique exact diagnosis label or a unique exact ICD10CM source synonym from NCIm whose code and current label can be confirmed in Clinical Tables. Record `code_system=ICD-10-CM`. Retain existing ICD-10 index mappings with their original version. A request specifically requiring WHO ICD-10 or another national modification needs that vocabulary's verified index or a separate adapter; do not substitute ICD-10-CM silently.

Read relevant source definitions when homonyms, subtypes or mapping relations are ambiguous. Synonym resolution must preserve the endpoint and concept meaning. A verified exact target-vocabulary synonym can resolve a label absent from that vocabulary's preferred-name search. Multiple exact CUIs or target codes remain ambiguous. Keep unmapped labels with null codes if a source is inaccessible or no mapping is established. Do not force medications, exposures or demographics into disease codes. The runner deduplicates label queries across workers, persists results and independently limits terminology request starts.

## Source and numerical validation

Validate the populated object against `prompts.json.schema`. Schema validity alone does not establish scientific attribution. Check the following conditions against each original abstract.

- Required fields and controlled values match the schema. An absent PICOS field is `Not reported` with an empty source quote. Quantitative rows have a reported number; qualitative rows retain an explicit direction or significance statement with a null value. Validate the NC and scoped sample-size fields described in [nc-fields.md](nc-fields.md).
- PICOS quotes occur in the title or abstract. Effect and significance quotes occur in the abstract and support the exact comparison, population, endpoint, timepoint and model. If case/whitespace alignment is used, replace the stored quote with the literal matched source span; omitted words or discontinuous text are invalid.
- Values and interval endpoints match their quoted result. Confidence limits belong to the same estimate, scale and stated confidence level. CI fields are reserved for 95% intervals; preserve other levels as `other_uncertainty`. Check bound ordering and consistency with the point estimate; retain and flag an inconsistency present in the source rather than silently repairing it.
- Original-scale HR/OR/RR estimates use null=1. Differences, log ratios, correlations and signed coefficients use null=0 only when their reported metric supports that interpretation. Determine direction from the reported contrast orientation. For paired incidence, compare the two percentages. A reported positive reduction magnitude denotes decrease. Unestablished direction remains `not found`.
- For the applicable null, a stated 95% CI excluding it supports significance, and an interval including it supports nonsignificance. Without an applicable interval, use a directly linked P value or statement for the exact result, respecting the reported threshold or multiple-testing adjustment. A P value at an ambiguous threshold remains unresolved. Trend, interaction and global tests cannot establish pairwise significance. Preserve CI-based and multiplicity-adjusted conclusions separately if they differ; describe the discrepancy in `other_uncertainty` and apply an explicitly requested inferential criterion consistently.
- Significance does not change point-estimate direction. Clinical desirability is separate from a measured endpoint's increase/decrease. Preserve distinction between an event hazard and a time-to-event duration.
- Keep effect-specific metadata when subgroup/model results differ. A statement about an adjusted association cannot change the significance assigned to a different unadjusted number.
- Paired-incidence rows retain both percentages. Other metrics retain a null `comparator_estimate` unless an explicit schema extension was made. Preserve the original units for differences and coefficients.
- Entity quotes contain their mentions. Concept identities agree with the focused fields, and negated baseline disease is not coded as present disease. The same concept can occupy multiple roles when explicitly supported.

## Corrections and output

On validation failure, retain the prior assistant response and append `Correct the JSON. ` followed by the specific error. Allow at most three total attempts per stage. Record unresolved errors. Keep valid evidence extraction when only terminology retrieval fails. Omitted invalid entities and unmapped concepts must remain visible in the run's status.

Create one table row per eligible effect, preserving its own P/I/C/O, method, model and timepoint. Retain study-level PICOS in structured output. For a complete study inventory, no-effect studies can occupy a PICOS-only row with blank effect fields; record them as no eligible result. Do not turn missingness into zero. Failed extraction is a distinct status.

For auditable runs, retain original inputs, question object, actual model, exact messages or their hashes, source quotes and terminology provenance. The question must accompany exports so results from different questions cannot be confused.
