# NC field audit and export contract

Audited 2026-11-05 against the project files listed below. This distinguishes the original notebook's ten-field prompt from the local extended script actually called by `two_step_extraction`. It does not attribute all local extensions to the published article.

## Source and downstream use

- [`notebooks/nc_revision (1).ipynb`](https://github.com/xuanyshi/llm-evidence-triangulation/blob/main/notebooks/nc_revision%20%281%29.ipynb): zero-based cell 75 defines the ten relation fields and the exposure/outcome entity step. Cells 97 and 101 recode nonsignificant associations to `direction=no_change`; cell 97 first excludes unknown significance. Cell 103 groups by design, exposure direction and outcome direction, using participant sums. Cell 130 caps sample sizes at 10,000 in a downstream analysis; this must never change extracted raw N.
- [`notebooks/extraction_matching.py`](https://github.com/xuanyshi/llm-evidence-triangulation/blob/main/notebooks/extraction_matching.py): `prompt_step2_template_new` adds design categories and statistical fields. `two_step_extraction` calls this extended template. Its design names are more detailed than the original MR/RCT/OS categories. Both levels are preserved below.
- [`notebooks/triangulation.py`](https://github.com/xuanyshi/llm-evidence-triangulation/blob/main/notebooks/triangulation.py): `analyze_tri_df` uses design, exposure direction, direction, PMID and participant counts. Participant weighting and tail filtering are downstream choices, not extraction rules.
- Associated [Nature Communications article](https://doi.org/10.1038/s41467-025-62783-x).

Source SHA-256 values at audit:

| File | SHA-256 |
| --- | --- |
| nc_revision (1).ipynb | 30dea3274e4571732b0440fe4445b2777b1da5c82ccf8ce91ca0c3f32c96cd87 |
| extraction_matching.py | ff97de1bf34b2f83679ff697780452b4319be5b9adcc0be11a25e4a6e8b9fd03 |
| triangulation.py | a84560d44a03161d7df35d180eb61479f71324e02d13bb781a9173ab56638a65 |

## Field crosswalk

Every effects row has explicit scope and count fields, even when the value is null. `results.jsonl.data` preserves the model output; `results.jsonl.nc_relations` contains deterministic aliases and derived analysis direction. Both arrays have the same row order and cardinality. There is one CSV evidence table. No separate effect-measure column is introduced.

| NC field | Current raw field / derived export | Required interpretation |
| --- | --- | --- |
| exposure | intervention → exposure | Actual exposure, dose and scale, not an ontology parent |
| exposure_direction | exposure_direction | increased/decreased; not found for unordered or unresolved contrasts |
| outcome | outcome | Exact measured endpoint |
| significance | significance | positive=significant; negative=nonsignificant; unknown=not found |
| direction | derived nc_relations.direction | negative → no_change; positive → raw effect_direction; unknown → not found |
| population_main_condition | population_main_condition | Baseline condition or eligibility; separate from outcome disease |
| comparator | comparator | Actual reference group |
| study_design | study_design plus study_design_detail | Original MR/RCT/OS/META/REVIEW/SR/Others category plus specific analysis method; Not reported when unknown |
| included_studies | included_studies → formatted NC alias | Review-level study counts, with reported total and per-design counts; strict null for primary studies |
| number_of_participants | number_of_participants | Participants belonging to this result, with basis and source quote |
| study_design_category | study_design_category | Extended-script analysis family; null if unestablished |
| statistical_parameter | effect_type → statistical_parameter | HR/OR/RR/etc.; no duplicate measure in the display table |
| statistical_value | effect_value → statistical_value | Original numeric estimate; null for a qualitative result |
| ci_lower | ci_lower | Stated 95% lower bound for that estimate |
| ci_upper | ci_upper | Stated 95% upper bound for that estimate |
| p_value | raw p_value → p_value_raw, p_value, p_value_operator, p_value_bound | Preserve inequalities; exact numeric p_value only for a reported exact value |

The original ten-field prompt used spaces in some field instructions and underscores in its example output; exports use underscores. The extended script labels statistical metrics differently in places (e.g. Beta/MD); retain the metric actually stated and apply explicit metric normalization only in a downstream adapter. Do not infer IRR merely from a ratio-shaped number.

The NC projection keeps the original coarse design in `study_design` for downstream grouping and the extended specific method in `study_design_detail`. It is a documented field crosswalk, not a drop-in reproduction of every historical script's taxonomy or missing-value convention. Unknown values use JSON null / not found, not the string "null".

## Sample-size scope

- `number_of_participants`: the explicitly reported analysis-specific N. `participants_basis=analysis_population` identifies this use. An enrolled N can be used with `participants_basis=enrolled_population` only when clearly applicable to that same analysis and no narrower N is stated.
- `participants_quote`: exact count and population passage. The selected N must also appear in a matching `sample_sizes` item.
- `sample_sizes`: separate counts for whole cohort, analysis population, intervention/comparator groups, cases, controls, events, exposure GWAS, outcome GWAS and other units. Each item carries `number`, `unit`, `population`, `role` and a source quote. Counts are copied, not calculated by summing groups. Event counts, clusters and person-years cannot stand in for participant N. Exposure and outcome GWAS sizes stay separate; there is no invented combined MR sample size.
- `included_studies`: only META/SR/REVIEW; object `{total, by_design:[{study_design,count}], source_quote}`. Preserve the count's overall or outcome-specific scope. Do not confuse number of studies with number of people. A missing count is null.

The notebook example itself illustrates the problem: a cohort has 3,681 participants, a hypertension-incidence analysis has 2,096, and repeated BP measurements have 1,499. The count must follow the actual endpoint and analysis. The historical example has imperfect population-condition placeholders; those are not valid scientific labels to copy.

The validator checks positive integer participant N, literal count tokens, source quote containment, selected-N consistency with its scoped count item, study-count applicability and null conventions. A narrow contrast check also rejects HRs labeled as group comparisons when methods explicitly define them relative to whole-population average risk, and rejects average-reference rows for explicitly high-versus-low questions. These checks do not establish general semantic eligibility. The validator cannot independently prove that a model assigned a valid count to the correct endpoint; retain the source passage for semantic review. It does not impose participant weights or cap large cohorts.

## Additional necessary context

- Supplied study ID, PMID and year: kept from input metadata, not invented by the model. CSV exports PMID/year when supplied. Each result receives `relation_id=<input-id>:relation:<row-number>`; this identifies rows within a run, not invariant identities across reruns with changed row ordering.
- Population, exposure, comparator, outcome, units, follow-up, adjustment/model and source quotes: retained per relation so the estimate remains attributable.
- `effect_direction`: raw point-estimate direction; an HR above 1 stays increase even if its CI includes 1. NC analysis direction is separate and may be no_change. Unknown-significance rows must not count as no_change in triangulation.
- Qualitative results: explicitly reported eligible relationships without an estimate are retained with a null value and source evidence; `result_kind=qualitative` in exports. Background hypotheses are not results. When no eligible relationship is reported, the status is `no_eligible_result`.
- Matching/QC: entity mapping and extraction checks remain distinct from scientific review. A valid row is not automatically a matched, adjudicated ETA input.

Multiple models, subgroups, timepoints or endpoints may reuse the same people. Before participant-weighted synthesis, define the independent evidence unit and select eligible relations or models; never sum all exported N values as if every row described distinct participants. Do not average estimates across incompatible contrasts merely because the study ID matches.

## Version boundary

This schema revision requires fresh extraction into a new run directory. Existing saved results do not acquire sample-size fields through terminology-only refresh. Historical example prompts remain unchanged for reproducibility. New runs use the current generic prompts, not the frozen salt example. Prompt/schema changes are deliberately incompatible with resume of an older run. The website and historical videos are separate deliverables and do not update when this skill changes.
