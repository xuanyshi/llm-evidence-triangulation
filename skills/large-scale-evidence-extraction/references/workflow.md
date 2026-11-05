# Messages, validation and table construction

## Extraction call

Use the complete `extract` string from `prompts.json` as the system message. Append this user message, substituting the supplied title and abstract verbatim.

```text
Extract PICOS and eligible quantitative findings from this article. Return the populated data object in JSON.
Title: {{article.title}}
Abstract: {{article.abstract}}
```

The prompt requests populated results, not a repeated schema. Validate against the bundled `schema` object. The schema permits null numerical fields, but an eligible effect row must contain a reported numerical effect value. Use an empty effects array when no eligible numerical result exists.

For the frozen sodium question, preserve the paper's actual exposure contrast. Keep the extreme groups when reported, otherwise the explicit higher-versus-lower reference. Retain sodium comparisons within a subgroup; exclude contrasts that simultaneously change sodium and another exposure such as genotype. Preserve each eligible adjustment model and method. Never import sample size, exposure means or effect estimates from a separate mortality/CVD analysis into the hypertension-incidence analysis.

Prefer comparative estimates over duplicate arm rates. Preserve reported percent reductions as percentages. If a ratio and percent reduction express the same finding, retain the ratio. Never invert or calculate an unreported estimate. When all categories have nonreference HRs and nondegenerate CIs, do not invent a lowest-category reference.

## Entity call

Use the complete `map` string as the system message. Serialize a user object with the keys `title`, `abstract`, and `extracted`. The first two contain original study text; `extracted` contains the populated, validated first-call object.

Validate entities against their actual mentions and extracted PICOS roles. Each entity needs an exact, nonempty mention and quote from the title or abstract, with the mention contained in its quote. Use methods/results passages for study facts. Exclude prior-literature background, negated baseline disease, generic control groups, numbers and statistical tests. At most 30 distinct concept-role pairs are requested by the frozen prompt.

## Terminology resolution

For a recognized `concept_id`, join the corresponding concept in `terminology.json` and retain `mesh`, `icd` and `umls` objects, including source URLs, relation and version. A broader mapping does not change the meaning of the extracted concept. Do not reinterpret all mappings as exact.

For an unindexed label, the original cloud implementation performs these requests with URL-encoded parameters.

- MeSH descriptor lookup at `https://id.nlm.nih.gov/mesh/lookup/descriptor`, with `label`, `match=exact`, and `limit=5`. Accept one unambiguous descriptor with its returned identifier and label.
- NCI Metathesaurus lookup at `https://api-evsrest.nci.nih.gov/api/v1/concept/ncim/search`, with `term`, `type=match`, `include=full`, and `pageSize=8`. Accept one CUI whose preferred name or a supported synonym matches after case/punctuation normalization. Prefer a unique preferred-name match. Record that the CUI was obtained via NCI Metathesaurus.

The original fallback performs at most eight distinct label lookups per study and leaves ICD null for unindexed concepts. If lookups fail, keep the entity and null codes. Do not treat a failed lookup as absence of the concept. Code prefixes, approximate name matches or LLM suggestions alone do not establish a vocabulary mapping.

## Source and numerical checks

Check these conditions before a row is treated as validated.

- Required fields, types and controlled values match the schema. Empty PICOS source quotes are acceptable only for a genuinely unreported field marked `Not reported`.
- PICOS quotes occur in the supplied title or abstract. Numerical result and significance quotes occur in the abstract and support this exact comparison, population, timepoint and model. The cloud implementation tolerates case/whitespace differences when locating a quote, then replaces it with the literal source span. It does not permit omitted words or discontinuous quotations.
- Every numerical value is supported by its quoted result passage. CI bounds belong to the same estimate, are ordered, and contain the point estimate. Subgroup, model and arm attribution must agree with the quote. Literal containment and schema validity alone cannot establish that attribution.
- Ratios above/below 1 imply increased/decreased point-estimate direction. Differences use 0. Paired cumulative incidences compare exposure with comparator. A positive reported reduction magnitude denotes decrease. Nonsignificance does not change point-estimate direction to no change.
- For a ratio, a stated 95% CI containing 1 implies nonsignificance; for a difference, use 0. Otherwise an excluding interval implies significance. Use this estimate's CI before a statement about another model. Without a CI, use a clearly linked pairwise P value or direct statement. Trend, interaction and global log-rank P values do not establish pairwise significance.
- Paired incidence rows retain both group percentages. Other metrics have a null `comparator_estimate`. Preserve uncertainty as missing when it is unreported.
- Entity quotes contain the mention, concepts match the extracted fields, and disease absence is not coded as disease presence. Keep different measurement concepts distinct, such as dietary sodium and urinary sodium.

The original cloud code implements a subset of these checks mechanically. Semantic comparison selection and numerical attribution still need direct review against the supplied abstract. Do not claim that automatic checks alone validate every scientific interpretation.

## Corrective calls

When validation fails, retain the previous assistant response and append a user message containing the concrete validation error.

```text
Correct the JSON. {{validation error message}}
```

Allow at most three total attempts per stage, including the first call. Keep the error record if unresolved. Do not silently convert a failed study into an empty-results study. For entity failures, retain only grounded entities and report omitted or unresolved mappings. A failed terminology lookup should not trigger repetition of a successful evidence extraction.

## Rows and traceability

Create one row per eligible effect. Use the effect's population, intervention, comparator, outcome, timepoint and adjustment metadata to prevent collapsing distinct findings. Retain study-level PICOS alongside effect-specific fields in the structured output. Do not carry effect-specific subgroup entities across unrelated rows.

For no-effect studies, show one study-level row with PICOS and missing effect fields when a complete study inventory is requested. Otherwise report the study separately as having no eligible numerical estimate. Never create an estimate row with value zero to represent missingness.

Keep the input record identifier, original abstract, actual model identifier, prompt or prompt hash, source quotes and mapping provenance with the extraction outputs when an auditable run is requested. The frozen vocabulary date and the execution date are different pieces of information.
