# Reported and inferred significance

Keep two independent classifications for each exact contrast, population, outcome, timepoint and adjustment model. The spelling is `inferred`.

| Field | Meaning |
| --- | --- |
| significance_reported | Explicit author statement: positive, negative, or not found |
| significance_reported_quote | Literal abstract statement; empty when absent |
| significance_inferred | Numeric-evidence inference: positive, negative, or not found |
| significance_inferred_basis | 95% CI, P value, or Not found |
| significance_inferred_quote | Literal passage supporting the numeric inference; empty when unknown |
| significance_inferred_alpha | Explicitly stated P-test threshold; null for CI inference or unknown inference |

`positive` means statistically significant; `negative` means nonsignificant. Neither label denotes benefit or harm. A reported HR and CI without an explicit significance statement populate only the inferred channel. Words such as higher, lower or associated alone do not populate reported significance. An explicit nonsignificance statement without numerical evidence populates only reported significance. Do not turn absence of a number into nonsignificance.

For 95% CI inference, assess the null applicable to that exact reported metric: 1 for original-scale ratios and 0 for differences or log ratios when appropriate. Exclusion of the null supports positive; inclusion, including equality at a boundary, supports negative. Do not infer from an unknown confidence level or from a paired descriptive difference alone. Prefer this result's applicable 95% CI before P-based inference.

P-based inference requires an explicitly reported applicable threshold in the evidence quote. Do not silently assume alpha=0.05. Preserve inequality operators and infer only if the entire permitted P range resolves the decision. For example, P<0.05 resolves below alpha=0.05, while P<0.1 does not. A value or inclusive bound exactly on the threshold remains unresolved because rounding and the stated decision convention may matter. Trend, interaction and global tests cannot supply significance for an individual comparison. Source attribution and the threshold's applicability still need semantic review.

## Compatibility and conflicts

The runner checks the legacy significance/basis/quote against these deterministic rules:

1. Both channels known and different: legacy significance=not found, basis=Not found, quote empty; exported significance_conflict=true. Keep both original channels and quotes. NC direction is not found, not no_change.
2. Reported present and no conflict: legacy significance follows reported, basis=Reported statement.
3. Only inferred present: legacy significance follows inferred and its numerical basis.
4. Both unknown: legacy significance=not found.

The CSV displays separate reported/inferred columns (blank for not found), their quotes, numerical inference basis, threshold and a conflict column. The resolved legacy field stays in JSON for NC consumers. Effect direction always retains the point-estimate direction.

Do not relabel the previous mixed significance column as reported. Old results require review of the original text to separate the channels, or fresh extraction. Any re-annotation of saved results must be marked separately from a new model run, preserving the original output. New prompt/schema runs use new output directories; old resume manifests remain incompatible.
