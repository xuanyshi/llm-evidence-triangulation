# Salt intake and incident hypertension

This optional example illustrates a high-versus-low exposure contrast. It does not set defaults for the generic workflow.

```json
{
  "exposure": "Dietary salt or sodium intake",
  "comparator": "The study's reported low-intake reference group",
  "outcome": "Incident hypertension",
  "outcome_type": "Incidence",
  "contrast": "Highest versus lowest reported intake, or the explicit higher-versus-lower reference"
}
```

Use this question with the generic messages in `../../references/prompts.json`. Use a separate question for sodium-reduction interventions versus usual care when that is the intended comparison; do not silently combine those questions.

The adjacent [prompts.json](prompts.json) and [terminology.json](terminology.json) are unchanged historical snapshots from the original demonstration, which included both high/low intake contrasts and isolated sodium-reduction interventions. Read them only to reproduce that original workflow or inspect its tailored rules. The index contains 92 concepts and is optional for the generic mapping stage. A supplied index never limits recognition of unindexed entities.

Historical execution used `deepseek-flash`, JSON output, temperature 0, max_tokens 6000, and disabled thinking where supported. These settings record the original demonstration rather than prescribe a model for future questions. Its software also contains fixed outcome filters, so a runner must be adapted before processing another question.
