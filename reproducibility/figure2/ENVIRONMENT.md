# Evaluation environment and BERTScore settings

## Historical evaluation settings

- Package interface: `bert_score.score`
- Language: `lang="en"`
- Model: `model_type="microsoft/deberta-xlarge-mnli"`
- Entity-pair threshold: strict `F1 > 0.8`
- IDF weighting: not enabled in the historical calls
- Baseline rescaling: not enabled in the historical calls
- Matching unit: candidate pairs within each PMID
- Label metrics: macro-averaged precision, recall, and F1 in the corresponding historical evaluation cells

The historical workflow contains more than one exploratory BERTScore calculation direction. The Figure 2 values are preserved in the saved notebook outputs. For example, validation dataset 2 DeepSeek two-step used `result_df_2_deepseek.csv` and produced rounded F1 values of 0.78 for exposure, 0.87 for outcome, and 0.75 for association.

## Dependency information

No exact environment lockfile from the original July 2024 runs was found. Consequently, a fully pinned publication-time environment cannot be stated with certainty.

Two later notebook execution snapshots provide partial information:

| Snapshot | Python | BERTScore | PyTorch | Transformers | pandas | NumPy |
|---|---:|---:|---:|---:|---:|---:|
| Later revision | 3.11 | 0.3.13 | 2.5.1+cu124 | 4.48.2 | 2.2.2 | 1.26.4 |
| Later rerun | 3.12 | 0.3.13 | 2.8.0+cu126 | 4.57.1 | 2.2.2 | 2.0.2 |

These snapshots are documented for transparency and should not be described as the exact original 2024 environment. The Hugging Face model revision was also not pinned in the archived workflow.
