# Figure 2 reproducibility package

This directory contains the archived inputs, gold standards, model-specific prediction files, and historical evaluation workflow used for the Figure 2 validation analyses.

## Contents

- `figure2_evaluation_historical.ipynb`: public copy of the historical Figure 2 workflow. Saved evaluation outputs are retained. Absolute storage paths and API credentials were replaced, and unrelated downstream analyses were omitted. The evaluation logic was retained.
- `data/`: archived validation inputs, gold standards, and 16 model-specific prediction files.
- `ENVIRONMENT.md`: BERTScore settings and the available dependency information.
- `SHA256SUMS`: integrity checksums for every archived data file.

## Validation datasets

### Validation dataset 1: direction and significance

- Primary gold standard: `data/amia_vali_df_new.csv`
- Historical alternative gold file referenced by some notebook sections: `data/amia_vali_df.csv`
- Source abstracts: `data/sdoh_pmids_100.csv`
- One-step predictions: `data/amia_result_df_1_<model>.csv`
- Two-step predictions: `data/amia_result_df_2_<model>.csv`

### Validation dataset 2: association

- Gold standard: `data/nc_vali_df.csv`
- Gold-standard source workbook: `data/nc_figure1.xlsx`
- Source abstracts: `data/nc_pmids_result.csv`
- One-step predictions: `data/result_df_1_<model>.csv`
- Two-step predictions: `data/result_df_2_<model>.csv`

## Model filename mapping

| Filename token | Model |
|---|---|
| `deepseek` | `deepseek-chat` |
| `glm` | `glm-4-airx` |
| `mini` | `GPT-4o-mini` |
| `qwen` | `qwen-plus` |

The `_1_` files correspond to one-step extraction, and the `_2_` files correspond to two-step extraction. The resulting matrix contains four models, two extraction workflows, and two validation datasets, for a total of 16 prediction files.

## Figure 2 file selection

The Figure 2 DeepSeek two-step result for validation dataset 2 was calculated from `data/result_df_2_deepseek.csv`. The later revised file `result_df_2_deepseek_rev.csv` is intentionally excluded from this package because it produces a different association score and was not the file underlying the Figure 2 value.

## Integrity

The archived data files were copied without changing their contents. Run the following command from this directory to verify them:

```bash
shasum -a 256 -c SHA256SUMS
```

## Reproducibility boundary

The archived notebooks did not contain a complete lockfile for the original 2024 software environment. The evaluation code, BERTScore model, threshold rule, prediction files, gold standards, and saved outputs are available. See `ENVIRONMENT.md` for the preserved settings and version limitations.
