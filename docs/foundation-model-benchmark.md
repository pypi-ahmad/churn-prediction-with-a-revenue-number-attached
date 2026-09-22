# Foundation-model benchmark runbook

This runbook is for maintainers running the CUDA-only foundation-model benchmark. It describes the benchmark script and does not publish performance results. An output directory is incomplete unless a successful run created `metrics.csv`.

## Run the benchmark

Use the locked project environment on Windows AMD64 with a CUDA-capable PyTorch installation. The benchmark downloads public datasets and model weights on its first run.

```powershell
uv sync --all-groups
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run --locked python scripts/run_foundation_benchmarks.py --all --output artifacts/foundation-benchmarks
```

The CUDA check must print `True`. The benchmark has no CPU path and exits before downloading data when CUDA is unavailable.

Use `--output` to place generated artifacts elsewhere. The script removes only the `models` child directory of that location before it starts; choose an output directory that is dedicated to this run.

### Completion criteria

After all four evaluations finish, the output directory contains:

| Path | Meaning |
|---|---|
| `models/` | AutoGluon predictor files created during the Mitra validation and test fits. This directory is recreated at the start of a run. |
| `metrics.csv` | One row for each completed evaluation. It is written only after every benchmark phase has returned. |
| `run.json` | PyTorch version, CUDA version, and the number of metric rows written by the completed run. |

`metrics.csv` is the report of record. If it is absent, do not quote partial console output or model directories as benchmark results. If an earlier `metrics.csv` already exists, remove or archive it before a new run so it cannot be mistaken for fresh output.

## Evaluation protocol

### Mitra zero-shot churn models

The runner evaluates `autogluon/mitra-classifier-2` through AutoGluon's `MITRA` model on the Iranian, IBM Telco, and Online Retail II churn datasets. Fine-tuning is disabled.

For each dataset, the runner makes stratified random 60/20/20 train/validation/test splits using the repository's `RANDOM_STATE`:

1. It fits Mitra on the training split and predicts the validation split.
2. It selects the F1 threshold from validation predictions.
3. It refits Mitra on the combined training and validation data.
4. It evaluates the held-out test split once using the frozen validation threshold.

To control memory, every Mitra fit with more than 500 rows uses a deterministic, stratified 500-row context sample with random state 42. This limits the evaluation context. The adapter asks AutoGluon to allow a maximum memory-use ratio of 1.15 and clears Python and CUDA caches after each fit.

### Retail TimesFM activity-risk backtest

The TimesFM phase uses only Online Retail II. It constructs one weekly activity series per customer from the number of distinct invoices each week before the cutoff. The model forecasts 13 weeks of future activity with `google/timesfm-3.0-pytorch` on CUDA.

The validation cutoff is 2011-03-01 and the test cutoff is 2011-06-01. The script converts each forecast to a score by taking the negative sum of non-negative forecast values. Higher scores therefore indicate lower forecast activity and higher relative risk.

The validation scores select an F1 threshold; the test set is evaluated once with that threshold. These scores rank risk rather than estimate churn probabilities. The report includes precision, recall, F1, ROC AUC, and PR AUC. It does not report calibration, Brier score, log loss, or expected value.

## Command reference

```text
uv run --locked python scripts/run_foundation_benchmarks.py --all [--output PATH]
```

| Argument | Required | Description |
|---|---:|---|
| `--all` | Yes | Runs the three Mitra evaluations and the Retail TimesFM evaluation. The script rejects invocations without it. |
| `--output PATH` | No | Artifact directory. Defaults to `artifacts/foundation-benchmarks`. |

The script writes four rows to `metrics.csv` on a complete run: Iranian Mitra, Telco Mitra, Retail Mitra, and Retail TimesFM raw risk.

## Troubleshooting

| Symptom | Cause confirmed by the code | Action |
|---|---|---|
| `CUDA is required; install the locked CUDA Torch environment first.` | `torch.cuda.is_available()` was false before the run began. | Run the CUDA check above. Install/sync the locked environment on a CUDA-capable Windows host before retrying. |
| Mitra stops for host or GPU memory | Mitra fits use AutoGluon and model artifacts; the code caps context at 500 rows and clears caches between fits, but has no lower-memory fallback. | Close GPU-intensive applications, ensure sufficient free system and GPU memory, then start a fresh dedicated output directory. |
| TimesFM raises an out-of-memory error | The first pass forecasts batches of four customer series. The adapter empties the CUDA cache and retries that batch one series at a time. | If the one-series retry also fails, free GPU memory or use a larger GPU; there is no CPU fallback. |
| No `metrics.csv` after a run | The CSV is written after all phases. A prior phase failed or the run was interrupted. | Inspect the error, correct the environment or data-download issue, and rerun. Do not report partial output. |
| Dataset or weight download fails | The runner reads public UCI/IBM data URLs and the named model checkpoints at runtime. | Restore network access and rerun. Keep downloaded artifacts outside version control. |

## Scope and licensing

Mitra is used here only in zero-shot mode. TimesFM is an exploratory Retail activity-risk signal and is not a calibrated churn model. Review the upstream [Mitra model card](https://huggingface.co/autogluon/mitra-classifier-2) and [TimesFM repository](https://github.com/google-research/timesfm) before redistributing models, weights, or derived work in a commercial setting.

For installation and repository-wide development checks, return to the [README](../README.md). For code-level behavior, see [`scripts/run_foundation_benchmarks.py`](../scripts/run_foundation_benchmarks.py) and [`src/churn_revenue/foundation.py`](../src/churn_revenue/foundation.py).
