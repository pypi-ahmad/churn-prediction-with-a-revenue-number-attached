---
type: Workflow
title: Foundation-model benchmark lifecycle
description: Run contract, evaluation boundaries, resource constraints, and completion artifacts for the separate Mitra and Retail TimesFM benchmark.
tags: [benchmark, foundation-models, cuda, operations]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:45.468Z
sources:
  - id: openwiki-source-8335f0fbd415dcbf74d12a96
    resource: repo://docs/foundation-model-benchmark.md
  - id: openwiki-source-a27ff25bcb065550524729a0
    resource: repo://scripts/run_foundation_benchmarks.py
  - id: openwiki-source-8152005220e6048de7e45a44
    resource: repo://src/churn_revenue/foundation.py
generated: { by: "codex", at: "2026-09-23T14:20:45.468Z" }
---

# Entry contract

The benchmark is an explicit, network-using CUDA workflow, not part of the ordinary notebook path. Use the locked environment and check CUDA before starting:

```powershell
uv sync --all-groups
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
uv run --locked python scripts/run_foundation_benchmarks.py --all --output artifacts/foundation-benchmarks
```

The CLI rejects runs without `--all`, has no CPU fallback, and exits before data downloads when CUDA is unavailable. The selected output location must be dedicated to this run: at startup the script removes only its `models/` child, then recreates the output directory.

# Evaluation flow

## Mitra: three churn datasets

The runner fetches Iranian, IBM Telco, and Online Retail II data; it builds stratified 60/20/20 train/validation/test splits. For each dataset, it fits zero-shot Mitra on train and scores validation, selects an F1 threshold there, refits on train plus validation, and evaluates the test split once with that frozen threshold. Fine-tuning is disabled. Training contexts above 500 rows are reduced to a deterministic stratified 500-row sample; the AutoGluon fit also sets its memory-use ratio to 1.15 and clears Python/CUDA caches after prediction.

## TimesFM: Retail activity-risk experiment

The Retail phase aggregates distinct invoice activity into weekly per-customer series. It forecasts 13 weeks on CUDA and converts non-negative forecasts into a risk ranking by negating their sum: lower predicted future activity means a higher raw risk score. Validation uses the 2011-03-01 cutoff; test uses 2011-06-01. The frozen F1 threshold is evaluated once on test. These are ranking scores, not calibrated churn probabilities, and the benchmark does not report Brier score, log loss, or expected value for this phase.

# Completion and troubleshooting

A complete run writes four metric rows to `metrics.csv` only after all three Mitra evaluations and the TimesFM phase return; `run.json` records Torch/CUDA versions and row count. `metrics.csv` is the report of record. If it is missing, treat the run as incomplete and do not report partial console output or leftover model files as results.

Mitra context is capped but has no smaller-context fallback. TimesFM first forecasts batches of four and retries an out-of-memory batch one series at a time; there is no CPU retry. If downloads fail, restore network access; if memory remains insufficient after the documented mitigation, the run is blocked on available host/GPU capacity.

For environment-wide checks and notebook validation, see [Development setup and verification](/openwiki/operations/development-verification.md). The package adapters are described in [Shared modeling and decision primitives](/openwiki/architecture/shared-churn-primitives.md).
