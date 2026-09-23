---
type: Reference
title: Repository quickstart
description: Choose a notebook path, prepare the locked environment, or jump to the right architecture, modeling, benchmark, and verification notes.
tags: [quickstart, navigation, churn, development]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:45.468Z
sources:
  - id: openwiki-source-f317ee207e1653d2033c81a4
    resource: repo://CONTRIBUTING.md
  - id: openwiki-source-8335f0fbd415dcbf74d12a96
    resource: repo://docs/foundation-model-benchmark.md
  - id: openwiki-source-23775c3de52f3ab95a13cb8b
    resource: repo://README.md
  - id: openwiki-source-fedef29342cd5f0e83e7fee0
    resource: repo://src/churn_revenue/__init__.py
generated: { by: "codex", at: "2026-09-23T14:20:45.468Z" }
---

# Start here

This repository teaches churn modeling and adds a revenue/value view to customer prioritization. The easiest learning path is the Iranian Telecom notebook, followed by IBM Telco and Online Retail II. The v3 notebooks are a separate, additive path for hybrid models, budget-aware contact policies, repeated CV, and segment analysis. Start with the `.ipynb` for interactive work or its neighboring Jupytext `.py` source for script debugging.

## Prepare the environment

From the repository root in PowerShell:

```powershell
uv sync --all-groups
uv run python -m ipykernel install --user --name churn-revenue-project --display-name "Python (churn-revenue-project)"
uv run jupyter lab
```

Open `notebooks/01_iranian_churn.ipynb` and select the `churn-revenue-project` kernel. This and the other notebook workflows download public data at runtime; TabFM is slower and more memory-sensitive without a GPU.

For quick script debugging:

```powershell
$env:MPLBACKEND = "Agg"
uv run python notebooks/01_iranian_churn.py
```

## Route by goal

| Goal | Start with |
|---|---|
| Understand the repository boundaries | [Repository systems and boundaries](/openwiki/architecture/repository-systems.md) |
| Follow the dataset pipelines and compare v1/v2 with v3 | [Notebook generations and churn workflow](/openwiki/workflows/notebook-generations.md) |
| Reuse model, feature, metric, or contact-policy helpers | [Shared modeling and decision primitives](/openwiki/architecture/shared-churn-primitives.md) |
| Run the CUDA-only Mitra/TimesFM experiment | [Foundation-model benchmark lifecycle](/openwiki/workflows/foundation-model-benchmark.md) |
| Check static analysis, notebook reruns, and actual test coverage | [Development setup and verification](/openwiki/operations/development-verification.md) |

The project documents `ruff check` and `ty check` as static checks; this checkout has no dedicated tracked test suite. Revenue-at-risk is exposure associated with flagged customers, while v3 expected-value policies use an assumed save rate. Neither is evidence of causal campaign savings.
