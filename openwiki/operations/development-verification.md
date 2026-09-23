---
type: Operations
title: Development setup and verification
description: The repository's uv environment, documented static and notebook checks, runtime prerequisites, and test coverage observed during initialization.
tags: [development, verification, uv, notebooks]
verified:
  - by: openwiki/0.5.2
    at: 2026-09-23T14:20:45.468Z
sources:
  - id: openwiki-source-f317ee207e1653d2033c81a4
    resource: repo://CONTRIBUTING.md
  - id: openwiki-source-8335f0fbd415dcbf74d12a96
    resource: repo://docs/foundation-model-benchmark.md
  - id: openwiki-source-05ccef8d4cf1698187f20464
    resource: repo://pyproject.toml
generated: { by: "codex", at: "2026-09-23T14:20:45.468Z" }
---

# Environment

Use the project-managed `uv` environment and lockfile. The declared Python range is `>=3.13.13,<3.14`; the project environment is constrained to Windows AMD64. The contributor guide's setup is `uv python install 3.13.13`, `uv python pin 3.13.13`, and `uv sync --all-groups`, followed by registering the `churn-revenue-project` Jupyter kernel. A GPU is recommended for TabFM, but the notebooks can use CPU at lower speed.

# Verification that exists

The contributor guide names two repository-wide static checks:

```powershell
uv run ruff check src main.py
uv run ty check src
```

There is no tracked `tests/` directory or test-named Python module in this checkout. Treat the commands above as lint/type checks, not a passing unit-test suite. For a meaningful notebook change, the guide recommends running its paired `.py` source or converting it with Jupytext and executing the notebook with the project kernel; execution downloads public data and can take substantial time.

The foundation-model benchmark is a separate, opt-in GPU verification. Its runbook requires CUDA to be available and says the run is complete only when all four evaluations finish and produce `metrics.csv` and `run.json`; partial console output or model folders alone are not reportable results. See [Foundation-model benchmark lifecycle](/openwiki/workflows/foundation-model-benchmark.md).

# Data and runtime considerations

Notebook data is fetched from public UCI and IBM sources at runtime, while model weights may be downloaded on first use. The full benchmark has no CPU fallback. For the three notebook generations and their different levels of execution cost, see [Notebook generations and churn workflow](/openwiki/workflows/notebook-generations.md). The [repository quickstart](/openwiki/quickstart.md) routes new contributors to setup and runnable entrypoints.
