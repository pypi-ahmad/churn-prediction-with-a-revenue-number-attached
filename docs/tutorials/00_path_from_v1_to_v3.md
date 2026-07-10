# Learning path: v1 → v2 → v3 (what to open, and why)

This project **keeps older notebooks for learning**. New work lives under `notebooks/v3/` and `docs/tutorials/`.

| Stage | Location | Goal |
|-------|----------|------|
| **v1 / v2 learning** | `notebooks/01_*.ipynb` … `03_*.ipynb` | EDA, LazyPredict, classical tuning, plain/ensemble TabFM, revenue-at-risk |
| **v3 awesome** | `notebooks/v3/` | Value policies, hybrid TabFM+GBM, multi-window RFM, nested CV, segments |
| **Concept tutorials** | `docs/tutorials/` | Why each technique exists (read before or after running code) |

## Recommended order

1. Read this file + `01_why_not_just_f1.md`  
2. Run `notebooks/v3/00_improvement_playbook.py` (or convert to ipynb)  
3. Per dataset:  
   - Iranian: `v3/01_iranian_v3_awesome.py`  
   - Telco: `v3/02_telco_v3_awesome.py`  
   - Retail: `v3/03_retail_v3_awesome.py`  
4. Read `02_hybrid_tabfm_gbm.md` and `03_multiwindow_rfm_survival.md` for deep dives  

## What we deliberately did **not** do

- **Edit or delete** `notebooks/01_*` … `03_*` (they remain the tutorial baseline).  
- Claim v3 always wins every metric (see README scorecards).  
- Ship an uplift model without campaign outcome data (we approximate with `p_save`).  

## How to re-run v3 only

```bash
uv sync
MPLBACKEND=Agg uv run python notebooks/v3/00_improvement_playbook.py
MPLBACKEND=Agg uv run python notebooks/v3/01_iranian_v3_awesome.py
MPLBACKEND=Agg uv run python notebooks/v3/02_telco_v3_awesome.py
MPLBACKEND=Agg uv run python notebooks/v3/03_retail_v3_awesome.py
```
