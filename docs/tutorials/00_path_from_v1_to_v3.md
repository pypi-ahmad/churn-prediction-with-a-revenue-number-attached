# Learning path: v1 → v2 → v3

The older notebooks remain the learning baseline. New work is in `notebooks/v3/` and `docs/tutorials/`.

| Stage | Location | Goal |
|-------|----------|------|
| v1 / v2 learning | `notebooks/01_*.ipynb` … `03_*.ipynb` | EDA, LazyPredict, classical tuning, plain and ensemble TabFM, revenue-at-risk |
| v3 | `notebooks/v3/` | Value policies, hybrid TabFM+GBM, multi-window RFM, nested CV, and segments |
| Concept tutorials | `docs/tutorials/` | Explanations for each technique |

## Suggested order

1. Read this file + `01_why_not_just_f1.md`  
2. Run `notebooks/v3/00_improvement_playbook.py` (or convert to ipynb)  
3. Per dataset:  
   - Iranian: `v3/01_iranian_v3_awesome.py`  
   - Telco: `v3/02_telco_v3_awesome.py`  
   - Retail: `v3/03_retail_v3_awesome.py`  
4. Read `02_hybrid_tabfm_gbm.md` and `03_multiwindow_rfm_survival.md` for more detail

## Boundaries

- The project does not edit or delete `notebooks/01_*` … `03_*`; they remain the tutorial baseline.
- v3 does not win every metric. See the README scorecards.
- The project does not include an uplift model because campaign outcome data is unavailable; `p_save` is an approximation.

## How to re-run v3 only

```bash
uv sync
MPLBACKEND=Agg uv run python notebooks/v3/00_improvement_playbook.py
MPLBACKEND=Agg uv run python notebooks/v3/01_iranian_v3_awesome.py
MPLBACKEND=Agg uv run python notebooks/v3/02_telco_v3_awesome.py
MPLBACKEND=Agg uv run python notebooks/v3/03_retail_v3_awesome.py
```
