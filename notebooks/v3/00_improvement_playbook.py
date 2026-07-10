# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python (churn-revenue-project)
#     language: python
#     name: churn-revenue-project
# ---

# %% [markdown]
# # v3 Improvement Playbook (Tutorial)
#
# **This notebook does not replace** `notebooks/01_*` … `03_*`.  
# Those remain the learning baseline (v1/v2). This file is the **map** of what
# v3 adds and *why*.
#
# | Layer | v1/v2 (old notebooks) | v3 (new `notebooks/v3/`) |
# |-------|----------------------|---------------------------|
# | Objective | F1 / PR-AUC at a score threshold | **Net expected value** + top-K budget |
# | Models | GBM + TabFM | **Hybrid meta(GBM, TabFM)** |
# | Features | Snapshot RFM / basic Telco | **Multi-window RFM**, OOF target encoding |
# | Validation | Train/val/test | + **repeated CV** & **segment reports** |
#
# Read the markdown tutorials in `docs/tutorials/` alongside this notebook.

# %%
from pathlib import Path

print("Project tutorials:")
root = Path(__file__).resolve().parents[2] if "__file__" in dir() else Path.cwd()
# when run from repo root via notebooks/v3/...
for base in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    tdir = base / "docs" / "tutorials"
    if tdir.exists():
        for p in sorted(tdir.glob("*.md")):
            print(" -", p.relative_to(base))
        break

print(
    """
HOW TO USE THIS REPO AFTER v3
=============================
1. Keep learning on notebooks/01_iranian_churn.ipynb → 03_...
2. Read docs/tutorials/00_path_from_v1_to_v3.md
3. Run v3 notebooks for state-of-the-art improvements:
     notebooks/v3/01_iranian_v3_awesome.py
     notebooks/v3/02_telco_v3_awesome.py
     notebooks/v3/03_retail_v3_awesome.py
4. Compare tables in README: v1 vs v2 vs v3 (old numbers kept)

KEY IDEAS (one line each)
=========================
• Threshold/F1 ≠ ROI → optimize top-K / EV policy on validation.
• Trees + TabFM err differently → blend probabilities with a meta-learner.
• Single RFM is myopic → multi-window + gap stats + hazard horizons.
• One holdout is fragile → repeated stratified CV mean±std.
• Global metrics hide pain → segment by contract / value quartile.
"""
)
