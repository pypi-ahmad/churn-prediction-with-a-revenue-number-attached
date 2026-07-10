# Tutorial: OOF target encoding, nested-style CV, segments

## Target encoding (Telco)

High-cardinality categoricals (`PaymentMethod`, etc.) explode under one-hot or get weak splits.

**Target encoding** replaces a category with the mean churn rate of that category.  
**Danger:** if you compute means on the full train set including the row itself, you **leak** the label.

**Fix — out-of-fold (OOF) encoding:**

1. Split train into K folds.  
2. For each fold, encode validation rows using means from the **other** folds only.  
3. For test, use means from **all** train data (after OOF features for train are done).

Implemented in `OutOfFoldTargetEncoder` (`src/churn_revenue/target_encoding.py`).

## Nested / repeated CV

A single 20% test set can be lucky. v3 runs **repeated stratified K-fold** on the modeling matrix and reports:

- mean ± std of PR-AUC, ROC-AUC, F1@inner-threshold  

This does not replace the fixed holdout used for final tables; it answers:  
“How stable is this model class?”

## Segment reports

Global F1 can hide failure modes:

- Great on month-to-month, weak on two-year contracts  
- Great on low-value, weak on high Monetary  

v3 prints metrics by segment (Contract, tenure band, value quartile).  
**Action:** if high-value segment recall is low, lower threshold or add features for that segment.

## Code map

- `target_encoding.py`, `nested_cv.py`, `segment_report.py`  
- Especially `notebooks/v3/02_telco_v3_awesome.py`  
