# Tutorial: OOF target encoding, nested-style CV, segments

## Target encoding (Telco)

High-cardinality categoricals such as `PaymentMethod` create many one-hot columns and can produce weak splits.

Target encoding replaces a category with its mean churn rate.
If the mean includes the row being encoded, the feature leaks the label.

Use out-of-fold (OOF) encoding:

1. Split train into K folds.  
2. For each fold, encode validation rows using means from the **other** folds only.  
3. For test, use means from **all** train data (after OOF features for train are done).

Implemented in `OutOfFoldTargetEncoder` (`src/churn_revenue/target_encoding.py`).

## Nested-style and repeated CV

A single 20% test set can be lucky. v3 runs **repeated stratified K-fold** on the modeling matrix and reports:

- mean ± std of PR-AUC, ROC-AUC, F1@inner-threshold  

The repeated-CV results supplement the fixed holdout used for final tables. They show how stable the model class is across splits.

## Segment reports

Global F1 can hide weak segments:

- Great on month-to-month, weak on two-year contracts  
- Great on low-value, weak on high Monetary  

v3 prints metrics by segment (Contract, tenure band, value quartile).  
If recall is low for a high-value segment, lower the threshold or add features for that segment.

## Code map

- `target_encoding.py`, `nested_cv.py`, `segment_report.py`  
- Especially `notebooks/v3/02_telco_v3_awesome.py`  
