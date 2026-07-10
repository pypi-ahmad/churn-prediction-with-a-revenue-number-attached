# Tutorial: Hybrid TabFM + gradient boosting

## Why combine them?

| Model family | Strengths | Weaknesses |
|--------------|-----------|------------|
| **GBM** (XGB/LGBM/CatBoost) | Strong on mixed tabular, fast, feature importance | Can miss subtle interactions; needs tuning |
| **TabFM** | Zero-shot ICL, strong priors from pretraining, mixed types natively | Memory ∝ context rows; different error modes |

When two strong models disagree, a **meta-learner** on their probabilities often beats both.

## What we implement

```text
Train base GBM on train features
TabFM.fit(train) as context

Predict P_gbm, P_tabfm on validation
Meta = LogisticRegression([P_gbm, P_tabfm, P_gbm * P_tabfm])  # fit on val

On test:
  hybrid_score = Meta.predict_proba([P_gbm_test, P_tabfm_test, product])
```

### Why validation stacking (not full OOF for TabFM)?

True out-of-fold TabFM requires refitting context many times (expensive on GPU).  
v3 uses **validation stacking**: meta is fit only on the validation fold, then frozen for test.  

This is a standard practical compromise. For research-grade purity, use OOF GBM (`oof_predict_proba`) + multi-fold TabFM when budget allows (`src/churn_revenue/hybrid.py` supports OOF for sklearn models).

## Why this improves results

1. **Error diversity** — TabFM and trees mis-rank different customers.  
2. **Calibration blend** — Meta can down-weight an overconfident model.  
3. **Interaction term** \(p_g \cdot p_t\) — captures “both agree high risk.”  

Typical pattern on Telco-like data: hybrid matches or exceeds the better parent on PR-AUC / F1@tuned threshold.

## How to read notebook outputs

- Coefficients of the meta logistic regression (sign/magnitude of `p_gbm` vs `p_tabfm`)  
- Test table: GBM alone | TabFM alone | Hybrid  
- Policy metrics using hybrid scores for top-K  

## Code map

- `src/churn_revenue/hybrid.py`  
- `notebooks/v3/01_iranian_v3_awesome.py`, `02_telco_v3_awesome.py`, `03_retail_v3_awesome.py`  
