# Tutorial: Hybrid TabFM + gradient boosting

## Why combine the models?

| Model family | Strengths | Weaknesses |
|--------------|-----------|------------|
| GBM (XGB/LGBM/CatBoost) | Strong on mixed tabular data, fast, provides feature importance | Can miss subtle interactions and needs tuning |
| TabFM | Zero-shot ICL, pretrained priors, native support for mixed types | Memory grows with context rows; has different error modes |

When the models disagree, a meta-learner can combine their probabilities.

## What we implement

```text
Train base GBM on train features
TabFM.fit(train) as context

Predict P_gbm, P_tabfm on validation
Meta = LogisticRegression([P_gbm, P_tabfm, P_gbm * P_tabfm])  # fit on val

On test:
  hybrid_score = Meta.predict_proba([P_gbm_test, P_tabfm_test, product])
```

### Why use validation stacking instead of full OOF for TabFM?

True out-of-fold TabFM requires refitting context many times (expensive on GPU).  
v3 uses **validation stacking**: meta is fit only on the validation fold, then frozen for test.  

This approach limits repeated GPU fitting. For a stricter evaluation, use OOF GBM (`oof_predict_proba`) with multi-fold TabFM when the compute budget allows. `src/churn_revenue/hybrid.py` supports OOF for sklearn models.

## Why this improves results

1. TabFM and trees mis-rank different customers.
2. The meta-learner can down-weight an overconfident model.
3. The interaction term \(p_g \cdot p_t\) captures cases where both models assign high risk.

On Telco-like data, the hybrid can match or exceed the stronger parent on PR-AUC or F1 at the tuned threshold.

## How to read notebook outputs

- Coefficients of the meta logistic regression (sign/magnitude of `p_gbm` vs `p_tabfm`)  
- Test table: GBM alone | TabFM alone | Hybrid  
- Policy metrics using hybrid scores for top-K  

## Code map

- `src/churn_revenue/hybrid.py`  
- `notebooks/v3/01_iranian_v3_awesome.py`, `02_telco_v3_awesome.py`, `03_retail_v3_awesome.py`  
