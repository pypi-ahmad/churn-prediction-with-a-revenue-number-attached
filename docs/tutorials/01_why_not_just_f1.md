# Tutorial: Why F1 is not enough (value thresholds & top-K)

## The business question

Retention teams do not ask: “What is your F1?”  
They ask: “Given a **budget**, whom do we call so we save the most revenue?”

F1 balances precision and recall at **one** threshold. It treats every customer equally. In reality:

- A high-value customer at 40% churn risk may be worth more than a low-value customer at 80% risk.  
- Call centers can only contact **top K%** of the base.  
- Contacts cost money; false alarms burn budget.

## What we do in v3

### 1. Expected value of a contact

\[
EV_i = \hat{p}_i \cdot V_i \cdot P(\text{save}\mid\text{contact}) - c
\]

- \(\hat{p}_i\): model churn probability  
- \(V_i\): value anchor (Customer Value / MonthlyCharges / Monetary)  
- \(P(\text{save})\): assumed save rate after contact (placeholder until you have uplift data)  
- \(c\): cost per contact  

**Contact if \(EV_i > 0\)** (or rank by \(EV_i\)).

### 2. Top-K budget policy

If you can only contact 10% of customers:

1. Rank by \(\hat{p}_i\) or by \(EV_i\).  
2. Take the top 10%.  
3. On **validation**, sweep K ∈ {5%, 10%, …} and pick K maximizing **net expected value**.  
4. Freeze K; evaluate once on **test**.

### 3. Why this improves “awesome” without always raising F1

| Outcome | Meaning |
|---------|---------|
| F1 flat, net EV up | Better campaign economics |
| Recall@K up | You cover more true churners inside budget |
| Precision@K up | Fewer wasted calls in the budgeted set |

## How to read the notebook outputs

Look for tables named like:

- `top_k_sweep_val` — validation budget sweep  
- `policy_test` — test metrics for best K and for EV>0  

Compare to the 0.5-threshold F1 from older notebooks: that is a **different objective**.

## Limitations (honest)

- Constant \(P(\text{save})\) is **not** an uplift model. Real save rates vary by segment.  
- Value columns are not always pure future CLV.  
- Still no causal estimate of campaign ROI without randomized pilots.

## Code map

- `src/churn_revenue/value_policy.py`  
- Used in all `notebooks/v3/*_awesome.py` notebooks  
