# COMPAS Algorithmic Fairness Audit

**Live demo:** (https://compas-fairness-audit-yzrxpmq4srgv7hntxsny7k.streamlit.app/)

## What this project does

This project audits the COMPAS recidivism algorithm — the AI tool used in 500+ US courtrooms to predict whether a defendant will reoffend. It measures bias, explains predictions, and tests a fairer alternative.

## Key findings

- African-American defendants are wrongly flagged as high-risk at **3x the rate** of white defendants with identical records
- This gap survives even when race is removed from the model
- Fairness-constrained retraining reduces the disparity by ~40% at a cost of ~1.8% accuracy
- That tradeoff is a policy decision, not a technical one

## What's inside

| File | Description |
|------|-------------|
| `app.py` | Streamlit web app — live fairness audit dashboard |
| `algorithmic_auditor.py` | Reusable audit class for any sklearn classifier |
| `recidivism_fairness_analysis.ipynb` | Full analysis notebook |

## How to use the AlgorithmicAuditor on your own model

```python
from algorithmic_auditor import AlgorithmicAuditor

auditor = AlgorithmicAuditor(
    model             = your_model,
    X_test            = X_test,
    y_test            = y_test,
    sensitive_feature = race_column,
    model_name        = "Your Model Name",
)

auditor.audit()
auditor.print_report()
auditor.generate_pdf("audit_report.pdf")
```

## Tech stack

Python · scikit-learn · SHAP · fairlearn · Streamlit · pandas · matplotlib

## References

- ProPublica COMPAS Analysis (2016)
- Chouldechova, A. (2017) — Fair prediction with disparate impact
- Lundberg & Lee (2017) — A unified approach to interpreting model predictions
- Hardt et al. (2016) — Equality of opportunity in supervised learning
