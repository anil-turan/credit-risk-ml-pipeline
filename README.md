# Credit Risk ML Pipeline

End-to-end machine learning pipeline for credit default prediction, built on the Home Credit Default Risk dataset.

**Dataset:** Home Credit Default Risk (Kaggle) — 307,511 loan applications, 120+ features across 8 relational tables  
**Best model:** LightGBM + Optuna · ROC-AUC **0.754** · PR-AUC **0.243** · KS **37.9**  
**Business evaluation:** Profit lift **£11.1M** vs default threshold · 19.2× risk separation across deciles  
**Stack:** Python 3.11 · scikit-learn · XGBoost · LightGBM · Optuna · SHAP · FastAPI

---

## Project Structure

```
credit-risk-ml-pipeline/
├── src/credit_risk/
│   ├── features/
│   │   ├── engineer.py       # domain feature engineering (DTI, ratios, age bins)
│   │   ├── selector.py       # variance + correlation + mutual information selection
│   │   └── woe_encoder.py    # Weight of Evidence encoding for categoricals
│   ├── models/
│   │   ├── trainer.py        # multi-model training + StratifiedKFold CV
│   │   └── evaluator.py      # ROC-AUC, PR-AUC, KS statistic
│   ├── explainability/       # SHAP helpers
│   └── serving/
│       └── app.py            # FastAPI prediction endpoint with risk grading
├── notebooks/
│   ├── 01_eda.ipynb                  # target distribution, missing data, feature distributions
│   ├── 02_feature_engineering.ipynb  # domain features, WoE encoding, IV scores
│   ├── 03_modeling.ipynb             # LR baseline → XGBoost → LightGBM → Optuna tuning
│   ├── 04_explainability.ipynb       # SHAP global + local + dependence plots
│   └── 05_business_evaluation.ipynb  # KS statistic, decile analysis, profit curve
├── configs/
│   └── model_config.yaml
├── data/
│   ├── raw/                  # original CSVs (not committed)
│   └── processed/            # engineered features (not committed)
├── outputs/
│   ├── figures/              # all evaluation and SHAP plots
│   ├── reports/
│   │   └── model_results.csv
│   └── best_model_bundle.pkl # saved model + preprocessor
└── tests/
```

---

## Dataset

[Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) — a Kaggle competition dataset from a consumer finance company serving unbanked populations.

| Table | Rows | Description |
|-------|------|-------------|
| `application_train.csv` | 307,511 | Main table — loan applications with `TARGET` label |
| `bureau.csv` | 1.7M | Previous credits from Credit Bureau |
| `bureau_balance.csv` | 27.3M | Monthly balances of Bureau credits |
| `previous_application.csv` | 1.67M | Previous loan applications at Home Credit |
| `POS_CASH_balance.csv` | 10M | Monthly POS and cash loan snapshots |
| `installments_payments.csv` | 13.6M | Repayment history |
| `credit_card_balance.csv` | 3.8M | Monthly credit card snapshots |

**Class imbalance:** ~8% default rate — all models account for this via `class_weight='balanced'` or `scale_pos_weight`.

---

## Model Results

| Model | ROC-AUC | PR-AUC | CV-AUC Mean | CV-AUC Std |
|-------|---------|--------|-------------|------------|
| Logistic Regression (baseline) | 0.7458 | 0.2278 | 0.7438 | 0.0023 |
| XGBoost | 0.7558 | 0.2444 | 0.7467 | 0.0014 |
| LightGBM (default) | 0.7535 | 0.2419 | 0.7467 | 0.0018 |
| **LightGBM + Optuna** | **0.7540** | **0.2426** | **0.7494** | **0.0017** |

> ROC-AUC is the primary metric in credit risk (alongside KS statistic). PR-AUC is low by design — an 8% default rate makes it harder to achieve high precision at all recall levels.

---

## Domain Features Engineered

All features are derived from financial risk concepts — no statistics learned from data, making them leak-free by design.

| Feature | Formula | Risk Signal |
|---------|---------|-------------|
| `DTI_RATIO` | annuity / income | higher → more financial stress |
| `CREDIT_INCOME_RATIO` | loan amount / income | higher → overextended borrower |
| `CREDIT_GOODS_RATIO` | loan amount / goods price | > 1 → possible overpricing |
| `ANNUITY_CREDIT_RATIO` | annuity / credit | higher → shorter repayment, higher burden |
| `AGE_YEARS` | −DAYS_BIRTH / 365.25 | younger applicants → higher default risk |
| `EMPLOYMENT_YEARS` | −DAYS_EMPLOYED / 365.25 | clipped at 50 to remove anomalous values |
| `EMPLOYMENT_AGE_RATIO` | employment years / age | career stability signal |
| `INCOME_PER_FAMILY` | income / family size | real purchasing power per person |
| `AGE_GROUP` | bins: 18–25, 26–35, 36–45, 46–55, 56–65, 65+ | captures non-linear age risk |

Implemented as `CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin)` in `src/credit_risk/features/engineer.py`.

---

## Feature Selection — Information Value (IV)

Features are ranked by IV score before model training. IV measures how well a feature separates defaulters from non-defaulters.

| IV Range | Predictive Power |
|----------|-----------------|
| < 0.02 | Useless |
| 0.02 – 0.1 | Weak |
| 0.1 – 0.3 | Medium |
| > 0.3 | Strong |

**Top features by IV:**

| Feature | IV Score |
|---------|----------|
| `ORGANIZATION_TYPE` | 0.073 |
| `NAME_INCOME_TYPE` | 0.060 |
| `NAME_EDUCATION_TYPE` | 0.051 |
| `OCCUPATION_TYPE` | 0.050 |

---

## Categorical Encoding — Weight of Evidence (WoE)

WoE is the industry-standard encoding for credit scorecards. It transforms categorical variables into log-odds space, making them directly interpretable as risk weights and compatible with logistic regression.

```
WoE_i = ln(Distribution of Events_i / Distribution of Non-Events_i)
```

Implemented in `src/credit_risk/features/woe_encoder.py` as a sklearn-compatible transformer — fit on training data only, applied to test via `Pipeline`.

---

## Business Evaluation (Notebook 05)

Standard ML metrics are not enough for credit risk. Notebook 05 translates model performance into lender-relevant business metrics.

**KS Statistic — 37.9** (Acceptable range: 30–40). The model separates defaulters from non-defaulters with a 37.9pp gap at the optimal cut-off.

**Decile Analysis** — top decile default rate: **27.1%** vs bottom decile: **1.4%** — a **19.2× separation ratio**. Clear monotonic decline confirms the scorecard is well-calibrated.

**Profit Curve** — using lender economics (TN = +£500, FP = −£500, FN = −£2,000), the optimal threshold (0.81) yields **£18.6M expected profit** vs £7.5M at the standard 0.5 cut-off — a **£11.1M lift**.

---

## API

Start the server:
```bash
uvicorn src.credit_risk.serving.app:app --reload
```

Interactive docs at `http://localhost:8000/docs`.

### `GET /health`
```json
{
  "status": "ok",
  "model": "LightGBM + Optuna credit risk pipeline",
  "test_roc_auc": 0.754
}
```

### `POST /predict`

**Request body (example — high-risk applicant):**
```json
{
  "AMT_INCOME_TOTAL": 90000,
  "AMT_CREDIT": 450000,
  "AMT_ANNUITY": 22500,
  "AMT_GOODS_PRICE": 400000,
  "DAYS_BIRTH": -9000,
  "DAYS_EMPLOYED": -180,
  "CNT_FAM_MEMBERS": 1,
  "NAME_EDUCATION_TYPE": "Secondary / secondary special",
  "NAME_INCOME_TYPE": "Working",
  "ORGANIZATION_TYPE": "Business Entity Type 3",
  "EXT_SOURCE_1": 0.25,
  "EXT_SOURCE_2": 0.30,
  "EXT_SOURCE_3": 0.20
}
```

**Response:**
```json
{
  "default_probability": 0.3821,
  "risk_grade": "D",
  "decision": "Review",
  "model_version": "lgb-optuna-auc0.754"
}
```

**Risk grades:**

| Grade | Probability | Decision |
|-------|------------|----------|
| A | < 5% | Approve |
| B | 5–10% | Approve |
| C | 10–20% | Review |
| D | 20–35% | Review |
| E | ≥ 35% | Decline |

---

## Quickstart

**1. Install dependencies**
```bash
pip install -e ".[dev]"
```

**2. Download data from Kaggle**
```bash
kaggle competitions download -c home-credit-default-risk -p data/raw --unzip
```

**3. Run notebooks in order**
```
notebooks/01_eda.ipynb
notebooks/02_feature_engineering.ipynb
notebooks/03_modeling.ipynb
notebooks/04_explainability.ipynb
notebooks/05_business_evaluation.ipynb
```

**4. Run tests**
```bash
pytest tests/ -v
```

**5. Start the API**
```bash
uvicorn src.credit_risk.serving.app:app --reload
```

---

## Key EDA Findings

- `DAYS_EMPLOYED` contains an anomalous value (365,243) for unemployed applicants — clipped in `EMPLOYMENT_YEARS`
- Missing data is concentrated in external source scores (`EXT_SOURCE_1/2/3`) and document flags
- Higher `DTI_RATIO` and `CREDIT_INCOME_RATIO` correlate with higher default rates
- Younger applicants (18–25) show higher default rates than the 36–45 age group
- `ORGANIZATION_TYPE` and `OCCUPATION_TYPE` are the strongest categorical predictors

---

## Technical Notes

- **No data leakage:** `CreditRiskFeatureEngineer` uses only rule-based transformations — no statistics learned from training data
- **WoE encoding:** fit on training data only, applied to test via sklearn `Pipeline`
- **CV strategy:** `StratifiedKFold(n_splits=5)` preserves the 8% default rate across all folds
- **Imbalanced classes:** handled with `class_weight='balanced'` (LR) and `scale_pos_weight` (XGBoost, LightGBM)
- **Optuna:** 50 TPE trials maximising 5-fold CV ROC-AUC, with `MedianPruner` to stop weak trials early
- **Explainability:** SHAP `TreeExplainer` for global feature importance and per-applicant decision explanations
- **Business threshold:** chosen by maximising expected profit, not F1 or accuracy

---

## License

MIT License — Copyright (c) 2026 Anil Turan
