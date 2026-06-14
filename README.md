# Credit Risk ML Pipeline

End-to-end machine learning pipeline for credit default prediction, built on the Home Credit Default Risk dataset.

**Dataset:** Home Credit Default Risk (Kaggle) — 307,511 loan applications, 120+ features across 8 relational tables  
**Baseline:** Logistic Regression · ROC-AUC 0.7458 · CV-AUC 0.7438 ± 0.0023  
**Stack:** Python 3.11 · scikit-learn · XGBoost · LightGBM · SHAP · WoE/IV

---

## Project Structure

```
credit-risk-ml-pipeline/
├── src/credit_risk/
│   ├── features/
│   │   ├── engineer.py       # domain feature engineering (DTI, ratios, age bins)
│   │   ├── selector.py       # IV-based feature selection
│   │   └── woe_encoder.py    # Weight of Evidence encoding for categoricals
│   ├── models/
│   │   ├── trainer.py        # multi-model training + StratifiedKFold CV
│   │   └── evaluator.py      # ROC-AUC, PR-AUC, KS statistic
│   └── explainability/       # SHAP helpers
├── notebooks/
│   ├── 01_eda.ipynb                  # target distribution, missing data, distributions
│   ├── 02_feature_engineering.ipynb  # domain features, WoE encoding, IV scores
│   ├── 03_modeling.ipynb             # LR baseline, XGBoost, LightGBM, CV comparison
│   ├── 04_explainability.ipynb       # SHAP global + local explanations
│   └── 05_business_evaluation.ipynb  # KS statistic, score bands, approval rate analysis
├── configs/
│   └── model_config.yaml
├── data/
│   ├── raw/                  # original CSVs (not committed)
│   └── processed/            # engineered features (not committed)
├── outputs/
│   ├── figures/              # EDA and evaluation plots
│   └── reports/
│       ├── model_results.csv
│       └── iv_summary.csv
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

## Domain Features Engineered

All features are derived from financial risk concepts — no statistics learned from data, making them leak-free.

| Feature | Formula | Risk Signal |
|---------|---------|-------------|
| `DTI_RATIO` | annuity / income | higher → more financial stress |
| `CREDIT_INCOME_RATIO` | loan amount / income | higher → overextended borrower |
| `CREDIT_GOODS_RATIO` | loan amount / goods price | > 1 → possible overpricing |
| `ANNUITY_CREDIT_RATIO` | annuity / credit | higher → shorter repayment, higher burden |
| `AGE_YEARS` | −DAYS_BIRTH / 365.25 | younger applicants → higher default risk |
| `EMPLOYMENT_YEARS` | −DAYS_EMPLOYED / 365.25 | clipped at 50, removes anomalous value |
| `EMPLOYMENT_AGE_RATIO` | employment years / age | career stability signal |
| `INCOME_PER_FAMILY` | income / family size | real purchasing power per person |
| `AGE_GROUP` | bins: 18–25, 26–35, 36–45, 46–55, 56–65, 65+ | captures non-linear age risk |

Implemented as `CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin)` in `src/credit_risk/features/engineer.py`.

---

## Feature Selection — Information Value (IV)

Features are ranked by IV score before model training. IV quantifies how well a feature separates defaulters from non-defaulters.

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

## Model Results

| Model | ROC-AUC | PR-AUC | CV-AUC Mean | CV-AUC Std |
|-------|---------|--------|-------------|------------|
| Logistic Regression (baseline) | 0.7458 | 0.2278 | 0.7438 | 0.0023 |
| XGBoost | — | — | see notebook 03 | — |
| LightGBM | — | — | see notebook 03 | — |

> PR-AUC is low due to the 8% default rate — this is expected and normal in credit risk. ROC-AUC and KS statistic are the primary evaluation metrics in this domain.

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

---

## Key EDA Findings

- `DAYS_EMPLOYED` contains an anomalous value (365,243) for unemployed applicants — clipped in `EMPLOYMENT_YEARS`
- Missing data concentrated in external source scores (`EXT_SOURCE_1/2/3`) and document flags
- Higher `DTI_RATIO` and `CREDIT_INCOME_RATIO` correlate with higher default rates
- Younger applicants (18–25) show higher default rates than 36–45 age group
- `ORGANIZATION_TYPE` and `OCCUPATION_TYPE` are the strongest categorical predictors

---

## Technical Notes

- **No data leakage:** `CreditRiskFeatureEngineer` uses only rule-based transformations — no statistics learned from training data
- **WoE encoding:** fit on training data only, applied to test via sklearn `Pipeline`
- **CV strategy:** `StratifiedKFold(n_splits=5)` preserves the 8% default rate across all folds
- **Imbalanced classes:** handled with `class_weight='balanced'` (Logistic Regression, LightGBM) and `scale_pos_weight` (XGBoost)
- **Explainability:** SHAP `TreeExplainer` for global feature importance and per-applicant decision explanations

---

## License

MIT License — Copyright (c) 2026 Anil Turan
