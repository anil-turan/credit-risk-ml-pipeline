"""
Credit Risk Feature Engineer
==============================
Creates domain-specific features from raw credit application data.
All features are based on real financial risk concepts used in the industry.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CreditRiskFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering for credit risk.

    New features created:
    - DTI_RATIO              : debt-to-income ratio
    - CREDIT_INCOME_RATIO    : how large the loan is vs income
    - CREDIT_GOODS_RATIO     : loan amount vs goods price (overpricing signal)
    - ANNUITY_CREDIT_RATIO   : monthly payment burden vs total loan
    - AGE_YEARS              : applicant age in years
    - EMPLOYMENT_YEARS       : years at current job
    - EMPLOYMENT_AGE_RATIO   : career stability signal
    - INCOME_PER_FAMILY      : income divided by family size
    - AGE_GROUP              : age bin category
    """

    def fit(self, X, y=None):
        # Nothing to learn from data — all features are rule-based
        return self

    def transform(self, X):
        X = X.copy()

        # ── Ratio features ────────────────────────────────────────────────

        # DTI: monthly payment / monthly income
        # Higher = more financial stress → higher default risk
        X['DTI_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)

        # Credit-to-income: how large the loan is relative to income
        X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)

        # Credit-to-goods: loan > goods price means possible overpricing
        X['CREDIT_GOODS_RATIO'] = X['AMT_CREDIT'] / (X['AMT_GOODS_PRICE'] + 1)

        # Annuity-to-credit: high ratio = short repayment period = higher burden
        X['ANNUITY_CREDIT_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_CREDIT'] + 1)

        # ── Time-based features ───────────────────────────────────────────

        # DAYS_BIRTH is negative in the dataset (days before application)
        # Convert to positive years
        X['AGE_YEARS'] = (-X['DAYS_BIRTH']) / 365.25

        # DAYS_EMPLOYED is also negative
        # clip(0, 50) removes the anomaly: value 365243 means "not employed"
        X['EMPLOYMENT_YEARS'] = (-X['DAYS_EMPLOYED']) / 365.25
        X['EMPLOYMENT_YEARS'] = X['EMPLOYMENT_YEARS'].clip(0, 50)

        # Employment-to-age ratio: how much of life spent working
        # Higher = more stable career history
        X['EMPLOYMENT_AGE_RATIO'] = X['EMPLOYMENT_YEARS'] / (X['AGE_YEARS'] + 1)

        # ── Family features ───────────────────────────────────────────────

        # Income per family member: measures real purchasing power
        X['INCOME_PER_FAMILY'] = X['AMT_INCOME_TOTAL'] / (X['CNT_FAM_MEMBERS'] + 1)

        # ── Binning ───────────────────────────────────────────────────────

        # Age group bins — captures non-linear risk patterns across age
        X['AGE_GROUP'] = pd.cut(
            X['AGE_YEARS'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )

        return X