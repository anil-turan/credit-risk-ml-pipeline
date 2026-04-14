"""
Credit Risk Data Preprocessor
==============================
Modular, reusable, sklearn-compatible preprocessing pipeline.
Inherits from BaseEstimator + TransformerMixin so it fits
directly into a sklearn Pipeline.
"""



import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer, KNNImputer


class CreditRiskPreprocessor(BaseEstimator, TransformerMixin):
    """
    Credit risk data preprocessing pipeline.

    Steps:
    1. Drop columns with too many missing values (missing_threshold)
    2. Add is_missing flag for MNAR columns
    3. Fill missing numeric values with median
    4. Fill missing categorical values with most frequent value
    5. Clip outliers using IQR method (winsorize)

    Parameters
    ----------
    missing_threshold : float, default=0.5
        Columns with more missing values than this ratio are dropped.
    outlier_method : str, default='iqr'
        Method for handling outliers. Only 'iqr' is supported for now.
    mnar_cols : list, default=None
        Columns known to be MNAR. A is_missing flag is added for each.
    """

    def __init__(self, missing_threshold=0.5, outlier_method='iqr', mnar_cols=None):
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.mnar_cols = mnar_cols or []

        # These will be learned during fit()
        # Trailing underscore is the sklearn naming convention
        self.drop_cols_ = None
        self.num_cols_ = None
        self.cat_cols_ = None
        self.num_imputer_ = None
        self.cat_imputer_ = None
        self.lower_ = {}
        self.upper_ = {}

    def fit(self, X, y=None):
        """
        Learn all preprocessing parameters from training data ONLY.
        Never fit on test data — this prevents data leakage.
        """
        # Step 1: Find columns with too many missing values
        missing_pct = X.isnull().sum() / len(X)
        self.drop_cols_ = missing_pct[missing_pct > self.missing_threshold].index.tolist()

        X_clean = X.drop(columns=self.drop_cols_)

        # Step 2: Save column names by type
        self.num_cols_ = X_clean.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X_clean.select_dtypes(include=['str']).columns.tolist()

        # Step 3: Fit imputers on training data only
        if self.num_cols_:
            self.num_imputer_ = SimpleImputer(strategy='median')
            self.num_imputer_.fit(X_clean[self.num_cols_])

        if self.cat_cols_:
            self.cat_imputer_ = SimpleImputer(strategy='most_frequent')
            self.cat_imputer_.fit(X_clean[self.cat_cols_])

        # Step 4: Learn outlier boundaries from training data
        # Q1=0.01, Q3=0.99 — only clip the most extreme values
        if self.outlier_method == 'iqr':
            for col in self.num_cols_:
                self.lower_[col] = X_clean[col].quantile(0.01)
                self.upper_[col] = X_clean[col].quantile(0.99)

        return self  # sklearn convention: fit always returns self

    def transform(self, X):
        """
        Apply the learned parameters to transform the data.
        Used on both train and test sets.
        """
        X = X.copy()  # never modify the original dataframe

        # Step 1: Drop high-missing columns
        X = X.drop(columns=self.drop_cols_, errors='ignore')

        # Step 2: Add is_missing flag for MNAR columns
        # e.g. OWN_CAR_AGE is missing because the customer has no car
        for col in self.mnar_cols:
            if col in X.columns:
                X[f'{col}_is_missing'] = X[col].isnull().astype(int)

        # Step 3: Fill missing values
        # Only impute columns that still exist after dropping
        num_cols_present = [c for c in self.num_cols_ if c in X.columns]
        cat_cols_present = [c for c in self.cat_cols_ if c in X.columns]

        if num_cols_present and self.num_imputer_:
            X[num_cols_present] = self.num_imputer_.transform(X[num_cols_present])

        if cat_cols_present and self.cat_imputer_:
            X[cat_cols_present] = self.cat_imputer_.transform(X[cat_cols_present])

        # Step 4: Clip outliers to learned boundaries (winsorize)
        if self.outlier_method == 'iqr':
            for col in num_cols_present:
                X[col] = X[col].clip(self.lower_[col], self.upper_[col])

        return X

    def get_feature_names_out(self):
        """Return output column names. Required for sklearn Pipeline compatibility."""
        mnar_flags = [f'{c}_is_missing' for c in self.mnar_cols]
        return self.num_cols_ + self.cat_cols_ + mnar_flags