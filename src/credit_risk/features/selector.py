"""
Credit Risk Feature Selector
==============================
Removes uninformative and redundant features before modeling.
Three-step approach: variance threshold → correlation filter → mutual information ranking.
All learned from training data only — never fit on test set.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif


class CreditRiskFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selection pipeline for credit risk modeling.

    Steps (in order):
    1. Variance threshold  — remove near-constant columns
    2. Correlation filter  — remove one of each highly correlated pair
    3. Mutual information  — rank remaining features by predictive power

    Parameters
    ----------
    variance_threshold : float, default=0.01
        Columns with variance below this are removed.
        Near-constant columns add noise without information.
    corr_threshold : float, default=0.90
        If two columns have |r| > this, the one with lower MI is dropped.
        Prevents multicollinearity — critical for Logistic Regression.
    mi_top_k : int or None, default=None
        Keep only the top k features by mutual information score.
        None = keep all features that pass variance and correlation filters.
    """

    def __init__(self, variance_threshold=0.01, corr_threshold=0.90, mi_top_k=None):
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.mi_top_k = mi_top_k

        # Learned during fit()
        self.low_var_cols_ = None
        self.high_corr_cols_ = None
        self.mi_scores_ = None
        self.selected_cols_ = None
        self.drop_cols_ = None

    def fit(self, X, y):
        """
        Learn which columns to keep from training data.
        Requires y because mutual information needs the target.
        """
        X_num = X.select_dtypes(include=[np.number])

        # ── Step 1: Variance threshold ────────────────────────────────────
        # Near-constant columns (e.g. FLAG_MOBIL = 1 for 99.9% of rows)
        # carry almost no information for the model.
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X_num)
        self.low_var_cols_ = X_num.columns[~vt.get_support()].tolist()

        X_filtered = X_num.drop(columns=self.low_var_cols_)

        # ── Step 2: Correlation filter ────────────────────────────────────
        # AMT_CREDIT and AMT_GOODS_PRICE are highly correlated (~0.98).
        # Keeping both confuses Logistic Regression coefficient interpretation.
        # We keep the one with higher mutual information with the target.
        corr_matrix = X_filtered.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Calculate MI scores for all filtered columns
        mi = mutual_info_classif(
            X_filtered.fillna(0), y, random_state=42
        )
        mi_series = pd.Series(mi, index=X_filtered.columns)

        # For each highly correlated pair, drop the one with lower MI
        self.high_corr_cols_ = []
        for col in upper.columns:
            correlated_with = upper.index[upper[col] > self.corr_threshold].tolist()
            for corr_col in correlated_with:
                # Drop the one with lower MI score
                if mi_series.get(col, 0) < mi_series.get(corr_col, 0):
                    if col not in self.high_corr_cols_:
                        self.high_corr_cols_.append(col)
                else:
                    if corr_col not in self.high_corr_cols_:
                        self.high_corr_cols_.append(corr_col)

        X_filtered2 = X_filtered.drop(columns=self.high_corr_cols_, errors='ignore')

        # ── Step 3: Mutual information ranking ───────────────────────────
        # Mutual information measures non-linear relationships with the target.
        # More robust than Pearson correlation for tree-based features.
        mi2 = mutual_info_classif(
            X_filtered2.fillna(0), y, random_state=42
        )
        self.mi_scores_ = pd.Series(mi2, index=X_filtered2.columns).sort_values(ascending=False)

        # Select top k if specified
        if self.mi_top_k is not None:
            self.selected_cols_ = self.mi_scores_.head(self.mi_top_k).index.tolist()
        else:
            self.selected_cols_ = self.mi_scores_.index.tolist()

        # Also keep categorical columns (WoE encoded columns end with _WOE)
        cat_cols = [c for c in X.columns if c.endswith('_WOE') or c.endswith('_is_missing')]
        self.selected_cols_ = list(set(self.selected_cols_ + cat_cols))

        # Full drop list for reporting
        all_num_cols = X_num.columns.tolist()
        self.drop_cols_ = [c for c in all_num_cols if c not in self.selected_cols_]

        return self

    def transform(self, X):
        """Keep only the selected columns."""
        cols_present = [c for c in self.selected_cols_ if c in X.columns]
        return X[cols_present]

    def get_selection_report(self):
        """
        Returns a summary DataFrame showing what was removed and why.
        Useful for documentation and regulatory reporting.
        """
        report = []

        for col in self.low_var_cols_:
            report.append({'Column': col, 'Reason': 'Low variance', 'MI Score': None})

        for col in self.high_corr_cols_:
            report.append({'Column': col, 'Reason': 'High correlation (redundant)', 'MI Score': None})

        return pd.DataFrame(report)

    def get_mi_ranking(self, top_n=30):
        """Returns top N features ranked by mutual information score."""
        if self.mi_scores_ is None:
            raise ValueError("Call fit() first.")
        return self.mi_scores_.head(top_n).reset_index().rename(
            columns={'index': 'Feature', 0: 'MI Score'}
        )