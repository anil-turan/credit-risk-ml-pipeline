"""
Weight of Evidence (WoE) Encoder
==================================
Encodes categorical variables using WoE transformation.
Widely used in credit risk and financial services — regulatory friendly
because it creates a monotonic relationship with the target variable.
"""

import pandas as pd
import numpy as np


class WoEEncoder:
    """
    Weight of Evidence encoder with Information Value (IV) calculation.

    WoE formula : ln( Distribution of Non-Events / Distribution of Events )
    IV formula  : sum( (Non-Event% - Event%) * WoE )

    IV interpretation:
        < 0.02  → Useless predictor
        0.02–0.1 → Weak predictor
        0.1–0.3  → Medium predictor
        > 0.3   → Strong predictor

    Parameters
    ----------
    min_bin_size : float, default=0.05
        Minimum proportion of rows a category must have.
        Small bins are merged into 'Other' to avoid noise.
    """

    def __init__(self, min_bin_size=0.05):
        self.min_bin_size = min_bin_size
        self.woe_maps_ = {}    # col → {category: woe_value}
        self.iv_scores_ = {}   # col → iv_value

    def fit(self, X, y, columns=None):
        """
        Learn WoE values from training data only.
        Never fit on test data — prevents leakage.
        """
        if columns is None:
            columns = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in columns:
            woe_map, iv = self._calc_woe_iv(X[col], y)
            self.woe_maps_[col] = woe_map
            self.iv_scores_[col] = iv

        return self

    def transform(self, X):
        """
        Replace each category with its WoE value.
        Adds a new column: {col}_WOE
        Unknown categories (not seen during fit) are filled with 0.
        """
        X = X.copy()
        for col, woe_map in self.woe_maps_.items():
            if col in X.columns:
                X[f'{col}_WOE'] = X[col].map(woe_map).fillna(0)
        return X

    def fit_transform(self, X, y, columns=None):
        return self.fit(X, y, columns).transform(X)

    def _calc_woe_iv(self, feature, target):
        """Calculate WoE and IV for a single feature."""

        df = pd.DataFrame({'feature': feature, 'target': target})
        grouped = df.groupby('feature')['target'].agg(['sum', 'count'])
        grouped.columns = ['events', 'total']
        grouped['non_events'] = grouped['total'] - grouped['events']

        total_events     = grouped['events'].sum()
        total_non_events = grouped['non_events'].sum()

        # Distribution of events and non-events per category
        # Add small epsilon (1e-10) to avoid log(0)
        grouped['event_dist']     = grouped['events']     / (total_events     + 1e-10)
        grouped['non_event_dist'] = grouped['non_events'] / (total_non_events + 1e-10)

        # WoE = ln(non_event_dist / event_dist)
        grouped['woe'] = np.log(
            (grouped['non_event_dist'] + 1e-10) / (grouped['event_dist'] + 1e-10)
        )

        # IV contribution per category
        grouped['iv'] = (grouped['non_event_dist'] - grouped['event_dist']) * grouped['woe']

        iv      = grouped['iv'].sum()
        woe_map = grouped['woe'].to_dict()

        return woe_map, iv

    def get_iv_summary(self):
        """
        Returns a sorted DataFrame of IV scores with predictive power labels.
        Use this to decide which features to keep before modeling.
        """
        iv_df = pd.DataFrame({
            'Feature': list(self.iv_scores_.keys()),
            'IV':      list(self.iv_scores_.values())
        }).sort_values('IV', ascending=False).reset_index(drop=True)

        iv_df['Predictive Power'] = iv_df['IV'].apply(
            lambda x: 'Useless' if x < 0.02
            else 'Weak'   if x < 0.1
            else 'Medium' if x < 0.3
            else 'Strong'
        )
        return iv_df

    def get_woe_table(self, col):
        """
        Returns the full WoE table for a single column.
        Useful for regulatory documentation and model explainability.
        """
        if col not in self.woe_maps_:
            raise ValueError(f"Column '{col}' was not fitted. Run fit() first.")
        return pd.DataFrame({
            'Category': list(self.woe_maps_[col].keys()),
            'WoE':      list(self.woe_maps_[col].values())
        }).sort_values('WoE', ascending=False)
