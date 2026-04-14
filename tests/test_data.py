"""
Preprocessor Unit Tests
========================
Each test checks one specific behaviour.
Test name format: test_WHAT_EXPECTED
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.credit_risk.data.preprocessor import CreditRiskPreprocessor


# ── Fixtures ──────────────────────────────────────────────────────────────
# Fixtures are reusable test data shared across multiple tests

@pytest.fixture
def sample_df():
    """Basic test data — mixed types, some missing values."""
    return pd.DataFrame({
        'income':    [50000, 80000, np.nan, 120000, 95000],
        'age':       [25, 35, 42, np.nan, 55],
        'loan_amt':  [10000, 25000, 15000, 40000, 20000],
        'gender':    ['M', 'F', np.nan, 'F', 'M'],
        'education': ['Higher', 'Secondary', 'Higher', np.nan, 'Secondary'],
    })

@pytest.fixture
def high_missing_df():
    """Test data with a column that has too many missing values."""
    return pd.DataFrame({
        'good_col': [1, 2, 3, 4, 5],
        'bad_col':  [np.nan, np.nan, np.nan, np.nan, 5],  # 80% missing
        'cat_col':  ['a', 'b', 'a', 'b', 'a'],
    })

@pytest.fixture
def outlier_df():
    """Test data with an extreme outlier value."""
    return pd.DataFrame({
        'salary': [3000, 3200, 3100, 3050, 3150, 3200, 999999],  # 999999 is the outlier
        'age':    [25, 30, 28, 35, 29, 31, 27],
    })


# ── Test 1: High Missing Columns ──────────────────────────────────────────

def test_preprocessor_removes_high_missing(high_missing_df):
    """Columns above the missing threshold must be dropped."""
    prep = CreditRiskPreprocessor(missing_threshold=0.5)
    prep.fit(high_missing_df)
    result = prep.transform(high_missing_df)

    assert 'bad_col' not in result.columns, "80% missing column was not dropped!"
    assert 'good_col' in result.columns,    "Good column was dropped by mistake!"


def test_preprocessor_keeps_columns_below_threshold(high_missing_df):
    """Columns below the threshold must be kept."""
    prep = CreditRiskPreprocessor(missing_threshold=0.9)  # very high threshold
    prep.fit(high_missing_df)
    result = prep.transform(high_missing_df)

    assert 'bad_col' in result.columns  # 80% missing is still below 90% threshold


# ── Test 2: Imputation ─────────────────────────────────────────────────────

def test_preprocessor_imputes_all_nulls(sample_df):
    """No NaN values should remain after fit_transform."""
    prep = CreditRiskPreprocessor()
    result = prep.fit_transform(sample_df)

    assert result.isnull().sum().sum() == 0, "NaN values still exist after imputation!"


def test_numeric_imputation_uses_median(sample_df):
    """Numeric columns must be filled with the median value."""
    prep = CreditRiskPreprocessor()
    prep.fit(sample_df)

    expected_median = sample_df['income'].median()
    result = prep.transform(sample_df)

    # Row 2 had NaN in income — should now be filled with median
    assert result['income'].iloc[2] == pytest.approx(expected_median, rel=1e-3)


def test_categorical_imputation_uses_mode(sample_df):
    """Categorical columns must be filled with the most frequent value."""
    prep = CreditRiskPreprocessor()
    prep.fit(sample_df)
    result = prep.transform(sample_df)

    # Row 2 had NaN in gender — should now be 'M' or 'F' (whichever appears more)
    assert result['gender'].iloc[2] in ['M', 'F']
    assert pd.notna(result['gender'].iloc[2])


# ── Test 3: Outlier Handling ───────────────────────────────────────────────

def test_outlier_winsorization_clips_extremes(outlier_df):
    """Extreme values must be clipped to the 99th percentile."""
    prep = CreditRiskPreprocessor(outlier_method='iqr')
    prep.fit(outlier_df)
    result = prep.transform(outlier_df)

    assert result['salary'].max() < 999999, "Outlier was not clipped!"
    assert result['salary'].max() <= outlier_df['salary'].quantile(0.99)


def test_outlier_bounds_learned_from_train_only():
    """Outlier boundaries must be learned from train data, not test data."""
    train = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    test  = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})

    prep = CreditRiskPreprocessor(outlier_method='iqr')
    prep.fit(train)           # fit on train only
    result = prep.transform(test)

    # 1000 in test must be clipped to train's 99th percentile
    assert result['x'].max() <= train['x'].quantile(0.99) * 1.01


# ── Test 4: MNAR Flag ─────────────────────────────────────────────────────

def test_mnar_flag_created():
    """An is_missing flag column must be created for MNAR columns."""
    df = pd.DataFrame({
        'own_car_age': [5.0, np.nan, 3.0, np.nan, 7.0],
        'income':      [50000, 80000, 60000, 70000, 90000],
    })
    prep = CreditRiskPreprocessor(mnar_cols=['own_car_age'])
    result = prep.fit_transform(df)

    assert 'own_car_age_is_missing' in result.columns
    assert result['own_car_age_is_missing'].iloc[1] == 1  # was NaN → flag = 1
    assert result['own_car_age_is_missing'].iloc[0] == 0  # was not NaN → flag = 0


# ── Test 5: sklearn Compatibility ─────────────────────────────────────────

def test_fit_transform_consistent(sample_df):
    """fit(X).transform(X) and fit_transform(X) must give the same result."""
    prep1 = CreditRiskPreprocessor()
    result1 = prep1.fit(sample_df).transform(sample_df)

    prep2 = CreditRiskPreprocessor()
    result2 = prep2.fit_transform(sample_df)

    pd.testing.assert_frame_equal(result1, result2)


def test_transform_does_not_modify_original(sample_df):
    """transform() must not change the original DataFrame."""
    original_shape = sample_df.shape
    original_nulls = sample_df.isnull().sum().sum()

    prep = CreditRiskPreprocessor()
    prep.fit(sample_df)
    _ = prep.transform(sample_df)

    assert sample_df.shape == original_shape
    assert sample_df.isnull().sum().sum() == original_nulls


def test_get_params_returns_init_params():
    """sklearn get_params() must return the correct init parameters."""
    prep = CreditRiskPreprocessor(missing_threshold=0.3, outlier_method='iqr')
    params = prep.get_params()

    assert params['missing_threshold'] == 0.3
    assert params['outlier_method'] == 'iqr'