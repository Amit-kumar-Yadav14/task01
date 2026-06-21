"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TEST SUITE — Pipeline Unit & Integration Tests                             ║
║  DecodeLabs · Data Science Project 1                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝
Run: pytest tests/test_pipeline.py -v
"""

import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from phase1_input_fidelity      import audit_missingness, handle_missing_values, detect_outliers_iqr, neutralize_outliers
from phase2_computation_engine  import engineer_features, encode_categoricals, eradicate_collinearity


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal realistic dataset for testing."""
    np.random.seed(99)
    n = 200
    df = pd.DataFrame({
        'age'                   : np.random.randint(22, 60, n).astype(float),
        'years_experience'      : np.random.randint(0, 30, n).astype(float),
        'monthly_salary'        : np.random.normal(70000, 15000, n),
        'performance_score'     : np.random.uniform(1, 10, n),
        'hours_worked_per_week' : np.random.normal(45, 8, n),
        'num_projects'          : np.random.randint(1, 15, n).astype(float),
        'training_hours'        : np.random.randint(0, 100, n).astype(float),
        'satisfaction_score'    : np.random.uniform(1, 5, n),
        'department'            : np.random.choice(['Eng', 'HR', 'Sales'], n),
        'education_level'       : np.random.choice(['Bachelor', 'Master'], n),
        'city'                  : np.random.choice(['Mumbai', 'Delhi'], n),
        'left_company'          : np.random.choice([0, 1], n),
    })
    # Inject controlled missing values
    df.loc[np.random.choice(n, 5,  replace=False), 'training_hours']     = np.nan  # <5%
    df.loc[np.random.choice(n, 20, replace=False), 'monthly_salary']     = np.nan  # ~10%
    df.loc[np.random.choice(n, 50, replace=False), 'performance_score']  = np.nan  # ~25%
    # Inject outliers
    df.loc[0, 'monthly_salary']        = 999999
    df.loc[1, 'hours_worked_per_week'] = 150
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase1MissingValues:
    def test_audit_returns_only_missing_columns(self, sample_df):
        report = audit_missingness(sample_df)
        assert report['missing_count'].min() > 0, "Audit must only list columns with NaN"

    def test_missingness_proportions_are_correct(self, sample_df):
        report = audit_missingness(sample_df)
        for col in report.index:
            expected_pct = sample_df[col].isnull().mean() * 100
            assert abs(report.loc[col, 'missing_pct'] - round(expected_pct, 2)) < 0.01

    def test_handle_missing_no_nan_remains(self, sample_df):
        df_clean = handle_missing_values(sample_df)
        assert df_clean.select_dtypes(include=np.number).isnull().sum().sum() == 0, \
            "All numeric NaN must be resolved after imputation"

    def test_handle_missing_preserves_row_shape(self, sample_df):
        """Rows might be dropped for Tier 1 (<5%) but shouldn't be excessive."""
        df_clean = handle_missing_values(sample_df)
        retention = len(df_clean) / len(sample_df)
        assert retention > 0.90, f"Row retention too low: {retention:.2%}"

    def test_handle_missing_preserves_columns(self, sample_df):
        df_clean = handle_missing_values(sample_df)
        assert df_clean.shape[1] == sample_df.shape[1], \
            "Column count must not change after imputation"


class TestPhase1Outliers:
    def test_iqr_bounds_computed_correctly(self, sample_df):
        df_clean = handle_missing_values(sample_df)
        result = detect_outliers_iqr(df_clean, 'monthly_salary')
        Q1  = df_clean['monthly_salary'].quantile(0.25)
        Q3  = df_clean['monthly_salary'].quantile(0.75)
        IQR = Q3 - Q1
        assert abs(result['lower_bound'] - (Q1 - 1.5 * IQR)) < 1e-6
        assert abs(result['upper_bound'] - (Q3 + 1.5 * IQR)) < 1e-6

    def test_winsorization_eliminates_outliers(self, sample_df):
        df_clean = handle_missing_values(sample_df)
        cols     = ['monthly_salary', 'hours_worked_per_week', 'num_projects']
        df_wins  = neutralize_outliers(df_clean, cols)

        for col in cols:
            info  = detect_outliers_iqr(df_clean, col)
            assert df_wins[col].max() <= info['upper_bound'] + 1e-6, \
                f"{col}: max value exceeds upper IQR bound after Winsorization"
            assert df_wins[col].min() >= info['lower_bound'] - 1e-6, \
                f"{col}: min value below lower IQR bound after Winsorization"

    def test_winsorization_preserves_row_count(self, sample_df):
        df_clean = handle_missing_values(sample_df)
        df_wins  = neutralize_outliers(df_clean, ['monthly_salary'])
        assert len(df_wins) == len(df_clean), \
            "Winsorization must NOT drop rows"


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase2FeatureEngineering:
    def setup_method(self):
        pass

    def _get_clean(self, sample_df):
        df = handle_missing_values(sample_df)
        df = neutralize_outliers(df, ['monthly_salary', 'hours_worked_per_week', 'num_projects'])
        return df

    def test_all_7_features_created(self, sample_df):
        df    = self._get_clean(sample_df)
        df_fe = engineer_features(df)
        expected = ['salary_per_year_exp', 'productivity_index', 'experience_age_ratio',
                    'training_investment_score', 'satisfaction_performance_composite',
                    'seniority_band', 'overwork_flag']
        for feat in expected:
            assert feat in df_fe.columns, f"Missing engineered feature: {feat}"

    def test_overwork_flag_is_binary(self, sample_df):
        df    = self._get_clean(sample_df)
        df_fe = engineer_features(df)
        assert set(df_fe['overwork_flag'].unique()).issubset({0, 1}), \
            "overwork_flag must only contain 0 or 1"

    def test_seniority_band_values(self, sample_df):
        df    = self._get_clean(sample_df)
        df_fe = engineer_features(df)
        assert set(df_fe['seniority_band'].unique()).issubset({0, 1, 2, 3}), \
            "seniority_band must be in {0,1,2,3}"

    def test_no_nan_in_engineered_features(self, sample_df):
        df    = self._get_clean(sample_df)
        df_fe = engineer_features(df)
        eng_cols = ['salary_per_year_exp', 'productivity_index', 'experience_age_ratio',
                    'training_investment_score', 'satisfaction_performance_composite']
        for col in eng_cols:
            assert df_fe[col].isnull().sum() == 0, f"NaN found in {col} after engineering"

    def test_productivity_index_non_negative(self, sample_df):
        df    = self._get_clean(sample_df)
        df_fe = engineer_features(df)
        assert (df_fe['productivity_index'] >= 0).all(), \
            "productivity_index must be non-negative"


class TestPhase2Encoding:
    def _get_engineered(self, sample_df):
        df = handle_missing_values(sample_df)
        df = neutralize_outliers(df, ['monthly_salary'])
        df = engineer_features(df)
        return df

    def test_no_object_columns_after_ohe(self, sample_df):
        df = self._get_engineered(sample_df)
        df_enc = encode_categoricals(df)
        assert df_enc.select_dtypes(include='object').empty, \
            "All categorical columns must be OHE-encoded"

    def test_ohe_increases_column_count(self, sample_df):
        df = self._get_engineered(sample_df)
        df_enc = encode_categoricals(df)
        assert df_enc.shape[1] > df.shape[1], \
            "OHE must add columns"

    def test_ohe_values_are_binary(self, sample_df):
        df    = self._get_engineered(sample_df)
        df_enc = encode_categoricals(df)
        ohe_cols = [c for c in df_enc.columns if c not in df.columns and c != 'left_company']
        for col in ohe_cols[:5]:  # sample check
            assert set(df_enc[col].unique()).issubset({0, 1}), \
                f"OHE column {col} contains non-binary values"


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineIntegration:
    def test_full_pipeline_runs_without_error(self, sample_df):
        """End-to-end smoke test: raw data → final feature store."""
        df = handle_missing_values(sample_df)
        df = neutralize_outliers(df, ['monthly_salary', 'hours_worked_per_week'])
        df = engineer_features(df)
        df = encode_categoricals(df)
        df = eradicate_collinearity(df, target_col='left_company')
        assert df.shape[0] > 0, "Pipeline must produce non-empty output"
        assert df.isnull().sum().sum() == 0, "Final output must have zero NaN"
        assert 'left_company' in df.columns, "Target column must be preserved"
