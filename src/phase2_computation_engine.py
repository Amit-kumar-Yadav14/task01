"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 2 — The Vectorized Computation Engine                                ║
║  DecodeLabs · Data Science Project 1 · Industrial Training Kit             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Implements:
  1. Vectorized Feature Engineering  — NO Python for-loops
  2. Categorical Translation         — One-Hot Encoding (OHE) into Coordinate Space
  3. Collinearity Eradication        — Pearson product-moment correlation algorithm
  4. Min-Max / Standard Scaling      — Normalize coordinate magnitudes
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. VECTORIZED FEATURE ENGINEERING  (≥3 new predictive features)
# ─────────────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer new predictive features using pure vectorized NumPy/Pandas operations.
    No Python for-loops — all operations execute via compiled C-level SIMD.

    New Features Created:
      1. salary_per_year_exp      — Earnings efficiency per year of experience
      2. productivity_index       — Projects per hour worked, scaled × 100
      3. experience_age_ratio     — Career start proxy (higher = early starter)
      4. training_investment_score— Weighted training engagement metric
      5. satisfaction_performance_composite — Holistic employee health index
      6. seniority_band           — Ordinal career stage classification (0–3)
      7. overwork_flag            — Binary flag: working > 55hrs/week
    """
    df = df.copy()

    print("=" * 65)
    print("  VECTORIZED FEATURE ENGINEERING")
    print("=" * 65)

    # ── Feature 1: Salary Efficiency per Year of Experience ──────────────────
    # Avoids division-by-zero via np.where (fully vectorized)
    df['salary_per_year_exp'] = np.where(
        df['years_experience'] > 0,
        df['monthly_salary'] / df['years_experience'],
        df['monthly_salary']  # For 0 experience, use raw salary
    )
    print("\n  ✅ [F1] salary_per_year_exp   — Monthly salary ÷ years_experience")

    # ── Feature 2: Productivity Index ────────────────────────────────────────
    # num_projects per hour worked, normalized ×100 for readability
    df['productivity_index'] = np.where(
        df['hours_worked_per_week'] > 0,
        (df['num_projects'] / df['hours_worked_per_week']) * 100,
        0
    )
    print("  ✅ [F2] productivity_index     — (Projects / HoursPerWeek) × 100")

    # ── Feature 3: Experience-to-Age Ratio ───────────────────────────────────
    # Proxy for career-start age; older-with-less-exp → lower ratio (late starters)
    df['experience_age_ratio'] = np.where(
        df['age'] > 0,
        df['years_experience'] / df['age'],
        0
    )
    print("  ✅ [F3] experience_age_ratio   — Years experience ÷ Age")

    # ── Feature 4: Training Investment Score ─────────────────────────────────
    # Normalized training hours × satisfaction × 10 → continuous engagement index
    max_training = df['training_hours'].max()
    df['training_investment_score'] = (
        (df['training_hours'] / max_training) *
        df['satisfaction_score'] * 10
    ).round(4)
    print("  ✅ [F4] training_investment_score — Normalized training × satisfaction")

    # ── Feature 5: Satisfaction-Performance Composite ────────────────────────
    # Geometric mean of scaled satisfaction and performance scores
    sat_scaled  = df['satisfaction_score'] / df['satisfaction_score'].max()
    perf_scaled = df['performance_score']  / df['performance_score'].max()
    df['satisfaction_performance_composite'] = np.sqrt(sat_scaled * perf_scaled).round(4)
    print("  ✅ [F5] satisfaction_performance_composite — Geometric mean of both scores")

    # ── Feature 6: Seniority Band  (Ordinal Binning, fully vectorized) ───────
    df['seniority_band'] = pd.cut(
        df['years_experience'],
        bins=[-1, 2, 7, 15, 100],
        labels=[0, 1, 2, 3]       # Junior / Mid / Senior / Principal
    ).astype(int)
    print("  ✅ [F6] seniority_band         — Ordinal: 0=Junior,1=Mid,2=Senior,3=Principal")

    # ── Feature 7: Overwork Flag  (Vectorized Boolean → Int) ─────────────────
    df['overwork_flag'] = (df['hours_worked_per_week'] > 55).astype(int)
    print("  ✅ [F7] overwork_flag          — Binary: 1 if >55 hrs/week else 0")

    newly_added = ['salary_per_year_exp', 'productivity_index', 'experience_age_ratio',
                   'training_investment_score', 'satisfaction_performance_composite',
                   'seniority_band', 'overwork_flag']
    print(f"\n  📦 {len(newly_added)} new features engineered.\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. CATEGORICAL TRANSLATION → ONE-HOT ENCODING (Orthogonal Coordinate Space)
# ─────────────────────────────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, drop_first: bool = True) -> pd.DataFrame:
    """
    Convert nominal categorical columns into orthogonal coordinate axes via OHE.
    drop_first=True eliminates the Dummy Variable Trap (perfect multicollinearity).

    Why NOT Label Encoding:
      Assigning ascending integers creates false mathematical distances
      (e.g., Engineering=3 implies it is '3×' more than HR=1).
    """
    categorical_cols = df.select_dtypes(include='object').columns.tolist()

    print("=" * 65)
    print("  CATEGORICAL TRANSLATION → ONE-HOT ENCODING")
    print("=" * 65)

    df_encoded = pd.get_dummies(df, columns=categorical_cols,
                                 drop_first=drop_first, dtype=int)

    new_cols = set(df_encoded.columns) - set(df.columns)
    print(f"\n  Input categorical columns : {categorical_cols}")
    print(f"  OHE columns generated     : {len(new_cols)}")
    print(f"  Total features after OHE  : {df_encoded.shape[1]}")
    print(f"  drop_first=True           : Dummy Variable Trap avoided ✓\n")
    return df_encoded


# ─────────────────────────────────────────────────────────────────────────────
# 3. COLLINEARITY ERADICATION ALGORITHM
# ─────────────────────────────────────────────────────────────────────────────

def eradicate_collinearity(df: pd.DataFrame,
                            target_col: str,
                            threshold: float = 0.80) -> pd.DataFrame:
    """
    4-Step Collinearity Eradication Algorithm:
      Step 1: Build Absolute Correlation Matrix
      Step 2: Isolate Upper Triangle (avoid duplicate pairs)
      Step 3: Identify feature pairs with |corr| > threshold (default 0.80)
      Step 4: Target Comparison — drop the feature with LOWER correlation to target
                                   (preserves maximum predictive information)

    When highly correlated features share a column space, X^T X becomes
    singular (non-invertible) — OLS coefficients become unstable.
    """
    print("=" * 65)
    print(f"  COLLINEARITY ERADICATION ALGORITHM (threshold={threshold})")
    print("=" * 65)

    numeric_df = df.select_dtypes(include=np.number)

    # Step 1: Absolute Correlation Matrix
    corr_matrix  = numeric_df.corr().abs()

    # Step 2: Isolate Upper Triangle
    upper_tri    = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Step 3: Find pairs > threshold
    collinear_pairs = [
        (col, row, upper_tri.loc[row, col])
        for col in upper_tri.columns
        for row in upper_tri.index
        if pd.notna(upper_tri.loc[row, col]) and upper_tri.loc[row, col] > threshold
    ]

    if not collinear_pairs:
        print(f"\n  ✅ No collinear feature pairs found above threshold {threshold}.")
        return df

    print(f"\n  Collinear pairs detected (|corr| > {threshold}):")
    cols_to_drop = set()
    for feat_a, feat_b, corr_val in collinear_pairs:
        # Step 4: Target Comparison — drop the weakest link
        corr_a = abs(numeric_df[feat_a].corr(df[target_col]))
        corr_b = abs(numeric_df[feat_b].corr(df[target_col]))
        drop   = feat_a if corr_a < corr_b else feat_b
        keep   = feat_b if drop == feat_a else feat_a
        cols_to_drop.add(drop)
        print(f"\n    Pair : {feat_a} ↔ {feat_b}  (corr={corr_val:.3f})")
        print(f"    Corr({feat_a}, target) = {corr_a:.3f}")
        print(f"    Corr({feat_b}, target) = {corr_b:.3f}")
        print(f"    → KEEP: {keep}  |  DROP: {drop}")

    df_clean = df.drop(columns=list(cols_to_drop))
    print(f"\n  Features dropped : {list(cols_to_drop)}")
    print(f"  Remaining features : {df_clean.shape[1]}")
    return df_clean


# ─────────────────────────────────────────────────────────────────────────────
# 4. FEATURE SCALING
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(df: pd.DataFrame,
                   target_col: str,
                   method: str = 'standard') -> tuple[pd.DataFrame, object]:
    """
    Scale numeric features to normalize coordinate magnitudes.
    Excludes binary/flag columns and the target variable.

    method='standard' → StandardScaler  (zero mean, unit variance)
    method='minmax'   → MinMaxScaler    ([0, 1] bounded range)
    """
    binary_cols = [c for c in df.select_dtypes(include=np.number).columns
                   if df[c].nunique() <= 2]
    exclude     = set([target_col] + binary_cols)
    scale_cols  = [c for c in df.select_dtypes(include=np.number).columns
                   if c not in exclude]

    scaler = StandardScaler() if method == 'standard' else MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[scale_cols] = scaler.fit_transform(df[scale_cols])

    print("=" * 65)
    print(f"  FEATURE SCALING ({method.upper()}SCALER)")
    print("=" * 65)
    print(f"  Columns scaled : {len(scale_cols)}")
    print(f"  Excluded       : binary flags + target variable '{target_col}'")
    print(f"  ✅ Scaling complete.\n")
    return df_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    in_path  = os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'processed', 'phase1_clean.csv')
    out_path = os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'processed', 'phase2_engineered.csv')

    print("\n" + "█" * 65)
    print("  PHASE 2: VECTORIZED COMPUTATION ENGINE")
    print("  DecodeLabs · Data Science Project 1")
    print("█" * 65 + "\n")

    df = pd.read_csv(in_path)
    print(f"📥 Phase 1 output loaded: {df.shape}\n")

    df = engineer_features(df)
    df = encode_categoricals(df, drop_first=True)
    df = eradicate_collinearity(df, target_col='left_company', threshold=0.80)
    df, scaler = scale_features(df, target_col='left_company', method='standard')

    df.to_csv(out_path, index=False)
    print(f"💾 Phase 2 output saved → {out_path}")
    print(f"   Final shape: {df.shape[0]} rows × {df.shape[1]} cols\n")
