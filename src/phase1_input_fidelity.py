"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — Securing Input Fidelity                                          ║
║  DecodeLabs · Data Science Project 1 · Industrial Training Kit             ║
║  Author  : [Your Name]                                                      ║
║  Batch   : 2026                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Implements the Missing Data Decision Matrix from the IPO Blueprint:
  ┌─ < 5%  missing  → Row Deletion       (preserve data, prevent synthetic bias)
  ├─ 5–20% missing  → Statistical Imputation (Median for skewed / Group-wise)
  └─ > 20% missing  → KNN Multi-Dimensional Estimation
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# 1. AUDIT — Calculate missingness proportion per feature
# ─────────────────────────────────────────────────────────────────────────────

def audit_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missingness proportion per feature.
    Returns a sorted DataFrame with counts, proportions, and recommended strategy.
    """
    total = len(df)
    miss  = df.isnull().sum()
    prop  = (miss / total * 100).round(2)

    def strategy(p):
        if p < 5:   return "Row Deletion (dropna)"
        if p <= 20: return "Statistical Imputation (Median / Group-Wise)"
        return "KNN Multi-Dimensional Estimation"

    report = pd.DataFrame({
        "missing_count"  : miss,
        "missing_pct"    : prop,
        "strategy"       : prop.apply(strategy),
        "dtype"          : df.dtypes,
    }).query("missing_count > 0").sort_values("missing_pct", ascending=False)

    print("=" * 65)
    print("  MISSINGNESS AUDIT REPORT")
    print("=" * 65)
    print(report.to_string())
    print("=" * 65)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 2. MISSING VALUE HANDLER  (Decision Matrix Logic)
# ─────────────────────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the three-tier Missing Data Decision Matrix:
      Tier 1 (< 5%)   → dropna on those columns
      Tier 2 (5–20%)  → Global Median (skewed numeric) or
                         Sub-Group Conditional (correlated categorical)
      Tier 3 (> 20%)  → KNN Imputation (captures multi-dimensional relationships)
    """
    df = df.copy()
    total = len(df)
    numeric_cols     = df.select_dtypes(include=np.number).columns.tolist()
    missing_pct      = df[numeric_cols].isnull().mean() * 100

    tier1 = missing_pct[missing_pct < 5].index.tolist()
    tier2 = missing_pct[(missing_pct >= 5) & (missing_pct <= 20)].index.tolist()
    tier3 = missing_pct[missing_pct > 20].index.tolist()

    # ── Tier 1: Row Deletion ──────────────────────────────────────────────────
    if tier1:
        before = len(df)
        df.dropna(subset=tier1, inplace=True)
        print(f"\n[Tier 1] Row Deletion → cols: {tier1}")
        print(f"         Rows removed : {before - len(df)} | Remaining: {len(df)}")

    # ── Tier 2: Statistical Imputation ───────────────────────────────────────
    for col in tier2:
        skewness = df[col].skew()
        if abs(skewness) > 1:
            # Skewed → Median (robust against extreme outliers)
            fill_val = df[col].median()
            df[col].fillna(fill_val, inplace=True)
            print(f"\n[Tier 2] Global Median imputation → '{col}' "
                  f"(skew={skewness:.2f}, filled with {fill_val:.2f})")
        else:
            # Correlated/Categorical → Sub-Group Conditional (group by department)
            if 'department' in df.columns:
                df[col] = df.groupby('department')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Fallback for any remaining NaN
                df[col].fillna(df[col].median(), inplace=True)
                print(f"\n[Tier 2] Group-Wise Conditional imputation → '{col}' "
                      f"(grouped by 'department')")
            else:
                fill_val = df[col].median()
                df[col].fillna(fill_val, inplace=True)

    # ── Tier 3: KNN Imputation ───────────────────────────────────────────────
    if tier3:
        print(f"\n[Tier 3] KNN Imputation (k=5) → cols: {tier3}")
        knn_features = [c for c in numeric_cols if c in df.columns]
        imputer      = KNNImputer(n_neighbors=5, weights='distance')
        df[knn_features] = imputer.fit_transform(df[knn_features])
        print(f"         KNN fitted on {len(knn_features)} numeric features.")

    remaining_null = df.isnull().sum().sum()
    print(f"\n✅ Missing value handling complete. Remaining NaN count: {remaining_null}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. OUTLIER DETECTION & NEUTRALIZATION via IQR (Winsorization)
# ─────────────────────────────────────────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame, column: str) -> dict:
    """
    Compute IQR-based outlier boundaries for a numeric column.
    Bounds:  Lower = Q1 - 1.5 × IQR  |  Upper = Q3 + 1.5 × IQR
    """
    Q1  = df[column].quantile(0.25)
    Q3  = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return {
        "Q1"           : Q1,
        "Q3"           : Q3,
        "IQR"          : IQR,
        "lower_bound"  : lower,
        "upper_bound"  : upper,
        "outlier_count": len(outliers),
        "outlier_pct"  : round(len(outliers) / len(df) * 100, 2),
    }


def neutralize_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Winsorization via numpy.clip() — caps values at IQR statistical boundaries.
    Preferred over deletion: preserves row count and sequential integrity.
    """
    df = df.copy()
    print("\n" + "=" * 65)
    print("  OUTLIER DETECTION & NEUTRALIZATION (IQR + Winsorization)")
    print("=" * 65)

    for col in columns:
        stats_dict = detect_outliers_iqr(df, col)
        if stats_dict['outlier_count'] > 0:
            df[col] = np.clip(df[col], stats_dict['lower_bound'], stats_dict['upper_bound'])
            print(f"\n  [{col}]")
            print(f"    IQR Bounds  : [{stats_dict['lower_bound']:.2f}, {stats_dict['upper_bound']:.2f}]")
            print(f"    Outliers    : {stats_dict['outlier_count']} rows ({stats_dict['outlier_pct']}%)")
            print(f"    Action      : Winsorized (clipped to boundary values)")
        else:
            print(f"\n  [{col}] — No outliers detected. ✓")

    print("\n✅ Outlier neutralization complete.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    raw_path = os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'raw', 'employee_data_raw.csv')

    print("\n" + "█" * 65)
    print("  PHASE 1: SECURING INPUT FIDELITY")
    print("  DecodeLabs · Data Science Project 1")
    print("█" * 65)

    df_raw = pd.read_csv(raw_path)
    print(f"\n📥 Raw dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    # Step 1 — Audit
    report = audit_missingness(df_raw)

    # Step 2 — Handle Missing Values
    df_imputed = handle_missing_values(df_raw)

    # Step 3 — Detect & Neutralize Outliers
    numeric_features = ['monthly_salary', 'hours_worked_per_week',
                        'num_projects', 'performance_score',
                        'years_experience', 'training_hours']
    df_clean = neutralize_outliers(df_imputed, numeric_features)

    # Save
    out_path = os.path.join(os.path.dirname(__file__),
                            '..', 'data', 'processed', 'phase1_clean.csv')
    df_clean.to_csv(out_path, index=False)
    print(f"\n💾 Phase 1 output saved → {out_path}")
    print(f"   Final shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} cols\n")
