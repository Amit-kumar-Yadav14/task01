"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PHASE 3 — Structural Contracts & Scaling                                   ║
║  DecodeLabs · Data Science Project 1 · Industrial Training Kit             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
import json, warnings
warnings.filterwarnings("ignore")


def build_output_schema() -> DataFrameSchema:
    """
    Runtime data contract for the processed feature dataset.
    strict=False → extra columns (OHE, optional engineered features) allowed.
    required=False on engineered columns that may be dropped by collinearity step.
    """
    schema = DataFrameSchema(
        columns={
            "age": Column(float, nullable=False),
            "monthly_salary": Column(
                float, nullable=False,
                checks=[Check(lambda s: s.notna().all(), error="salary has NaN")],
            ),
            "performance_score": Column(
                float, nullable=False,
                checks=[Check(lambda s: s.notna().all(), error="performance_score has NaN")],
            ),
            "satisfaction_score": Column(
                float, nullable=False,
                checks=[Check(lambda s: s.notna().all(), error="satisfaction_score has NaN")],
            ),
            "overwork_flag": Column(
                int, nullable=False,
                checks=[Check(lambda s: s.isin([0, 1]).all(), error="overwork_flag must be binary")],
                description="Binary: 1 if >55 hrs/week"
            ),
            "left_company": Column(
                int, nullable=False,
                checks=[Check(lambda s: s.isin([0, 1]).all(), error="Target must be binary")],
                description="Target variable: employee attrition label"
            ),
            # Engineered features — optional; may be dropped by collinearity eradication
            "salary_per_year_exp": Column(float, nullable=False, required=False),
            "productivity_index": Column(float, nullable=False, required=False),
            "experience_age_ratio": Column(float, nullable=False, required=False),
            "training_investment_score": Column(float, nullable=False, required=False),
            "satisfaction_performance_composite": Column(float, nullable=False, required=False),
            "seniority_band": Column(
                int, nullable=False, required=False,
                checks=[Check(lambda s: s.isin([0,1,2,3]).all(), error="seniority_band ∈ {0,1,2,3}")]
            ),
        },
        strict=False,
        coerce=True,
        name="DecodeLabs_DS_Project1_OutputSchema",
    )
    return schema


def validate_dataset(df: pd.DataFrame) -> tuple:
    schema = build_output_schema()
    print("=" * 65)
    print("  RUNTIME SCHEMA VALIDATION (Pandera — lazy=True)")
    print("=" * 65)
    try:
        validated_df = schema.validate(df, lazy=True)
        print("\n  ✅ ALL schema assertions PASSED.")
        print(f"  Dataset shape  : {validated_df.shape}")
        print("  Contract status: VALID — safe to serve to downstream estimators.\n")
        return validated_df, True
    except pa.errors.SchemaErrors as exc:
        print("\n  ❌ SCHEMA VIOLATIONS DETECTED:")
        print(exc.failure_cases.to_string())
        print("\n  ⚠️  Pipeline continued — review failure_cases log above.\n")
        return df, False


def generate_summary_report(df_raw, df_final, validation_passed):
    report = {
        "pipeline"            : "DecodeLabs DS Project 1 — Advanced EDA & Feature Engineering",
        "batch"               : "2026",
        "raw_shape"           : list(df_raw.shape),
        "final_shape"         : list(df_final.shape),
        "rows_retained_pct"   : round(len(df_final) / len(df_raw) * 100, 2),
        "new_features_added"  : 7,
        "schema_valid"        : validation_passed,
        "null_count_final"    : int(df_final.isnull().sum().sum()),
        "engineered_features" : [
            "salary_per_year_exp", "productivity_index", "experience_age_ratio",
            "training_investment_score", "satisfaction_performance_composite",
            "seniority_band", "overwork_flag",
        ],
        "pipeline_phases" : {
            "Phase 1 — Input Fidelity"     : ["Missing value audit", "3-tier Decision Matrix", "IQR Winsorization"],
            "Phase 2 — Computation Engine" : ["7 vectorized features", "OHE categorical encoding", "Collinearity eradication", "StandardScaler"],
            "Phase 3 — Contracts"          : ["Pandera schema (lazy=True)", "Runtime type+boundary checks", "JSON report export"],
        }
    }
    print("=" * 65)
    print("  PIPELINE SUMMARY REPORT")
    print("=" * 65)
    print(f"\n  Raw shape           : {report['raw_shape']}")
    print(f"  Final shape         : {report['final_shape']}")
    print(f"  Rows retained       : {report['rows_retained_pct']}%")
    print(f"  New features        : {report['new_features_added']}")
    print(f"  Remaining NaN       : {report['null_count_final']}")
    print(f"  Schema contract     : {'✅ PASSED' if validation_passed else '❌ FAILED'}")
    return report


if __name__ == "__main__":
    import os
    raw_path    = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw',       'employee_data_raw.csv')
    in_path     = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'phase2_engineered.csv')
    out_path    = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'final_feature_store.csv')
    report_path = os.path.join(os.path.dirname(__file__), '..', 'outputs',           'pipeline_report.json')

    print("\n" + "█" * 65)
    print("  PHASE 3: STRUCTURAL CONTRACTS & SCALING")
    print("  DecodeLabs · Data Science Project 1")
    print("█" * 65 + "\n")

    df_raw   = pd.read_csv(raw_path)
    df_final = pd.read_csv(in_path)
    print(f"📥 Phase 2 output loaded: {df_final.shape}\n")

    df_validated, is_valid = validate_dataset(df_final)
    report = generate_summary_report(df_raw, df_validated, is_valid)

    df_validated.to_csv(out_path, index=False)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Final feature store → {out_path}")
    print(f"📋 Pipeline report     → {report_path}\n")
