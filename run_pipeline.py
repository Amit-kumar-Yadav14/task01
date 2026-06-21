"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MAIN PIPELINE — Complete IPO Architecture Runner                           ║
║  DecodeLabs · Data Science Project 1 · Industrial Training Kit             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Orchestrates the full Input → Process → Output pipeline in one command:
  python run_pipeline.py

Output artifacts:
  data/raw/employee_data_raw.csv          ← Synthetic dataset
  data/processed/phase1_clean.csv         ← After imputation + outlier removal
  data/processed/phase2_engineered.csv    ← After feature engineering + OHE
  data/processed/final_feature_store.csv  ← Validated production-ready store
  outputs/pipeline_report.json            ← Structured run report
"""

import os
import sys
import time

# ── Add src/ to path ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd

from phase1_input_fidelity   import audit_missingness, handle_missing_values, neutralize_outliers
from phase2_computation_engine import engineer_features, encode_categoricals, eradicate_collinearity, scale_features
from phase3_contracts_scaling  import validate_dataset, generate_summary_report


BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║    ██████╗ ███████╗ ██████╗ ██████╗ ██████╗ ███████╗               ║
║    ██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██╔════╝               ║
║    ██║  ██║█████╗  ██║     ██║   ██║██║  ██║█████╗                 ║
║    ██║  ██║██╔══╝  ██║     ██║   ██║██║  ██║██╔══╝                 ║
║    ██████╔╝███████╗╚██████╗╚██████╔╝██████╔╝███████╗               ║
║    ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝               ║
║                                                                      ║
║    LABS  ·  Data Science Project 1  ·  Batch 2026                  ║
║    Advanced EDA & Feature Engineering                               ║
║    IPO Architecture: Input → Process → Output                       ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def run_pipeline():
    print(BANNER)
    start_time = time.time()

    # ── Paths ─────────────────────────────────────────────────────────────────
    base         = os.path.dirname(os.path.abspath(__file__))
    raw_path     = os.path.join(base, 'data', 'raw',       'employee_data_raw.csv')
    p1_out       = os.path.join(base, 'data', 'processed', 'phase1_clean.csv')
    p2_out       = os.path.join(base, 'data', 'processed', 'phase2_engineered.csv')
    final_out    = os.path.join(base, 'data', 'processed', 'final_feature_store.csv')
    report_out   = os.path.join(base, 'outputs',           'pipeline_report.json')

    os.makedirs(os.path.join(base, 'data', 'processed'), exist_ok=True)
    os.makedirs(os.path.join(base, 'outputs'),            exist_ok=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — INPUT: Securing Fidelity
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("  STAGE 1 — INPUT: SECURING FIDELITY")
    print("▓" * 65)

    df_raw = pd.read_csv(raw_path)
    print(f"\n  Raw dataset loaded: {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")

    # Audit → Impute → Neutralize Outliers
    audit_missingness(df_raw)
    df_clean = handle_missing_values(df_raw)

    numeric_targets = ['monthly_salary', 'hours_worked_per_week',
                       'num_projects', 'performance_score',
                       'years_experience', 'training_hours']
    df_clean = neutralize_outliers(df_clean, numeric_targets)
    df_clean.to_csv(p1_out, index=False)
    print(f"\n  💾 Stage 1 saved → phase1_clean.csv  ({df_clean.shape})")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — PROCESS: Vectorized Computation Engine
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("  STAGE 2 — PROCESS: VECTORIZED COMPUTATION ENGINE")
    print("▓" * 65 + "\n")

    df_engineered = engineer_features(df_clean)
    df_encoded    = encode_categoricals(df_engineered, drop_first=True)
    df_no_collin  = eradicate_collinearity(df_encoded, target_col='left_company', threshold=0.80)
    df_scaled, _  = scale_features(df_no_collin, target_col='left_company', method='standard')
    df_scaled.to_csv(p2_out, index=False)
    print(f"  💾 Stage 2 saved → phase2_engineered.csv  ({df_scaled.shape})")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — OUTPUT: Contracts & Serving
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "▓" * 65)
    print("  STAGE 3 — OUTPUT: STRUCTURAL CONTRACTS & SERVING")
    print("▓" * 65 + "\n")

    df_validated, is_valid = validate_dataset(df_scaled)
    import json
    report = generate_summary_report(df_raw, df_validated, is_valid)
    df_validated.to_csv(final_out, index=False)
    with open(report_out, 'w') as f:
        json.dump(report, f, indent=2)

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPLETION
    # ═══════════════════════════════════════════════════════════════════════════
    elapsed = round(time.time() - start_time, 2)
    print("\n" + "═" * 65)
    print("  🎯  PIPELINE EXECUTION COMPLETE")
    print("═" * 65)
    print(f"\n  Raw input shape        : {df_raw.shape}")
    print(f"  Final feature store    : {df_validated.shape}")
    print(f"  Rows retained          : {round(len(df_validated)/len(df_raw)*100,1)}%")
    print(f"  New features added     : 7")
    print(f"  Schema validation      : {'✅ PASSED' if is_valid else '⚠️  Review failures'}")
    print(f"  Total execution time   : {elapsed}s")
    print(f"\n  Outputs:")
    print(f"    📁 data/processed/phase1_clean.csv")
    print(f"    📁 data/processed/phase2_engineered.csv")
    print(f"    📁 data/processed/final_feature_store.csv")
    print(f"    📋 outputs/pipeline_report.json")
    print("\n" + "═" * 65 + "\n")


if __name__ == "__main__":
    run_pipeline()
