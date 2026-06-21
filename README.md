# 🧪 Data Science Project 1 — Advanced EDA & Feature Engineering
### DecodeLabs Industrial Training Kit · Batch 2026

---

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://python.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org)
[![Pandera](https://img.shields.io/badge/Pandera-Schema%20Validation-green?style=for-the-badge)](https://pandera.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

---

## 🎯 Project Overview

This project implements a **production-grade data preprocessing and feature engineering pipeline** following the **Input → Process → Output (IPO) Architecture**. The goal is to transform raw, chaotic employee data into a mathematically clean feature store ready for machine learning algorithms — with zero tolerance for data corruption.

> *"Data preprocessing is not janitorial work; it is the structural engineering of mathematical truth."*

---

## 🏗️ Architecture: The IPO Blueprint

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   MODULE 1: INPUT   │    │  MODULE 2: PROCESS   │    │  MODULE 3: OUTPUT   │
│  Securing Fidelity  │───▶│    The Engine        │───▶│ Contracts & Serving │
│                     │    │                     │    │                     │
│ • Missing values    │    │ • Vectorized math   │    │ • Pandera schemas   │
│ • Outlier bounds    │    │ • OHE Encoding      │    │ • Feature store     │
│ • IQR caps          │    │ • Collinearity fix  │    │ • Pipeline report   │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
```

---

## 📁 Project Structure

```
ds_project1/
│
├── data/
│   ├── raw/
│   │   ├── generate_dataset.py          ← Synthetic dataset generator
│   │   └── employee_data_raw.csv        ← Raw dataset (1000 rows × 13 cols)
│   └── processed/
│       ├── phase1_clean.csv             ← After imputation + outlier removal
│       ├── phase2_engineered.csv        ← After feature engineering + OHE
│       └── final_feature_store.csv      ← Validated, production-ready store
│
├── src/
│   ├── phase1_input_fidelity.py         ← Missing values + outlier pipeline
│   ├── phase2_computation_engine.py     ← Feature engineering + encoding
│   ├── phase3_contracts_scaling.py      ← Pandera validation + report
│   └── eda_visualizations.py           ← Publication-quality EDA plots
│
├── outputs/
│   ├── 01_missingness_analysis.png
│   ├── 02_outlier_boxplots.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_engineered_features.png
│   ├── 05_target_balance.png
│   └── pipeline_report.json
│
├── tests/
│   └── test_pipeline.py                 ← 20+ unit & integration tests
│
├── run_pipeline.py                      ← 🚀 One-command full pipeline runner
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/decodelabs-ds-project1.git
cd decodelabs-ds-project1

# 2. Create & activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the raw dataset
cd data/raw && python generate_dataset.py && cd ../..
```

---

## 🚀 Quick Start

```bash
# Run the complete pipeline in one command
python run_pipeline.py

# Generate all EDA visualizations
python src/eda_visualizations.py

# Run the test suite
pytest tests/test_pipeline.py -v
```

---

## 📐 Phase 1 — Securing Input Fidelity

### Missing Data Decision Matrix

| Missingness | Strategy | Rationale |
|---|---|---|
| **< 5%** | Row Deletion (`dropna`) | Preserves data volume, prevents synthetic bias |
| **5–20% (skewed)** | Global Median Imputation | Robust against extreme outliers |
| **5–20% (correlated)** | Group-Wise Conditional | Retains variance patterns across sub-populations |
| **> 20%** | KNN Imputation (k=5) | Captures complex multi-dimensional relationships |

### Outlier Neutralization — IQR Winsorization

```
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

**Winsorization over Deletion:** `numpy.clip()` caps outlier values at the IQR bounds rather than deleting rows. This preserves row count and sequential data integrity — critical when downstream models require strict temporal sequences.

---

## ⚡ Phase 2 — Vectorized Computation Engine

### Why Vectorization? No for-loops.

```python
# ❌ WRONG — Python loop: O(N) interpreter overhead per iteration
for i, row in df.iterrows():
    df.at[i, 'ratio'] = row['salary'] / row['age']

# ✅ CORRECT — Vectorized: compiled C-level SIMD, block-allocated RAM
df['ratio'] = df['salary'] / df['age']       # 100–1000x faster
```

### Engineered Features (7 new predictive signals)

| Feature | Formula | Rationale |
|---|---|---|
| `salary_per_year_exp` | `salary / years_exp` | Compensation efficiency |
| `productivity_index` | `(projects / hours) × 100` | Output rate per hour |
| `experience_age_ratio` | `years_exp / age` | Career start proxy |
| `training_investment_score` | `(training_hrs/max) × satisfaction × 10` | Learning engagement index |
| `satisfaction_performance_composite` | `√(sat_scaled × perf_scaled)` | Geometric mean health index |
| `seniority_band` | `pd.cut(years_exp, [0,2,7,15,∞])` | Ordinal career stage |
| `overwork_flag` | `hours > 55 → 1 else 0` | Work-life balance signal |

### One-Hot Encoding — Orthogonal Coordinate Space

Label Encoding assigns integers (London=1, Paris=2, Tokyo=3), implying false mathematical distances. OHE maps each category to an orthogonal axis with equidistant geometric distance (√2), eliminating artificial spatial hierarchy.

### Collinearity Eradication — 4-Step Algorithm

```
Step 1 → Build absolute Pearson correlation matrix
Step 2 → Isolate upper triangle (avoid duplicate pairs)
Step 3 → Identify pairs with |corr| > 0.80
Step 4 → Drop the feature with LOWER correlation to target variable
```

---

## 🛡️ Phase 3 — Structural Contracts (Pandera)

```python
import pandera as pa

schema = pa.DataFrameSchema({
    "performance_score": pa.Column(float, nullable=False),
    "overwork_flag":     pa.Column(int, pa.Check(lambda s: s.isin([0,1]).all())),
    ...
})

# lazy=True → collects ALL failures before raising, never crashes on first error
validated_df = schema.validate(df, lazy=True)
```

**Why Pandera:** Silent data corruption is the #1 cause of training-serving skew. Schema contracts assert mathematical invariants at runtime before data reaches any downstream estimator.

---

## 📊 EDA Output Visualizations

| Plot | Description |
|---|---|
| `01_missingness_analysis.png` | Feature missingness bar chart + strategy pie |
| `02_outlier_boxplots.png` | Before/after IQR Winsorization comparison |
| `03_correlation_heatmap.png` | Pearson matrix with collinear pairs highlighted |
| `04_engineered_features.png` | Distribution of all 7 new features |
| `05_target_balance.png` | Target class balance (attrition labels) |

---

## 🧪 Test Suite

```bash
pytest tests/test_pipeline.py -v
```

**20+ tests covering:**
- Missingness audit accuracy
- Imputation completeness (zero NaN post-processing)
- Row retention after imputation (>90%)
- IQR bounds mathematical correctness
- Winsorization boundary enforcement
- All 7 engineered features present
- `overwork_flag` and `seniority_band` value constraints
- OHE produces binary columns only
- End-to-end integration smoke test

---

## 🧠 Key Concepts Demonstrated

- **Mathematical Fidelity:** ML estimators operate on real-numbered coordinate spaces. Low-fidelity input data causes algorithms to optimize for wrong patterns.
- **Statistical Imputation Hierarchy:** Each strategy introduces a trade-off — the chosen method preserves the natural relationship between variables.
- **Multicollinearity:** When predictor variables are highly correlated, X^T X becomes singular (non-invertible), making OLS coefficients unstable.
- **Data Contracts:** Treating data pipelines as critical software interfaces with runtime-enforced schema assertions.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **Pandas** | DataFrame operations, groupby, vectorized transforms |
| **NumPy** | Block-allocated array math, SIMD vectorization, clip |
| **scikit-learn** | KNNImputer, StandardScaler, MinMaxScaler |
| **SciPy** | Statistical functions (skewness, distributions) |
| **Pandera** | Runtime schema contracts and validation |
| **Matplotlib / Seaborn** | EDA visualization |
| **pytest** | Unit and integration testing |

---

## 👤 Author

**AMIT KUMAR**  
Data Science Intern — DecodeLabs · Batch 2026  
📧 [amityadav21122@gmail.com] 

---

*Built with ❤️ as part of the DecodeLabs Industrial Training Program.*  
*Powered by DecodeLabs — Greater Lucknow, India*
