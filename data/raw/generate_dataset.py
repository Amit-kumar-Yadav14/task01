"""
Dataset Generator — DecodeLabs Internship Project 1
Generates a realistic Employee/HR Analytics dataset with intentional
missing values and outliers for the EDA & Feature Engineering pipeline.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

departments    = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance', 'Operations']
education_lvls = ['High School', 'Bachelor', 'Master', 'PhD']
cities         = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Pune']

df = pd.DataFrame({
    'employee_id'         : range(1001, 1001 + N),
    'age'                 : np.random.randint(22, 60, N).astype(float),
    'years_experience'    : np.random.randint(0, 35, N).astype(float),
    'monthly_salary'      : np.random.normal(75000, 20000, N),
    'performance_score'   : np.random.uniform(1, 10, N),
    'hours_worked_per_week': np.random.normal(45, 8, N),
    'num_projects'        : np.random.randint(1, 15, N).astype(float),
    'training_hours'      : np.random.randint(0, 120, N).astype(float),
    'satisfaction_score'  : np.random.uniform(1, 5, N),
    'department'          : np.random.choice(departments, N),
    'education_level'     : np.random.choice(education_lvls, N),
    'city'                : np.random.choice(cities, N),
    'left_company'        : np.random.choice([0, 1], N, p=[0.75, 0.25]),
})

# ── Introduce Missing Values (realistic proportions) ──────────────────────────
# < 5%  → Row Deletion territory
df.loc[np.random.choice(N, int(N * 0.03), replace=False), 'training_hours'] = np.nan

# 5–20% → Statistical Imputation territory
df.loc[np.random.choice(N, int(N * 0.12), replace=False), 'monthly_salary']  = np.nan
df.loc[np.random.choice(N, int(N * 0.08), replace=False), 'satisfaction_score'] = np.nan

# > 20% → KNN Imputation territory
df.loc[np.random.choice(N, int(N * 0.22), replace=False), 'performance_score'] = np.nan

# ── Introduce Outliers ─────────────────────────────────────────────────────────
outlier_idx = np.random.choice(N, 15, replace=False)
df.loc[outlier_idx[:5],  'monthly_salary']       = np.random.uniform(400000, 600000, 5)
df.loc[outlier_idx[5:10], 'hours_worked_per_week'] = np.random.uniform(90, 120, 5)
df.loc[outlier_idx[10:], 'num_projects']          = np.random.randint(50, 100, 5)

df.to_csv('employee_data_raw.csv', index=False)
print(f"✅ Dataset generated: {df.shape[0]} rows × {df.shape[1]} cols")
print(f"\nMissing value counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
