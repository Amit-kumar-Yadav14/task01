"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  EDA Visualizations — Exploratory Data Analysis Report                     ║
║  DecodeLabs · Data Science Project 1 · Industrial Training Kit             ║
╚══════════════════════════════════════════════════════════════════════════════╝
Generates publication-quality plots:
  1. Missingness heatmap
  2. Distribution plots (before vs. after imputation)
  3. IQR outlier boxplots
  4. Correlation heatmap (before collinearity eradication)
  5. Engineered feature distributions
  6. Target class balance
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings, os
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor' : '#0d1117',
    'axes.facecolor'   : '#161b22',
    'axes.edgecolor'   : '#30363d',
    'axes.labelcolor'  : '#e6edf3',
    'xtick.color'      : '#8b949e',
    'ytick.color'      : '#8b949e',
    'text.color'       : '#e6edf3',
    'grid.color'       : '#21262d',
    'grid.alpha'       : 0.5,
    'font.family'      : 'DejaVu Sans',
})
ACCENT   = '#00d4ff'
ACCENT2  = '#f0c050'
DANGER   = '#ff6b6b'
SUCCESS  = '#3fb950'

BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT_DIR  = os.path.join(BASE_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)


def savefig(name: str):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight',
                facecolor=plt.rcParams['figure.facecolor'])
    plt.close()
    print(f"  📊 Saved → outputs/{name}")


# ─────────────────────────────────────────────────────────────────────────────

def plot_missingness(df_raw: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Missing Data Analysis — Decision Matrix Applied",
                 fontsize=14, fontweight='bold', color=ACCENT, y=1.02)

    # Bar chart
    missing = (df_raw.isnull().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    colors  = [DANGER if v > 20 else ACCENT2 if v >= 5 else SUCCESS for v in missing.values]
    axes[0].barh(missing.index, missing.values, color=colors, edgecolor='#30363d')
    axes[0].axvline(5,  color=ACCENT2, ls='--', lw=1.5, label='5% threshold')
    axes[0].axvline(20, color=DANGER,  ls='--', lw=1.5, label='20% threshold')
    axes[0].set_xlabel('Missing %', color='#8b949e')
    axes[0].set_title('Missingness by Feature', fontsize=11, color='#e6edf3')
    axes[0].legend(fontsize=9)
    for bar, val in zip(axes[0].patches, missing.values):
        axes[0].text(val + 0.3, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', va='center', fontsize=9, color='#e6edf3')

    # Strategy pie
    tiers = {
        f'< 5%\nRow Deletion\n({(missing < 5).sum()} features)'  : (missing < 5).sum(),
        f'5–20%\nStatistical Imputation\n({((missing >= 5) & (missing <= 20)).sum()} features)': ((missing >= 5) & (missing <= 20)).sum(),
        f'> 20%\nKNN Estimation\n({(missing > 20).sum()} features)'  : (missing > 20).sum(),
    }
    pie_vals = [v for v in tiers.values() if v > 0]
    pie_lbls = [k for k, v in tiers.items() if v > 0]
    pie_cols = [SUCCESS, ACCENT2, DANGER][:len(pie_vals)]
    axes[1].pie(pie_vals, labels=pie_lbls, colors=pie_cols,
                autopct='%1.0f%%', startangle=140,
                textprops={'color': '#e6edf3', 'fontsize': 8},
                wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2})
    axes[1].set_title('Imputation Strategy Distribution', fontsize=11, color='#e6edf3')

    plt.tight_layout()
    savefig("01_missingness_analysis.png")


def plot_outlier_boxplots(df_raw: pd.DataFrame, df_clean: pd.DataFrame):
    cols = ['monthly_salary', 'hours_worked_per_week', 'num_projects']
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Outlier Analysis — Before vs. After IQR Winsorization",
                 fontsize=13, fontweight='bold', color=ACCENT)

    for i, col in enumerate(cols):
        for j, (data, label, color) in enumerate([
            (df_raw[col].dropna(),   'BEFORE', DANGER),
            (df_clean[col].dropna(), 'AFTER',  SUCCESS),
        ]):
            ax = axes[j][i]
            bp = ax.boxplot(data, vert=True, patch_artist=True,
                            medianprops=dict(color=ACCENT2, linewidth=2),
                            boxprops=dict(facecolor=color + '33', edgecolor=color),
                            whiskerprops=dict(color='#8b949e'),
                            capprops=dict(color='#8b949e'),
                            flierprops=dict(marker='o', color=DANGER,
                                            markersize=3, alpha=0.5))
            ax.set_title(f"{col}\n({label})", fontsize=9, color='#e6edf3')
            ax.set_xticks([])
            n_out = len(data[(data < data.quantile(0.25) - 1.5*(data.quantile(0.75)-data.quantile(0.25))) |
                             (data > data.quantile(0.75) + 1.5*(data.quantile(0.75)-data.quantile(0.25)))])
            ax.set_xlabel(f"Outliers: {n_out}", fontsize=8, color='#8b949e')

    plt.tight_layout()
    savefig("02_outlier_boxplots.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric = df.select_dtypes(include=np.number)
    # Select top features for readability
    top_feats = numeric.columns[:14].tolist()
    corr = numeric[top_feats].corr()

    fig, ax = plt.subplots(figsize=(13, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt='.2f', annot_kws={'size': 8},
                linewidths=0.5, linecolor='#21262d',
                ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title("Pearson Correlation Matrix — Collinearity Detection\n"
                 "(pairs > |0.80| flagged for eradication)",
                 fontsize=12, fontweight='bold', color=ACCENT, pad=15)

    # Highlight high-corr cells
    for i in range(len(corr)):
        for j in range(i):
            if abs(corr.iloc[i, j]) > 0.80:
                ax.add_patch(plt.Rectangle((j, i), 1, 1,
                                           fill=False, edgecolor=DANGER,
                                           lw=2.5, clip_on=False))
    plt.tight_layout()
    savefig("03_correlation_heatmap.png")


def plot_engineered_features(df: pd.DataFrame):
    feats = ['salary_per_year_exp', 'productivity_index',
             'experience_age_ratio', 'training_investment_score',
             'satisfaction_performance_composite']
    feats = [f for f in feats if f in df.columns]

    fig, axes = plt.subplots(1, len(feats), figsize=(16, 4))
    fig.suptitle("Engineered Feature Distributions (Post-Scaling)",
                 fontsize=13, fontweight='bold', color=ACCENT)

    colors = [ACCENT, ACCENT2, SUCCESS, '#b392f0', '#ffa657']
    for i, (feat, color) in enumerate(zip(feats, colors)):
        data = df[feat].dropna()
        axes[i].hist(data, bins=40, color=color + '99', edgecolor=color, lw=0.5)
        axes[i].axvline(data.mean(),   color='white',  ls='--', lw=1.5, label='Mean')
        axes[i].axvline(data.median(), color=ACCENT2,  ls=':',  lw=1.5, label='Median')
        axes[i].set_title(feat.replace('_', '\n'), fontsize=8, color='#e6edf3')
        axes[i].set_xlabel('Value', fontsize=7, color='#8b949e')
        if i == 0:
            axes[i].legend(fontsize=7)

    plt.tight_layout()
    savefig("04_engineered_features.png")


def plot_target_balance(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Target Variable Distribution — Employee Attrition",
                 fontsize=13, fontweight='bold', color=ACCENT)

    counts = df['left_company'].value_counts().sort_index()
    labels = ['Stayed (0)', 'Left (1)']
    colors = [SUCCESS, DANGER]

    # Bar
    axes[0].bar(labels, counts.values, color=colors, edgecolor='#30363d', width=0.5)
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                     str(val), ha='center', fontsize=11, color='#e6edf3', fontweight='bold')
    axes[0].set_title('Class Counts', fontsize=10, color='#e6edf3')
    axes[0].set_ylabel('Count', color='#8b949e')

    # Pie
    axes[1].pie(counts.values, labels=labels, colors=colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'color': '#e6edf3', 'fontsize': 10},
                wedgeprops={'edgecolor': '#0d1117', 'linewidth': 2})
    axes[1].set_title('Class Proportion', fontsize=10, color='#e6edf3')

    plt.tight_layout()
    savefig("05_target_balance.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "█" * 65)
    print("  EDA VISUALIZATION REPORT")
    print("  DecodeLabs · Data Science Project 1")
    print("█" * 65 + "\n")

    raw_path   = os.path.join(BASE_DIR, 'data', 'raw',       'employee_data_raw.csv')
    p1_path    = os.path.join(BASE_DIR, 'data', 'processed', 'phase1_clean.csv')
    p2_path    = os.path.join(BASE_DIR, 'data', 'processed', 'phase2_engineered.csv')
    final_path = os.path.join(BASE_DIR, 'data', 'processed', 'final_feature_store.csv')

    df_raw   = pd.read_csv(raw_path)
    df_p1    = pd.read_csv(p1_path)
    df_p2    = pd.read_csv(p2_path)
    df_final = pd.read_csv(final_path)

    print("Generating plots...")
    plot_missingness(df_raw)
    plot_outlier_boxplots(df_raw, df_p1)
    plot_correlation_heatmap(df_p2)
    plot_engineered_features(df_p2)
    plot_target_balance(df_final)

    print(f"\n✅ All 5 visualizations saved to outputs/\n")
