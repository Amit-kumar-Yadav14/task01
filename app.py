import streamlit as st
import pandas as pd
import json
from PIL import Image
import os

st.set_page_config(page_title="Data Pipeline & EDA Showcase", layout="wide")

st.title("🧪 Data Science Project 1 — Advanced EDA & Feature Engineering")
st.markdown("### DecodeLabs Industrial Training Kit · Batch 2026")
st.markdown("---")

# 1. Load Data
@st.cache_data
def load_data(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

st.header("1. Dataset Preview")
st.markdown("This project implements a production-grade data preprocessing and feature engineering pipeline.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Raw Data")
    raw_df = load_data("data/raw/employee_data_raw.csv")
    if raw_df is not None:
        st.dataframe(raw_df.head(10))
    else:
        st.warning("Raw data not found. Please run the pipeline first.")

with col2:
    st.subheader("Final Processed Data (Feature Store)")
    final_df = load_data("data/processed/final_feature_store.csv")
    if final_df is not None:
        st.dataframe(final_df.head(10))
    else:
        st.warning("Processed data not found.")

st.markdown("---")

# 2. Pipeline Report
st.header("2. Pipeline Report")
report_path = "outputs/pipeline_report.json"
if os.path.exists(report_path):
    with open(report_path, "r") as f:
        report = json.load(f)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Raw Rows", report.get("raw_shape", [0])[0])
    col_b.metric("Final Rows", report.get("final_shape", [0])[0])
    col_c.metric("Features Added", report.get("new_features_added", 0))
    col_d.metric("Schema Valid", "✅ Yes" if report.get("schema_valid") else "❌ No")
    
    with st.expander("View Full Report Details"):
        st.json(report)
else:
    st.info("Pipeline report not found.")

st.markdown("---")

# 3. EDA Visualizations
st.header("3. Exploratory Data Analysis")

images = [
    ("Missingness Analysis", "outputs/01_missingness_analysis.png"),
    ("Outlier Boxplots", "outputs/02_outlier_boxplots.png"),
    ("Correlation Heatmap", "outputs/03_correlation_heatmap.png"),
    ("Engineered Features", "outputs/04_engineered_features.png"),
    ("Target Balance", "outputs/05_target_balance.png")
]

tabs = st.tabs([img[0] for img in images])

for i, (title, img_path) in enumerate(images):
    with tabs[i]:
        if os.path.exists(img_path):
            image = Image.open(img_path)
            st.image(image, caption=title, use_container_width=True)
        else:
            st.warning(f"Image not found: {img_path}")

st.markdown("---")
st.markdown("*Built with ❤️ as part of the DecodeLabs Industrial Training Program.*")
