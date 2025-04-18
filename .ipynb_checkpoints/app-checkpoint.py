# app.py
import streamlit as st
import pandas as pd
from pathlib import Path

# 1. Configure pages
st.set_page_config(
    page_title="Stock Forecasting Dashboard",
    layout="wide",
)

# 2. Points to your results folder
RESULTS_DIR = Path("results")

# 3. Helpers
@st.cache_data
def list_stocks():
    return sorted([p.name for p in RESULTS_DIR.iterdir() if p.is_dir()])

@st.cache_data
def list_models(stock: str):
    folder = RESULTS_DIR / stock
    models = set()
    for csv in folder.glob(f"{stock}_*_metrics_summary.csv"):
        # filename: AAPL_svm_metrics_summary.csv
        parts = csv.stem.split("_")
        if len(parts) >= 3:
            models.add(parts[1])
    return sorted(models)

@st.cache_data
def load_metrics(stock: str, model: str) -> pd.DataFrame:
    path = RESULTS_DIR / stock / f"{stock}_{model}_metrics_summary.csv"
    return pd.read_csv(path)

# 4. Sidebar widgets
st.sidebar.header("Filters")
stock = st.sidebar.selectbox("Select stock", list_stocks())
model = st.sidebar.selectbox("Select model", list_models(stock))

# 5. Main display
st.title(f"📈 {stock} — {model.upper()} Strategy Comparison")

# 5a. Metrics table
st.subheader("Per‑Window Metrics")
df = load_metrics(stock, model)
st.dataframe(df, use_container_width=True)

# 5b. Cumulative performance chart
st.subheader("Cumulative Performance")
img_path = RESULTS_DIR / stock / f"{stock}_{model}_full_comparison.png"
st.image(str(img_path), caption=f"{stock} {model} (Rolling vs Expanding vs Buy & Hold)", use_column_width=True)
