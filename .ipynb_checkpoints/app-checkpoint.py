import streamlit as st
import pandas as pd
from pathlib import Path

# 1. Page configuration
st.set_page_config(
    page_title="Stock Forecasting Dashboard",
    layout="wide",
)

# 2. Directories
RESULTS_DIR = Path("results")
PKL_DIR = RESULTS_DIR / "stocks_code"

# 3. Helper functions for metrics
@st.cache_data
def list_stocks():
    """Return stock folders under results/, excluding the stocks_code directory."""
    return sorted([
        p.name for p in RESULTS_DIR.iterdir()
        if p.is_dir() and p.name != PKL_DIR.name
    ])

@st.cache_data
def list_models(stock: str):
    """Return model names for a given stock by parsing metrics CSV filenames."""
    models = set()
    folder = RESULTS_DIR / stock
    for csv in folder.glob(f"{stock}_*_metrics_summary.csv"):
        parts = csv.stem.split("_")
        if len(parts) >= 3:
            models.add(parts[1])
    return sorted(models)

@st.cache_data
def load_metrics(stock: str, model: str):
    """Load the metrics summary CSV for the selected stock/model."""
    path = RESULTS_DIR / stock / f"{stock}_{model}_metrics_summary.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

# 4. Sidebar filters for metrics
st.sidebar.header("Metrics Filters")
stocks = list_stocks()
if not stocks:
    st.sidebar.error("No stock folders under results/. Please generate metrics first.")
    st.stop()
stock = st.sidebar.selectbox("Select stock", stocks)

models = list_models(stock)
if not models:
    st.sidebar.error(f"No models found for '{stock}'.")
    st.stop()
model = st.sidebar.selectbox("Select model", models)

# 5. Page title
st.title(f"📈 {stock} — {model.upper()} Strategy Dashboard")

# 6. Display metrics table
metrics = load_metrics(stock, model)
st.subheader("Per-Window Metrics")
if metrics is not None:
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("No metrics CSV found; skipping metrics table.")

# 7. Display performance chart
st.subheader("Cumulative Performance")
chart_path = RESULTS_DIR / stock / f"{stock}_{model}_full_comparison.png"
if chart_path.exists():
    st.image(
        str(chart_path),
        caption=f"{stock} {model} (Rolling vs Expanding vs Buy & Hold)",
        use_column_width=True
    )
else:
    st.info("No performance chart found; skipping.")

# 8. Nested Pickle Downloader
st.sidebar.header("Pickle Downloader")
if PKL_DIR.exists():
    model_dirs = sorted([d.name for d in PKL_DIR.iterdir() if d.is_dir()])
    if model_dirs:
        sel_model_dir = st.sidebar.selectbox("Select model directory", model_dirs)
        pkl_paths = sorted((PKL_DIR / sel_model_dir).glob("*.pkl"))
        if pkl_paths:
            sel_pkl = st.sidebar.selectbox("Select a .pkl file", [f.name for f in pkl_paths])
            pkl_path = PKL_DIR / sel_model_dir / sel_pkl
            data = pkl_path.read_bytes()
            st.sidebar.download_button(
                label="Download pickle",
                data=data,
                file_name=sel_pkl,
                mime="application/octet-stream"
            )
        else:
            st.sidebar.info(f"No .pkl files in {PKL_DIR / sel_model_dir}")
    else:
        st.sidebar.info(f"No model directories found in {PKL_DIR}")
else:
    st.sidebar.error(f"{PKL_DIR} not found. Run pipeline to generate pickles.")
