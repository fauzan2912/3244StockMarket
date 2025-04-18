import streamlit as st
import pandas as pd
from pathlib import Path

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Stock Forecasting Dashboard",
    layout="wide",
)

# -------------------------------
# 2. Directories
# -------------------------------
BASE_DIR = Path("results")
PKL_BASE_DIR = BASE_DIR / "stocks_code"

# -------------------------------
# 3. Helper Functions
# -------------------------------
@st.cache_data
def list_stocks():
    """List all stock directories under results/, excluding stocks_code."""
    return sorted([
        d.name for d in BASE_DIR.iterdir()
        if d.is_dir() and d.name != PKL_BASE_DIR.name
    ])

@st.cache_data
def list_models_for_metrics(stock: str):
    """List model subdirectories under results/{stock}/ for metrics."""
    metrics_dir = BASE_DIR / stock
    return sorted([d.name for d in metrics_dir.iterdir() if d.is_dir()])

@st.cache_data
def load_metrics(stock: str, model: str):
    """Load the metrics CSV from results/{stock}/{model}/."""
    path = BASE_DIR / stock / model / f"{stock}_{model}_metrics_summary.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def list_pkl_models():
    """List all model subdirectories under results/stocks_code/."""
    return sorted([d.name for d in PKL_BASE_DIR.iterdir() if d.is_dir()]) if PKL_BASE_DIR.exists() else []

@st.cache_data
def list_pickles_for_model(model: str):
    """List all .pkl files under results/stocks_code/{model}/."""
    folder = PKL_BASE_DIR / model
    return sorted(folder.glob("*.pkl")) if folder.exists() else []

# -------------------------------
# 4. Sidebar: Filters
# -------------------------------
st.sidebar.header("Metrics Filters")

# Stock selection for metrics
stocks = list_stocks()
if not stocks:
    st.sidebar.error("No stock directories found under results/. Run pipeline first.")
    st.stop()
stock = st.sidebar.selectbox("Select Stock for Metrics", stocks)

# Model selection for metrics
metrics_models = list_models_for_metrics(stock)
if not metrics_models:
    st.sidebar.error(f"No model subfolders found under results/{stock}/.")
    st.stop()
metrics_model = st.sidebar.selectbox("Select Model for Metrics", metrics_models)

# -------------------------------
# 5. Sidebar: Pickle Downloader
# -------------------------------
st.sidebar.header("Pickle Downloader")

# Model selection for pickles
pkl_models = list_pkl_models()
if not pkl_models:
    st.sidebar.error("No model directories found under results/stocks_code/. Run pipeline first.")
    st.stop()
pkl_model = st.sidebar.selectbox("Select Model for Pickles", pkl_models)

# Pickle files listing
pickles = list_pickles_for_model(pkl_model)
if pickles:
    selected_pkl = st.sidebar.selectbox(
        "Select .pkl file to download", [p.name for p in pickles]
    )
    pkl_path = PKL_BASE_DIR / pkl_model / selected_pkl
    data = pkl_path.read_bytes()
    st.sidebar.download_button(
        label="Download Pickle",
        data=data,
        file_name=selected_pkl,
        mime="application/octet-stream"
    )
else:
    st.sidebar.info(f"No .pkl files found in results/stocks_code/{pkl_model}/.")

# -------------------------------
# 6. Main Page: Title
# -------------------------------
st.title(f"📈 {stock} — {metrics_model.upper()} Strategy Dashboard")

# -------------------------------
# 7. Display Metrics Table
# -------------------------------
st.subheader("Per-Window Metrics")
metrics_df = load_metrics(stock, metrics_model)
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("Metrics CSV not found in results/{stock}/{metrics_model}/. Run pipeline first.")

# -------------------------------
# 8. Display Cumulative Performance Chart
# -------------------------------
st.subheader("Cumulative Performance Chart")
chart_path = BASE_DIR / stock / metrics_model / f"{stock}_{metrics_model}_full_comparison.png"
if chart_path.exists():
    st.image(
        str(chart_path),
        caption=f"{stock} {metrics_model} (Rolling vs Expanding vs Buy & Hold)",
        use_column_width=True
    )
else:
    st.info("Performance chart not found in results/{stock}/{metrics_model}/. Run pipeline first.")