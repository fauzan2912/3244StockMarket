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
PKL_BASE_DIR = RESULTS_DIR / "stocks_code"

# 3. Helper functions for metrics
@st.cache_data
def list_stocks():
    """Return stock folders under results/, excluding stocks_code directory."""
    return sorted([
        p.name for p in RESULTS_DIR.iterdir()
        if p.is_dir() and p.name != PKL_BASE_DIR.name
    ])

@st.cache_data
def list_models(stock: str):
    """Return model names for a given stock by parsing metrics CSV filenames."""
    folder = RESULTS_DIR / stock
    models = set()
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

# 4. Sidebar: Metrics Filters
st.sidebar.header("Metrics Filters")
stocks = list_stocks()
if not stocks:
    st.sidebar.error("No stock folders under results/. Please generate metrics first.")
    st.stop()
stock = st.sidebar.selectbox("Select stock for metrics", stocks, key="metrics_stock")

models = list_models(stock)
if not models:
    st.sidebar.error(f"No models found for '{stock}'.")
    st.stop()
model = st.sidebar.selectbox("Select model for metrics", models, key="metrics_model")

# 5. Sidebar: Pickle Downloader (based on selected model)
st.sidebar.header("Pickle Downloader")
pkl_dir = PKL_BASE_DIR / model
if pkl_dir.exists() and pkl_dir.is_dir():
    pkl_files = sorted(pkl_dir.glob("*.pkl"))
    if pkl_files:
        selected = st.sidebar.selectbox(
            "Select .pkl file to download", [p.name for p in pkl_files], key="pkl_file"
        )
        pkl_path = pkl_dir / selected
        data = pkl_path.read_bytes()
        st.sidebar.download_button(
            label="Download selected pickle",
            data=data,
            file_name=selected,
            mime="application/octet-stream"
        )
    else:
        st.sidebar.info(f"No .pkl files found in {pkl_dir}.")
else:
    st.sidebar.error(f"Pickle directory not found for model '{model}'. Expected at {pkl_dir}.")

# 6. Main page: Title
st.title(f"📈 {stock} — {model.upper()} Strategy Dashboard")

# 7. Display per-window metrics
st.subheader(f"Per-Window Metrics for {stock} / {model}")
metrics = load_metrics(stock, model)
if metrics is not None:
    st.dataframe(metrics, use_container_width=True)
else:
    st.info("No metrics CSV found for this stock/model. Skipping metrics table.")

# 8. Display cumulative performance chart
st.subheader("Cumulative Performance Chart")
chart_path = RESULTS_DIR / stock / f"{stock}_{model}_full_comparison.png"
if chart_path.exists():
    st.image(
        str(chart_path),
        caption=f"{stock} {model} (Rolling vs Expanding vs Buy & Hold)",
        use_column_width=True
    )
else:
    st.info("Performance chart not found. Skipping chart.")
