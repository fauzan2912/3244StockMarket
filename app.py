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
    """List all stock directories under results/ except stocks_code."""
    return sorted([
        d.name for d in BASE_DIR.iterdir()
        if d.is_dir() and d.name != PKL_BASE_DIR.name
    ])

@st.cache_data
def list_metrics_models(stock: str):
    """List available model names based on CSV files in results/{stock}/."""
    folder = BASE_DIR / stock
    models = set()
    for csv in folder.glob(f"{stock}_*_metrics_summary.csv"):
        parts = csv.stem.split("_")
        if len(parts) >= 3:
            models.add(parts[1])
    return sorted(models)

@st.cache_data
def load_metrics_df(stock: str, model: str):
    """Load the metrics summary CSV for a given stock/model from results/{stock}/."""
    path = BASE_DIR / stock / f"{stock}_{model}_metrics_summary.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def get_chart_path(stock: str, model: str):
    """Get the path to the performance chart PNG for stock/model in results/{stock}/."""
    path = BASE_DIR / stock / f"{stock}_{model}_full_comparison.png"
    return path if path.exists() else None

@st.cache_data
def list_pkl_models():
    """List model subdirectories under results/stocks_code/."""
    if PKL_BASE_DIR.exists():
        return sorted([d.name for d in PKL_BASE_DIR.iterdir() if d.is_dir()])
    return []

@st.cache_data
def list_pickles_for_model(model: str):
    """List all pickle files under results/stocks_code/{model}/."""
    folder = PKL_BASE_DIR / model
    return sorted(folder.glob("*.pkl")) if folder.exists() else []

# -------------------------------
# 4. Sidebar: Metrics Filters
# -------------------------------
st.sidebar.header("Metrics Filters")

# 4a. Select stock
stocks = list_stocks()
if not stocks:
    st.sidebar.error("No stock folders found under results/. Run your pipeline first.")
    st.stop()
stock = st.sidebar.selectbox("Select Stock", stocks)

# 4b. Select metrics model
metrics_models = list_metrics_models(stock)
if not metrics_models:
    st.sidebar.error(f"No metrics CSVs found for '{stock}'. Run your pipeline first.")
    st.stop()
metrics_model = st.sidebar.selectbox("Select Model for Metrics", metrics_models)

# -------------------------------
# 5. Sidebar: Pickle Downloader
# -------------------------------
st.sidebar.header("Pickle Downloader")

# 5a. Select pickle model directory
pkl_models = list_pkl_models()
if not pkl_models:
    st.sidebar.error("No model dirs under results/stocks_code/. Run pipeline to generate pickles.")
    st.stop()
pkl_model = st.sidebar.selectbox("Select Model for Pickles", pkl_models)

# 5b. Select pickle file
pkl_files = list_pickles_for_model(pkl_model)
if pkl_files:
    selected_pkl = st.sidebar.selectbox("Select Pickle File", [f.name for f in pkl_files])
    data = (PKL_BASE_DIR / pkl_model / selected_pkl).read_bytes()
    st.sidebar.download_button(
        label="Download Pickle",
        data=data,
        file_name=selected_pkl,
        mime="application/octet-stream",
    )
else:
    st.sidebar.info(f"No .pkl files in results/stocks_code/{pkl_model}/.")

# -------------------------------
# 6. Main Page: Title
# -------------------------------
st.title(f"📈 {stock} — {metrics_model.upper()} Strategy Dashboard")

# -------------------------------
# 7. Display Metrics Table
# -------------------------------
st.subheader("Per-Window Metrics")
metrics_df = load_metrics_df(stock, metrics_model)
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("Metrics table not found. Ensure 'results/{stock}/{stock}_{metrics_model}_metrics_summary.csv' exists.")

# -------------------------------
# 8. Display Cumulative Performance Chart
# -------------------------------
st.subheader("Cumulative Performance Chart")
chart_path = get_chart_path(stock, metrics_model)
if chart_path:
    st.image(
        str(chart_path),
        caption=f"{stock} {metrics_model} (Rolling vs Expanding vs Buy & Hold)",
        use_column_width=True,
    )
else:
    st.info("Chart not found. Ensure 'results/{stock}/{stock}_{metrics_model}_full_comparison.png' exists.")