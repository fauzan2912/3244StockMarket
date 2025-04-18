import streamlit as st
import pandas as pd
from pathlib import Path

# -------------------------------
# 1. Page configuration
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
# 3. Helper functions
# -------------------------------
@st.cache_data
def list_stocks():
    """List all stock directories under results/, excluding the stocks_code folder."""
    return sorted(
        [d.name for d in BASE_DIR.iterdir() if d.is_dir() and d.name != PKL_BASE_DIR.name]
    )

@st.cache_data
def list_metrics_models(stock: str):
    """List models available for metrics based on CSV filenames in results/{stock}/."""
    folder = BASE_DIR / stock
    models = set()
    for csv in folder.glob(f"{stock}_*_metrics_summary.csv"):
        parts = csv.stem.split("_")
        if len(parts) >= 3:
            models.add(parts[1])
    return sorted(models)

@st.cache_data
def load_metrics_csv(stock: str, model: str):
    """Load metrics summary CSV for a specific stock/model."""
    path = BASE_DIR / stock / f"{stock}_{model}_metrics_summary.csv"
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def list_pickle_models():
    """List all model subdirectories under results/stocks_code/."""
    return sorted([d.name for d in PKL_BASE_DIR.iterdir() if d.is_dir()])

@st.cache_data
def list_pickles_for_model(model: str):
    """List all .pkl files under results/stocks_code/{model}/."""
    folder = PKL_BASE_DIR / model
    return sorted(folder.glob("*.pkl")) if folder.exists() else []

# -------------------------------
# 4. Sidebar: Filters
# -------------------------------
st.sidebar.header("Metrics Filters")

# 4a. Stock selector for metrics
stocks = list_stocks()
if not stocks:
    st.sidebar.error("No stock folders found under results/. Generate metrics first.")
    st.stop()
stock = st.sidebar.selectbox("Select stock", stocks)

# 4b. Model selector for metrics (from metrics CSVs)
metrics_models = list_metrics_models(stock)
if not metrics_models:
    st.sidebar.error(f"No metrics found for '{stock}'.")
    st.stop()
metrics_model = st.sidebar.selectbox("Select model for metrics", metrics_models)

# -------------------------------
# 5. Sidebar: Pickle Downloader
# -------------------------------
st.sidebar.header("Pickle Downloader")

# 5a. Model directories for pickles
pickle_models = list_pickle_models()
if not pickle_models:
    st.sidebar.error("No model directories found under results/stocks_code/.")
    st.stop()
pickle_model = st.sidebar.selectbox("Select model directory for pickles", pickle_models)

# 5b. Pickle files under chosen model directory
pickles = list_pickles_for_model(pickle_model)
if pickles:
    selected_pkl = st.sidebar.selectbox(
        "Select .pkl file to download", [p.name for p in pickles]
    )
    pkl_path = PKL_BASE_DIR / pickle_model / selected_pkl
    data = pkl_path.read_bytes()
    st.sidebar.download_button(
        "Download pickle", data=data, file_name=selected_pkl,
        mime="application/octet-stream"
    )
else:
    st.sidebar.info(f"No .pkl files found in stocks_code/{pickle_model}/.")

# -------------------------------
# 6. Main page: Title
# -------------------------------
st.title(f"📈 {stock} — {metrics_model.upper()} Strategy Dashboard")

# -------------------------------
# 7. Display metrics table
# -------------------------------
st.subheader("Per-Window Metrics")
metrics_df = load_metrics_csv(stock, metrics_model)
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info("Metrics CSV not found for this selection. Generate with pipeline first.")

# -------------------------------
# 8. Display cumulative performance chart
# -------------------------------
st.subheader("Cumulative Performance Chart")
chart_path = BASE_DIR / stock / f"{stock}_{metrics_model}_full_comparison.png"
if chart_path.exists():
    st.image(
        str(chart_path),
        caption=f"{stock} {metrics_model} (Rolling vs Expanding vs Buy & Hold)",
        use_column_width=True
    )
else:
    st.info("Performance chart not found for this selection.")
