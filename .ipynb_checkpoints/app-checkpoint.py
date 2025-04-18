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
# Use current working directory to locate 'results'
BASE_DIR = Path.cwd() / "results"
PKL_BASE_DIR = BASE_DIR / "stocks_code"

# -------------------------------
# 3. Helper Functions
# -------------------------------
@st.cache_data
def list_models():
    """List model directories under results/stocks_code/."""
    if not PKL_BASE_DIR.exists():
        return []
    return sorted(d.name for d in PKL_BASE_DIR.iterdir() if d.is_dir())

@st.cache_data
def list_stocks_for_model(model: str):
    """Extract stock codes from metrics CSV filenames in a given model directory."""
    model_dir = PKL_BASE_DIR / model
    if not model_dir.exists():
        return []
    files = list(model_dir.glob("*_metrics_summary.csv"))
    stocks = {f.stem.split("_")[0] for f in files}
    return sorted(stocks)

@st.cache_data
def load_metrics_df(model: str, stock: str):
    """Load the metrics summary CSV for a given model and stock."""
    model_dir = PKL_BASE_DIR / model
    files = list(model_dir.glob(f"{stock}_*metrics_summary.csv"))
    if files:
        return pd.read_csv(files[0])
    return None

@st.cache_data
def get_chart_path(model: str, stock: str):
    """Get the performance chart PNG path for a given model and stock."""
    model_dir = PKL_BASE_DIR / model
    files = list(model_dir.glob(f"{stock}_*full_comparison.png"))
    return files[0] if files else None

@st.cache_data
def list_pickles_for_model_stock(model: str, stock: str):
    """List all pickle files for a given model and stock."""
    model_dir = PKL_BASE_DIR / model
    return sorted(model_dir.glob(f"{stock}_*.pkl")) if model_dir.exists() else []

# -------------------------------
# 4. Sidebar: Filters
# -------------------------------
st.sidebar.header("Filters")

# 4a. Select Model
models = list_models()
if not models:
    st.sidebar.error(f"No model directories under {PKL_BASE_DIR}. Run pipeline first.")
    st.stop()
model = st.sidebar.selectbox("Select Model", models)

# 4b. Select Stock based on model
stocks = list_stocks_for_model(model)
if not stocks:
    st.sidebar.error(f"No metrics files found in {PKL_BASE_DIR/model}. Run pipeline first.")
    st.stop()
stock = st.sidebar.selectbox("Select Stock", stocks)

# -------------------------------
# 5. Main Page: Title
# -------------------------------
st.title(f"📈 {stock} — {model.upper()} Strategy Dashboard")

# -------------------------------
# 6. Display Metrics Table
# -------------------------------
st.subheader("Per-Window Metrics")
metrics_df = load_metrics_df(model, stock)
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info(f"No metrics CSV found for {stock} in model {model} under {PKL_BASE_DIR/model}.")

# -------------------------------
# 7. Display Cumulative Performance Chart
# -------------------------------
st.subheader("Cumulative Performance Chart")
chart_path = get_chart_path(model, stock)
if chart_path:
    st.image(
        str(chart_path),
        caption=f"{stock} {model} (Rolling vs Expanding vs Buy & Hold)",
        use_column_width=True,
    )
else:
    st.info(f"No chart PNG found for {stock} in model {model} under {PKL_BASE_DIR/model}.")

# -------------------------------
# 8. Download Model Pickle
# -------------------------------
st.subheader("Download Trained Model Pickle")
pkl_files = list_pickles_for_model_stock(model, stock)
if pkl_files:
    selected_pkl = st.selectbox("Select Pickle File to Download", [p.name for p in pkl_files])
    pkl_path = PKL_BASE_DIR / model / selected_pkl
    data = pkl_path.read_bytes()
    st.download_button(
        label="Download Pickle",
        data=data,
        file_name=selected_pkl,
        mime="application/octet-stream",
    )
else:
    st.info(f"No .pkl files found for {stock} in model {model} under {PKL_BASE_DIR/model}.")
