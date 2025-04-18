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
    """List all stock codes based on directories under BASE_DIR (excluding stocks_code)."""
    return sorted([
        d.name for d in BASE_DIR.iterdir()
        if d.is_dir() and d.name != PKL_BASE_DIR.name
    ])

@st.cache_data
def list_models():
    """List all model directories under results/stocks_code/."""
    return sorted([
        d.name for d in PKL_BASE_DIR.iterdir()
        if d.is_dir()
    ]) if PKL_BASE_DIR.exists() else []

@st.cache_data
def load_metrics_df(stock: str, model: str):
    """Load the metrics summary CSV from results/stocks_code/{model}/ matching '{stock}_*metrics_summary.csv'."""
    model_dir = PKL_BASE_DIR / model
    pattern = f"{stock}_*metrics_summary.csv"
    files = list(model_dir.glob(pattern))
    if files:
        return pd.read_csv(files[0])
    return None

@st.cache_data
def get_chart_path(stock: str, model: str):
    """Get the performance chart PNG path from results/stocks_code/{model}/ matching '{stock}_*full_comparison.png'."""
    model_dir = PKL_BASE_DIR / model
    pattern = f"{stock}_*full_comparison.png"
    files = list(model_dir.glob(pattern))
    return files[0] if files else None

@st.cache_data
def list_pickles_for_model(model: str):
    """List all .pkl files under results/stocks_code/{model}/."""
    model_dir = PKL_BASE_DIR / model
    return sorted(model_dir.glob("*.pkl")) if model_dir.exists() else []

# -------------------------------
# 4. Sidebar: Filters
# -------------------------------
st.sidebar.header("Filters")

# 4a. Select Stock
stocks = list_stocks()
if not stocks:
    st.sidebar.error("No stock directories found under results/. Run pipeline to generate metrics.")
    st.stop()
stock = st.sidebar.selectbox("Select Stock", stocks)

# 4b. Select Model
models = list_models()
if not models:
    st.sidebar.error("No model directories under results/stocks_code/. Run pipeline first.")
    st.stop()
model = st.sidebar.selectbox("Select Model", models)

# -------------------------------
# 5. Display Metrics Table
# -------------------------------
st.title(f"📈 {stock} — {model.upper()} Strategy Dashboard")
st.subheader("Per-Window Metrics")
metrics_df = load_metrics_df(stock, model)
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info(f"No metrics CSV found for stock '{stock}' in model '{model}'.")

# -------------------------------
# 6. Display Cumulative Performance Chart
# -------------------------------
st.subheader("Cumulative Performance Chart")
chart_path = get_chart_path(stock, model)
if chart_path:
    st.image(
        str(chart_path),
        caption=f"{stock} {model} (Rolling vs Expanding vs Buy & Hold)",
        use_column_width=True,
    )
else:
    st.info(f"No chart PNG found for stock '{stock}' in model '{model}'.")

# -------------------------------
# 7. Download Model Pickle
# -------------------------------
st.subheader("Download Trained Model Pickle")
pkl_files = list_pickles_for_model(model)
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
    st.info(f"No .pkl files found under results/stocks_code/{model}/.")