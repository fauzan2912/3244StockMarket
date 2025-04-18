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
# 2. Base Directory
# -------------------------------
BASE_DIR = Path.cwd() / "results"

# -------------------------------
# 3. Helper Functions
# -------------------------------
@st.cache_data
def list_stocks():
    """List all stock directories under results/."""
    return sorted(
        [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
    )

@st.cache_data
def list_models(stock: str):
    """List model subdirectories under results/{stock}/."""
    stock_dir = BASE_DIR / stock
    if not stock_dir.exists():
        return []
    return sorted(
        [d.name for d in stock_dir.iterdir() if d.is_dir()]
    )

@st.cache_data
def load_metrics_df(stock: str, model: str):
    """Load metrics summary CSV from results/{stock}/{model}/."""
    dir_path = BASE_DIR / stock / model
    pattern = f"{stock}_{model}_metrics_summary.csv"
    files = list(dir_path.glob(pattern))
    if files:
        return pd.read_csv(files[0])
    return None

@st.cache_data
def get_chart_path(stock: str, model: str):
    """Get performance chart PNG path from results/{stock}/{model}/."""
    dir_path = BASE_DIR / stock / model
    pattern = f"{stock}_{model}_full_comparison.png"
    files = list(dir_path.glob(pattern))
    return files[0] if files else None

@st.cache_data
def list_pickles(stock: str, model: str):
    """List all .pkl files under results/{stock}/{model}/."""
    dir_path = BASE_DIR / stock / model
    if not dir_path.exists():
        return []
    return sorted(dir_path.glob("*.pkl"))

# -------------------------------
# 4. Sidebar: Filters
# -------------------------------
st.sidebar.header("Filters")

# Stock selection
stocks = list_stocks()
if not stocks:
    st.sidebar.error(
        "No stock folders found under results/. Run pipeline first."
    )
    st.stop()
stock = st.sidebar.selectbox("Select Stock", stocks)

# Model selection based on stock
models = list_models(stock)
if not models:
    st.sidebar.error(
        f"No model directories under results/{stock}/. Run pipeline first."
    )
    st.stop()
model = st.sidebar.selectbox("Select Model", models)

# -------------------------------
# 5. Main Page: Title
# -------------------------------
st.title(
    f"📈 {stock} — {model.upper()} Strategy Dashboard"
)

# -------------------------------
# 6. Display Metrics Table
# -------------------------------
st.subheader("Per-Window Metrics")
metrics_df = load_metrics_df(stock, model)
if metrics_df is not None:
    st.dataframe(metrics_df, use_container_width=True)
else:
    st.info(
        f"Metrics CSV not found at results/{stock}/{model}/."
    )

# -------------------------------
# 7. Display Cumulative Performance Chart
# -------------------------------
st.subheader("Cumulative Performance Chart")
chart_path = get_chart_path(stock, model)
if chart_path:
    st.image(
        str(chart_path),
        caption=(
            f"{stock} {model} (Rolling vs Expanding vs Buy & Hold)"
        ),
        use_column_width=True,
    )
else:
    st.info(
        f"Chart not found at results/{stock}/{model}/."
    )

# -------------------------------
# 8. Download Trained Model Pickle
# -------------------------------
st.subheader("Download Trained Model Pickle")
pkl_files = list_pickles(stock, model)
if pkl_files:
    selected = st.selectbox(
        "Select Pickle File to Download",
        [p.name for p in pkl_files]
    )
    pkl_path = BASE_DIR / stock / model / selected
    data = pkl_path.read_bytes()
    st.download_button(
        label="Download Pickle",
        data=data,
        file_name=selected,
        mime="application/octet-stream",
    )
else:
    st.info(
        f"No .pkl files found at results/{stock}/{model}/."
    )
