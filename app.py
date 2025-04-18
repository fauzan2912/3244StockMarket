import streamlit as st
import pandas as pd
from pathlib import Path
import io

# 1. Page configuration
st.set_page_config(
    page_title="Stock Forecasting Dashboard",
    layout="wide",
)

# 2. Results directories
RESULTS_DIR = Path("results")

# 3. Helper functions
@st.cache_data
def list_stocks():
    """Return a sorted list of stock codes (directories) under results/."""
    return sorted([p.name for p in RESULTS_DIR.iterdir() if p.is_dir()])

@st.cache_data
def list_models(stock: str):
    """Return a sorted list of model names for the selected stock."""
    folder = RESULTS_DIR / stock
    models = set()
    for csv in folder.glob(f"{stock}_*_metrics_summary.csv"):
        parts = csv.stem.split("_")
        if len(parts) >= 3:
            models.add(parts[1])
    return sorted(models)

@st.cache_data
def load_metrics(stock: str, model: str) -> pd.DataFrame:
    """Load the metrics summary CSV for the selected stock/model."""
    path = RESULTS_DIR / stock / f"{stock}_{model}_metrics_summary.csv"
    return pd.read_csv(path)

@st.cache_data
def list_model_pickles(stock: str, model: str):
    """List all .pkl files in results/{stock}/ matching stock_model_*.pkl."""
    folder = RESULTS_DIR / stock
    pattern = f"{stock}_{model}_*.pkl"
    return sorted(folder.glob(pattern))

# 4. Sidebar for stock & model selection
st.sidebar.header("Filters")
stock = st.sidebar.selectbox("Select stock", list_stocks())

models = list_models(stock)
if not models:
    st.error(f"No models found for '{stock}'. Please run your pipeline first.")
    st.stop()
model = st.sidebar.selectbox("Select model", models)

# 5. Page title
st.title(f"📈 {stock} — {model.upper()} Strategy Comparison")

# 6. Display per-window metrics table
st.subheader("Per-Window Metrics")
metrics_df = load_metrics(stock, model)
st.dataframe(metrics_df, use_container_width=True)

# 7. Display cumulative performance chart
st.subheader("Cumulative Performance")
chart_path = RESULTS_DIR / stock / f"{stock}_{model}_full_comparison.png"
if chart_path.exists():
    st.image(
        str(chart_path),
        caption=f"{stock} {model} (Rolling vs Expanding vs Buy & Hold)",
        use_column_width=True
    )
else:
    st.error(f"Performance chart not found at {chart_path}.")

# 8. Download trained model pickles
st.subheader("Download Trained Model Pickle")
pkl_files = list_model_pickles(stock, model)
if pkl_files:
    selected_pkl = st.selectbox("Select a .pkl file", [f.name for f in pkl_files])
    pkl_path = RESULTS_DIR / stock / selected_pkl
    with open(pkl_path, "rb") as file:
        bytes_data = file.read()
    st.download_button(
        label="Download model pickle",
        data=bytes_data,
        file_name=selected_pkl,
        mime="application/octet-stream"
    )
else:
    st.info(f"No .pkl files found in {RESULTS_DIR/stock} for {stock} + {model}.")