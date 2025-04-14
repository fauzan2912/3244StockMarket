# src/data/loader.py

import os
import pickle
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
STOCKS_FILE = os.path.join(DATA_DIR, "stocks.pkl")
RAW_ETF_DIR = os.path.join(DATA_DIR, "raw", "ETFs")
RAW_STOCK_DIR = os.path.join(DATA_DIR, "raw", "Stocks")

def ensure_data_ready():
    """
    Ensures the processed stocks.pkl exists.
    If not, fetch and preprocess from Kaggle dataset.
    """
    if os.path.exists(STOCKS_FILE):
        return

    from kagglehub import dataset_download
    import concurrent.futures
    import shutil

    dataset_name = "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    path = dataset_download(dataset_name)

    print(f"[INFO] Downloaded dataset to: {path}")
    
    def clean_stock_name(filename):
        name = filename.replace(".txt", "").upper().replace(".", "-")
        if name.endswith("-US"):
            name = name[:-3]
        return name

    def process_file(file_path):
        try:
            df = pd.read_csv(file_path, low_memory=False)
            if df.empty or len(df.columns) < 7:
                return None
            stock = clean_stock_name(os.path.basename(file_path))
            df.insert(0, "Stock", stock)
            return df
        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            return None

    def load_and_merge(folder):
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".txt")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            dfs = list(executor.map(process_file, files))
        return pd.concat([df for df in dfs if df is not None], ignore_index=True)

    df_etfs = load_and_merge(os.path.join(path, "ETFs"))
    df_stocks = load_and_merge(os.path.join(path, "Stocks"))
    df_combined = pd.concat([df_etfs, df_stocks], ignore_index=True)

    # Drop bad rows, sort, and create target
    df_combined.dropna(inplace=True)
    df_combined['Date'] = pd.to_datetime(df_combined['Date'])
    df_combined.sort_values(by=['Stock', 'Date'], inplace=True)
    df_combined['Target'] = (df_combined['Close'].shift(-1) > df_combined['Close']).astype(int)
    df_combined = df_combined.groupby('Stock').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

    # Save as stocks.pkl
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(STOCKS_FILE, "wb") as f:
        pickle.dump(df_combined, f)

    # Clean up raw
    shutil.rmtree(path)
    print("[INFO] Processed stock data saved. Raw files cleaned.")

def get_stocks(symbols=None, start_date=None, end_date=None):
    """
    Returns filtered stock DataFrame. Auto-downloads data if not available.
    """
    ensure_data_ready()

    with open(STOCKS_FILE, "rb") as f:
        df = pickle.load(f)

    if isinstance(symbols, str):
        symbols = [symbols]

    if symbols:
        df = df[df["Stock"].isin(symbols)]
    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["Date"] <= pd.to_datetime(end_date)]

    return df.copy()
