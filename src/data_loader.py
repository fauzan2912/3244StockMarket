#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import pickle
import kagglehub
import shutil
import concurrent.futures  # For parallel processing
from datetime import datetime

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Go up one level
DATA_DIR = os.path.join(BASE_DIR, "data")
ETFS_FILE = os.path.join(DATA_DIR, "etfs.pkl")
STOCKS_FILE = os.path.join(DATA_DIR, "stocks.pkl")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset():
    """Downloads dataset only if it doesn't already exist."""
    if os.path.exists(ETFS_FILE) and os.path.exists(STOCKS_FILE):
        print("[OK] Processed data already exists. Skipping download.")
        return None  # No need to download

    dataset_name = "borismarjanovic/price-volume-data-for-all-us-stocks-etfs"
    print(f"[DOWNLOAD] Downloading dataset: {dataset_name}")

    # Download dataset
    path = kagglehub.dataset_download(dataset_name)

    print(f"[OK] Dataset downloaded to: {path}")
    return path

def clean_stock_name(filename):
    """Cleans stock/ETF names by removing '-US' suffix."""
    name = filename.replace(".txt", "").upper().replace(".", "-")
    if name.endswith("-US"):
        name = name[:-3]  # Remove '-US' from the end
    return name

def process_file(file_path):
    """Loads a single stock/ETF file into a DataFrame."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
        if df.empty or len(df.columns) < 7:
            return None
        
        stock_name = clean_stock_name(os.path.basename(file_path))
        df.insert(0, "Stock", stock_name)
        return df
    except Exception as e:
        print(f"[ERROR] Error processing {file_path}: {e}")
        return None

def load_and_preprocess_data(folder_path, dataset_type, selected_symbols=None):
    """Loads and preprocesses stock or ETF data in parallel."""
    if not os.path.exists(folder_path):
        print(f"[ERROR] Directory not found: {folder_path}")
        return pd.DataFrame()

    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".txt")]

    # Use multithreading to process files faster
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        data_frames = list(executor.map(process_file, files))

    # Remove None values from failed file loads
    data_frames = [df for df in data_frames if df is not None]

    if not data_frames:
        return pd.DataFrame()

    df_combined = pd.concat(data_frames, ignore_index=True)

    # Apply filtering if selected stocks/ETFs are specified
    if selected_symbols:
        df_combined = df_combined[df_combined["Stock"].isin(selected_symbols)]

    # Preprocessing
    df_combined.dropna(inplace=True)
    df_combined['Date'] = pd.to_datetime(df_combined['Date'])
    df_combined.sort_values(by=['Stock', 'Date'], inplace=True)

    # Create target variable for classification (1 if price goes up, 0 if down)
    df_combined['Target'] = (df_combined['Close'].shift(-1) > df_combined['Close']).astype(int)

    # Drop last row per stock since there's no next-day data
    df_combined = df_combined.groupby('Stock').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

    # Select features for modeling
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    df_combined = df_combined[['Stock', 'Date'] + features + ['Target']]

    print(f"[OK] Processed {dataset_type} dataset with {len(df_combined)} records.")
    return df_combined

def process_data():
    """Downloads and processes data if not already done."""
    if os.path.exists(ETFS_FILE) and os.path.exists(STOCKS_FILE):
        print("[OK] Processed data already exists. Skipping processing.")
        return

    print("\n[START] Downloading and Processing Data...\n")
    raw_data_path = download_dataset()
    
    if not raw_data_path:
        return  # Data already processed

    # Load and preprocess ETFs
    etfs_path = os.path.join(raw_data_path, "ETFs")
    etfs_df = load_and_preprocess_data(etfs_path, "ETFs")
    with open(ETFS_FILE, "wb") as f:
        pickle.dump(etfs_df, f)

    # Load and preprocess Stocks
    stocks_path = os.path.join(raw_data_path, "Stocks")
    stocks_df = load_and_preprocess_data(stocks_path, "Stocks")
    with open(STOCKS_FILE, "wb") as f:
        pickle.dump(stocks_df, f)

    # Delete raw dataset asynchronously
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(shutil.rmtree, raw_data_path)

    print("[CLEAN] Raw dataset removed to save space.")

    print("\n[OK] Data Processing Completed!\n")

def get_stocks(stocks=None, start_date=None, end_date=None):
    """Returns a DataFrame of selected stocks (default: all) with optional date filtering.
    
    - `stocks="NVDA"` (string) → Returns only NVDA
    - `stocks=["NVDA", "AAPL"]` (list) → Returns NVDA & AAPL
    - `stocks=None` → Returns all stocks
    - `start_date="2016-01-01"` (string) → Filter from this date
    - `end_date="2016-12-31"` (string) → Filter until this date
    """
    process_data()
    with open(STOCKS_FILE, "rb") as f:
        df = pickle.load(f)

    # Convert single string input to a list
    if isinstance(stocks, str):
        stocks = [stocks]

    # Apply filtering if stocks are provided
    if stocks:
        df = df[df["Stock"].isin(stocks)]
    
    # Apply date filtering if provided
    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        df = df[df["Date"] >= start_date]
    
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        df = df[df["Date"] <= end_date]
    
    # Print date range of data
    if not df.empty:
        min_date = df["Date"].min().strftime("%Y-%m-%d")
        max_date = df["Date"].max().strftime("%Y-%m-%d")
        print(f"[INFO] Date range: {min_date} to {max_date}")
    
    return df

def get_etfs(etfs=None, start_date=None, end_date=None):
    """Returns a DataFrame of selected ETFs (default: all) with optional date filtering.
    
    - `etfs="SPY"` (string) → Returns only SPY
    - `etfs=["SPY", "QQQ"]` (list) → Returns SPY & QQQ
    - `etfs=None` → Returns all ETFs
    - `start_date="2016-01-01"` (string) → Filter from this date
    - `end_date="2016-12-31"` (string) → Filter until this date
    """
    process_data()
    with open(ETFS_FILE, "rb") as f:
        df = pickle.load(f)

    # Convert single string input to a list
    if isinstance(etfs, str):
        etfs = [etfs]

    # Apply filtering if ETFs are provided
    if etfs:
        df = df[df["Stock"].isin(etfs)]
    
    # Apply date filtering if provided
    if start_date:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        df = df[df["Date"] >= start_date]
    
    if end_date:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        df = df[df["Date"] <= end_date]
    
    # Print date range of data
    if not df.empty:
        min_date = df["Date"].min().strftime("%Y-%m-%d")
        max_date = df["Date"].max().strftime("%Y-%m-%d")
        print(f"[INFO] Date range: {min_date} to {max_date}")
    
    return df

def get_technical_indicators(df):
    """Calculates and adds technical indicators and price-based features to the DataFrame."""

    import numpy as np
    from sklearn.preprocessing import StandardScaler

    required_columns = {'Close', 'Open', 'High', 'Low', 'Volume', 'Date'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_columns - set(df.columns)}")

    df = df.copy()

    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['Hist'] = df['MACD'] - df['Signal']
    df['MACD_Buy'] = df['MACD'] > df['Signal']
    df['MACD_Sell'] = df['MACD'] < df['Signal']

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_Buy'] = df['RSI'] < 30
    df['RSI_Sell'] = df['RSI'] > 70

    # KDJ
    df['Low_Min'] = df['Low'].rolling(window=9).min()
    df['High_Max'] = df['High'].rolling(window=9).max()
    df['RSV'] = (df['Close'] - df['Low_Min']) / (df['High_Max'] - df['Low_Min']) * 100
    df['K'] = df['RSV'].ewm(alpha=1/3).mean()
    df['D'] = df['K'].ewm(alpha=1/3).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']

    # OSC
    df['OSC'] = (df['Close'].rolling(window=5).mean() - df['Close'].rolling(window=10).mean()) / df['Close'].rolling(window=10).mean() * 100
    df['OSC_Buy'] = df['OSC'] > 0
    df['OSC_Sell'] = df['OSC'] < 0

    # Bollinger Bands
    df['BOLL_Mid'] = df['Close'].rolling(window=20).mean()
    df['BOLL_STD'] = df['Close'].rolling(window=20).std()
    df['BOLL_Upper'] = df['BOLL_Mid'] + 2 * df['BOLL_STD']
    df['BOLL_Lower'] = df['BOLL_Mid'] - 2 * df['BOLL_STD']
    df['BOLL_Buy'] = df['Close'] <= df['BOLL_Lower']
    df['BOLL_Sell'] = df['Close'] >= df['BOLL_Upper']

    # BIAS
    df['BIAS'] = ((df['Close'] - df['Close'].rolling(window=6).mean()) / df['Close'].rolling(window=6).mean()) * 100
    df['BIAS_Buy'] = df['BIAS'] < 0
    df['BIAS_Sell'] = df['BIAS'] > 0

    # Stochastics
    df['%K'] = df['RSV']
    df['%D'] = df['%K'].rolling(window=3).mean()
    df['STOCHS_Buy'] = (df['%K'] > df['%D']) & (df['%K'] < 20)
    df['STOCHS_Sell'] = (df['%K'] < df['%D']) & (df['%K'] > 80)

    # ADX (simplified proxy)
    df['+DI'] = 100 * (df['High'] - df['Low']).rolling(window=14).mean()
    df['-DI'] = 100 * (df['Low'] - df['High']).rolling(window=14).mean()
    df['ADX'] = (df['+DI'] - df['-DI']).abs().rolling(window=14).mean()
    df['ADX_Buy'] = df['ADX'] > 25
    df['ADX_Sell'] = df['ADX'] < 25

    # Aroon
    df['Aroon_Up'] = 100 * df['High'].rolling(window=25).apply(lambda x: x.argmax() / 25, raw=True)
    df['Aroon_Down'] = 100 * df['Low'].rolling(window=25).apply(lambda x: x.argmin() / 25, raw=True)
    df['Aroon_Buy'] = df['Aroon_Up'] > df['Aroon_Down']
    df['Aroon_Sell'] = df['Aroon_Up'] < df['Aroon_Down']

    # Price Differencing
    df['Close_Diff_1'] = df['Close'].diff(1)
    df['Close_Diff_1'] = StandardScaler().fit_transform(df[['Close_Diff_1']].fillna(0))

    # Lagged Prices
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_1'] = df['Close_Lag_1'].fillna(method='bfill')  # Handle NaN in the first row

    print("[OK] Full technical indicators and price features added.")
    return df