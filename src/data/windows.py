# src/data/windows.py

import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.data.loader import get_stocks
from src.data.processor import get_technical_indicators

def prepare_time_window_data(stock_symbol, train_start, train_end, test_start, test_end):
    # Step 1: Load full date range that covers train + test
    df_all = get_stocks(stock_symbol, start_date=train_start, end_date=test_end)
    
    # Step 2: Add indicators using full historical context
    df_all = get_technical_indicators(df_all)
    df_all['Returns'] = df_all['Close'].pct_change().fillna(0)

    # Step 3: Slice AFTER indicator creation
    df_train = df_all[(df_all['Date'] >= train_start) & (df_all['Date'] <= train_end)].copy()
    df_test = df_all[(df_all['Date'] >= test_start) & (df_all['Date'] <= test_end)].copy()

    if df_train.empty or df_test.empty:
        print(f"[WARN] No data for {stock_symbol} in: {train_start.date()} to {test_end.date()}")
        return None, None

    return df_train, df_test

