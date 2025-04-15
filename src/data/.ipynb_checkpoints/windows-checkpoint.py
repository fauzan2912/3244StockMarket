# src/data/windows.py

import os
import sys
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.data.loader import get_stocks
from src.data.processor import get_technical_indicators

def prepare_time_window_data(stock_symbol, train_start, train_end, test_start, test_end):
    df_train = get_stocks(stock_symbol, train_start, train_end)
    df_test = get_stocks(stock_symbol, test_start, test_end)

    if df_train.empty or df_test.empty:
        print(f"[WARN] No data for {stock_symbol} in: {train_start.date()} to {test_end.date()}")
        return None, None

    df_train = get_technical_indicators(df_train)
    df_test = get_technical_indicators(df_test)

    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)
    df_train['Returns'] = df_train['Close'].pct_change().fillna(0)
    df_test['Returns'] = df_test['Close'].pct_change().fillna(0)

    return df_train, df_test
