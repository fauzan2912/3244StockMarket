# src/data/processor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_technical_indicators(df):
    """
    Compute technical indicators and engineered features.

    Args:
        df: DataFrame with at least ['Open', 'High', 'Low', 'Close', 'Volume']

    Returns:
        DataFrame with additional features
    """
    df = df.copy()

    # === MACD ===
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()

    # === RSI ===
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # === Bollinger Bands ===
    df['BOLL_Mid'] = df['Close'].rolling(window=20).mean()
    df['BOLL_Std'] = df['Close'].rolling(window=20).std()
    df['BOLL_Upper'] = df['BOLL_Mid'] + 2 * df['BOLL_Std']
    df['BOLL_Lower'] = df['BOLL_Mid'] - 2 * df['BOLL_Std']

    # === Momentum features ===
    df['Momentum_1D'] = df['Close'].diff(1)
    df['Momentum_3D'] = df['Close'].diff(3)
    df['Momentum_7D'] = df['Close'].diff(7)

    # === Price ratios ===
    df['Close/Open'] = df['Close'] / df['Open']
    df['High/Low'] = df['High'] / df['Low']

    # === Lag Features ===
    df['Close_Lag1'] = df['Close'].shift(1)

    # === Standardize selected features ===
    scale_cols = ['Momentum_1D', 'Momentum_3D', 'Momentum_7D', 'Close_Lag1']
    df[scale_cols] = df[scale_cols].fillna(0)
    df[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

    return df
