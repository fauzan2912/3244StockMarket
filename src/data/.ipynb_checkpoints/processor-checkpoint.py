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
    df['BOLL_Width'] = df['BOLL_Upper'] - df['BOLL_Lower']

    # === Momentum features ===
    df['Momentum_1D'] = df['Close'].diff(1)
    df['Momentum_3D'] = df['Close'].diff(3)
    df['Momentum_7D'] = df['Close'].diff(7)

    # === Price ratios ===
    df['Close/Open'] = df['Close'] / df['Open']
    df['High/Low'] = df['High'] / df['Low']

    # === Lag Features ===
    for lag in [1, 2, 3, 5, 10]:
        df[f'Close_Lag{lag}'] = df['Close'].shift(lag)

    # === Rolling Averages ===
    df['RollingMean_5'] = df['Close'].rolling(window=5).mean()
    df['RollingStd_5'] = df['Close'].rolling(window=5).std()
    df['RollingMax_10'] = df['Close'].rolling(10).max()
    df['RollingMin_10'] = df['Close'].rolling(10).min()

    df['FutureReturn_5D'] = df['Close'].pct_change(periods=5).shift(-5)

    # === Volatility  ===
    df['Volatility_5'] = df['Close'].rolling(window=5).std()
    df['Volatility_10'] = df['Close'].rolling(window=10).std()

    df['High_Low'] = df['High'] - df['Low']
    df['High_Close'] = np.abs(df['High'] - df['Close'].shift(1))
    df['Low_Close'] = np.abs(df['Low'] - df['Close'].shift(1))
    df['TrueRange'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)
    df['ATR_14'] = df['TrueRange'].rolling(window=14).mean()


    # === Standardize selected features ===
    scale_cols = ['Momentum_1D', 'Momentum_3D', 'Momentum_7D', 'Close_Lag1']
    df[scale_cols] = df[scale_cols].fillna(0)
    df[scale_cols] = StandardScaler().fit_transform(df[scale_cols])

    return df
