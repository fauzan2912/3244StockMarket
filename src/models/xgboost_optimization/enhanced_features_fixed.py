"""
Enhanced Feature Engineering for XGBoost Stock Prediction

This script implements additional advanced features for stock prediction:
1. Market regime features
2. Volatility-based features
3. Seasonality features
4. Price pattern recognition
5. Enhanced correlation features

These features can be integrated into the main XGBoost optimization script.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Function that was missing in the original implementation
def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Moving Average Convergence Divergence (MACD)
    df['MACD'] = df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    
    # Price Rate of Change
    df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
    df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
    df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
    
    # Average Directional Index (ADX)
    # True Range
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1))
        ),
        np.abs(df['Low'] - df['Close'].shift(1))
    )
    
    # Directional Movement
    df['DM_plus'] = np.where(
        (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
        np.maximum(df['High'] - df['High'].shift(1), 0),
        0
    )
    
    df['DM_minus'] = np.where(
        (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
        np.maximum(df['Low'].shift(1) - df['Low'], 0),
        0
    )
    
    # Smoothed True Range and Directional Movement
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    df['DI_plus_14'] = 100 * (df['DM_plus'].rolling(window=14).mean() / df['ATR_14'])
    df['DI_minus_14'] = 100 * (df['DM_minus'].rolling(window=14).mean() / df['ATR_14'])
    
    # ADX
    df['DX'] = 100 * np.abs(df['DI_plus_14'] - df['DI_minus_14']) / (df['DI_plus_14'] + df['DI_minus_14'])
    df['ADX'] = df['DX'].rolling(window=14).mean()
    
    # Volume Indicators
    df['Volume_ROC'] = df['Volume'].pct_change(periods=1) * 100
    df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
    df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_5']
    
    # On-Balance Volume (OBV)
    df['OBV'] = np.where(
        df['Close'] > df['Close'].shift(1),
        df['Volume'],
        np.where(
            df['Close'] < df['Close'].shift(1),
            -df['Volume'],
            0
        )
    ).cumsum()
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(periods=lag)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def add_market_regime_features(df, window=20):
    """
    Add market regime features using clustering on returns and volatility.
    
    Args:
        df: DataFrame with stock price data
        window: Window size for regime detection
        
    Returns:
        DataFrame with added market regime features
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate returns and volatility
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=window).std()
    
    # Prepare data for clustering
    data_for_clustering = df[['Returns', 'Volatility']].dropna().values
    
    # Apply KMeans clustering to identify market regimes
    n_clusters = 3  # Typically: bull, bear, and sideways markets
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # Skip if not enough data
    if len(data_for_clustering) < n_clusters:
        print("Not enough data for clustering. Skipping market regime features.")
        return df
    
    # Fit KMeans
    kmeans.fit(data_for_clustering)
    
    # Get cluster labels
    labels = kmeans.predict(df[['Returns', 'Volatility']].fillna(0))
    df['Market_Regime'] = labels
    
    # Create dummy variables for each regime
    for i in range(n_clusters):
        df[f'Regime_{i}'] = (df['Market_Regime'] == i).astype(int)
    
    # Calculate regime persistence
    df['Regime_Change'] = df['Market_Regime'].diff().ne(0).astype(int)
    df['Regime_Duration'] = df['Regime_Change'].cumsum()
    
    # Calculate regime-specific metrics
    for i in range(n_clusters):
        # Average return in this regime
        regime_returns = df.loc[df['Market_Regime'] == i, 'Returns']
        if not regime_returns.empty:
            avg_return = regime_returns.mean()
            df[f'Avg_Return_Regime_{i}'] = avg_return
        else:
            df[f'Avg_Return_Regime_{i}'] = 0
    
    return df

def add_volatility_features(df):
    """
    Add enhanced volatility-based features.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        DataFrame with added volatility features
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate returns if not already present
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change()
    
    # Historical Volatility (HV) for different windows
    for window in [5, 10, 20, 50]:
        df[f'HV_{window}'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)  # Annualized
    
    # Volatility of Volatility
    df['Vol_of_Vol_10'] = df['HV_10'].rolling(window=10).std()
    
    # GARCH-like volatility features (simplified)
    df['Abs_Returns'] = np.abs(df['Returns'])
    df['Sq_Returns'] = df['Returns'] ** 2
    
    # Exponentially weighted volatility
    df['EWMA_Vol_10'] = df['Returns'].ewm(span=10).std() * np.sqrt(252)
    df['EWMA_Vol_20'] = df['Returns'].ewm(span=20).std() * np.sqrt(252)
    
    # Volatility Ratio features
    df['Vol_Ratio_5_20'] = df['HV_5'] / df['HV_20']
    df['Vol_Ratio_10_50'] = df['HV_10'] / df['HV_50']
    
    # Volatility Trend
    df['Vol_Trend_10'] = df['HV_10'].pct_change(5)
    
    # Parkinson Volatility (uses High-Low range)
    df['Parkinson_Vol_10'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        (np.log(df['High'] / df['Low']) ** 2).rolling(window=10).mean() * 
        252
    )
    
    # Relative volatility compared to recent past
    df['Rel_Vol_Current_Past'] = df['HV_10'] / df['HV_50']
    
    return df

def add_seasonality_features(df):
    """
    Add calendar-based seasonality features.
    
    Args:
        df: DataFrame with stock price data (index must be DatetimeIndex)
        
    Returns:
        DataFrame with added seasonality features
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            print("Error: DataFrame must have DatetimeIndex or 'Date' column")
            return df
    
    # Extract calendar features
    df['Day_of_Week'] = df.index.dayofweek
    df['Day_of_Month'] = df.index.day
    df['Week_of_Year'] = df.index.isocalendar().week
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year
    
    # Create dummy variables for days of the week
    for i in range(5):  # 0=Monday, 4=Friday (only trading days)
        df[f'Is_Day_{i}'] = (df['Day_of_Week'] == i).astype(int)
    
    # Month-end and month-start effects
    df['Is_Month_End'] = df.index.is_month_end.astype(int)
    df['Is_Month_Start'] = df.index.is_month_start.astype(int)
    df['Is_Quarter_End'] = df.index.is_quarter_end.astype(int)
    df['Is_Quarter_Start'] = df.index.is_quarter_start.astype(int)
    df['Is_Year_End'] = df.index.is_year_end.astype(int)
    df['Is_Year_Start'] = df.index.is_year_start.astype(int)
    
    # Days to month/quarter end
    df['Days_to_Month_End'] = df.index.days_in_month - df.index.day
    
    # Seasonal decomposition for longer time series
    if len(df) >= 2 * 252:  # At least 2 years of data
        try:
            # Decompose the time series
            decomposition = sm.tsa.seasonal_decompose(
                df['Close'], 
                model='additive', 
                period=252  # Assuming 252 trading days per year
            )
            
            # Add components to dataframe
            df['Seasonal_Component'] = decomposition.seasonal
            df['Trend_Component'] = decomposition.trend
            df['Residual_Component'] = decomposition.resid
            
            # Fill NaN values
            df['Seasonal_Component'].fillna(method='bfill', inplace=True)
            df['Trend_Component'].fillna(method='bfill', inplace=True)
            df['Residual_Component'].fillna(method='bfill', inplace=True)
        except:
            print("Warning: Seasonal decomposition failed. Skipping these features.")
    
    return df

def add_price_pattern_features(df):
    """
    Add price pattern recognition features using pandas/numpy instead of TA-Lib.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        DataFrame with added price pattern features
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate basic momentum indicators
    
    # RSI (already implemented in add_technical_indicators)
    if 'RSI' not in df.columns:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    # Williams %R
    highest_high = df['High'].rolling(window=14).max()
    lowest_low = df['Low'].rolling(window=14).min()
    df['WILLR'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
    
    # Commodity Channel Index (CCI)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_ma = tp.rolling(window=20).mean()
    tp_md = (tp - tp_ma).abs().rolling(window=20).mean()
    df['CCI'] = (tp - tp_ma) / (0.015 * tp_md)
    
    # Average Directional Index (ADX)
    # Already implemented in add_technical_indicators
    
    # MACD
    # Already implemented in add_technical_indicators
    
    # Stochastic Oscillator
    df['STOCH_K'] = 100 * (df['Close'] - lowest_low) / (highest_high - lowest_low)
    df['STOCH_D'] = df['STOCH_K'].rolling(window=3).mean()
    
    # Bollinger Bands
    # Already implemented in add_technical_indicators
    
    # Simple pattern detection (without TA-Lib)
    
    # Doji pattern (simplified)
    df['DOJI'] = ((np.abs(df['Open'] - df['Close']) / (df['High'] - df['Low'])) < 0.1).astype(int)
    
    # Hammer pattern (simplified)
    df['HAMMER'] = (
        (df['Close'] > df['Open']) &  # Bullish
        ((df['High'] - df['Close']) < 0.3 * (df['High'] - df['Low'])) &  # Small upper shadow
        ((df['Open'] - df['Low']) > 2 * (df['Close'] - df['Open']))  # Long lower shadow
    ).astype(int)
    
    # Shooting Star pattern (simplified)
    df['SHOOTING_STAR'] = (
        (df['Open'] > df['Close']) &  # Bearish
        ((df['High'] - df['Open']) > 2 * (df['Open'] - df['Close'])) &  # Long upper shadow
        ((df['Close'] - df['Low']) < 0.3 * (df['High'] - df['Low']))  # Small lower shadow
    ).astype(int)
    
    # Engulfing pattern (simplified)
    df['ENGULFING_BULLISH'] = (
        (df['Close'].shift(1) < df['Open'].shift(1)) &  # Previous day was bearish
        (df['Close'] > df['Open']) &  # Current day is bullish
        (df['Open'] < df['Close'].shift(1)) &  # Open below previous close
        (df['Close'] > df['Open'].shift(1))  # Close above previous open
    ).astype(int)
    
    df['ENGULFING_BEARISH'] = (
        (df['Close'].shift(1) > df['Open'].shift(1)) &  # Previous day was bullish
        (df['Close'] < df['Open']) &  # Current day is bearish
        (df['Open'] > df['Close'].shift(1)) &  # Open above previous close
        (df['Close'] < df['Open'].shift(1))  # Close below previous open
    ).astype(int)
    
    # Add pattern count features
    pattern_columns = ['DOJI', 'HAMMER', 'SHOOTING_STAR', 'ENGULFING_BULLISH', 'ENGULFING_BEARISH']
    df['Bullish_Pattern_Count'] = df[['HAMMER', 'ENGULFING_BULLISH']].sum(axis=1)
    df['Bearish_Pattern_Count'] = df[['SHOOTING_STAR', 'ENGULFING_BEARISH']].sum(axis=1)
    df['Pattern_Signal'] = df['Bullish_Pattern_Count'] - df['Bearish_Pattern_Count']
    
    return df

def add_statistical_features(df):
    """
    Add statistical features based on price and returns.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        DataFrame with added statistical features
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate returns if not already present
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change()
    
    # Statistical moments for returns
    for window in [10, 20, 50]:
        # Skewness
        df[f'Returns_Skew_{window}d'] = df['Returns'].rolling(window).skew()
        
        # Kurtosis
        df[f'Returns_Kurt_{window}d'] = df['Returns'].rolling(window).kurt()
        
        # Z-score
        mean = df['Returns'].rolling(window).mean()
        std = df['Returns'].rolling(window).std()
        df[f'Returns_ZScore_{window}d'] = (df['Returns'] - mean) / std
    
    # Autocorrelation of returns
    for lag in [1, 2, 3, 5]:
        df[f'Returns_Autocorr_{lag}d'] = df['Returns'].rolling(window=20).apply(
            lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
        )
    
    # Runs test for randomness
    def runs_test(series):
        if len(series) < 10:
            return np.nan
        
        # Convert to binary sequence (1 for positive, 0 for negative)
        binary = (series > 0).astype(int)
        
        # Count runs
        runs = len([i for i in range(1, len(binary)) if binary[i] != binary[i-1]]) + 1
        
        # Count positive and negative values
        pos = sum(binary)
        neg = len(binary) - pos
        
        # Expected runs and standard deviation
        exp_runs = (2 * pos * neg) / (pos + neg) + 1
        std_runs = np.sqrt((2 * pos * neg * (2 * pos * neg - pos - neg)) / 
                           ((pos + neg)**2 * (pos + neg - 1)))
        
        # Z-statistic
        z = (runs - exp_runs) / std_runs
        
        return z
    
    # Apply runs test to returns
    df['Returns_RunsTest_20d'] = df['Returns'].rolling(window=20).apply(runs_test)
    
    # Hurst exponent (simplified)
    def hurst_exponent(series):
        if len(series) < 20:
            return np.nan
        
        # Calculate range of cumulative deviation
        cumdev = np.cumsum(series - np.mean(series))
        R = max(cumdev) - min(cumdev)
        
        # Calculate standard deviation
        S = np.std(series)
        
        # Calculate R/S ratio
        if S == 0:
            return np.nan
        
        RS = R / S
        
        # Simplified Hurst exponent
        H = np.log(RS) / np.log(len(series))
        
        return H
    
    # Apply Hurst exponent to returns
    df['Returns_Hurst_50d'] = df['Returns'].rolling(window=50).apply(hurst_exponent)
    
    # Jarque-Bera test for normality
    def jarque_bera_test(series):
        if len(series) < 10:
            return np.nan
        
        # Calculate skewness and kurtosis
        skew = stats.skew(series)
        kurt = stats.kurtosis(series)
        
        # Calculate JB statistic
        n = len(series)
        JB = n/6 * (skew**2 + (kurt**2)/4)
        
        return JB
    
    # Apply Jarque-Bera test to returns
    df['Returns_JB_20d'] = df['Returns'].rolling(window=20).apply(jarque_bera_test)
    
    return df

# Example usage
if __name__ == "__main__":
    # This is just an example of how to use these functions
    # In practice, you would integrate them into the main XGBoost optimization script
    
    # Create sample data
    dates = pd.date_range(start='2010-01-01', end='2017-12-31')
    np.random.seed(42)
    
    # Generate random price data
    n = len(dates)
    close_prices = np.random.normal(loc=100, scale=1, size=n).cumsum()
    daily_volatility = 2
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': close_prices + np.random.normal(0, daily_volatility, n),
        'High': close_prices + np.random.normal(daily_volatility, daily_volatility/2, n),
        'Low': close_prices - np.random.normal(daily_volatility, daily_volatility/2, n),
        'Close': close_prices,
        'Volume': np.random.normal(1000000, 200000, n).astype(int),
        'OpenInt': np.zeros(n)
    })
    
    df.set_index('Date', inplace=True)
    
    # Apply feature engineering
    df = add_technical_indicators(df)
    df = add_market_regime_features(df)
    df = add_volatility_features(df)
    df = add_seasonality_features(df)
    df = add_price_pattern_features(df)
    df = add_statistical_features(df)
    
    # Print feature names
    print(f"Total features: {len(df.columns)}")
    print("Feature names:")
    for col in df.columns:
        print(f"- {col}")
