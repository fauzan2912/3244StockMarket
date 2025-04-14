"""
XGBoost Stock Price Prediction Optimization

This script implements an optimized XGBoost model for stock price prediction with:
1. Comparison of rolling vs expanding window approaches
2. Multiclass prediction (price movement by percentage ranges)
3. Stock correlation features
4. Hyperparameter optimization
5. Enhanced feature engineering

Based on the Huge Stock Market Dataset from Kaggle.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import shap
import warnings
warnings.filterwarnings('ignore')

# Constants
STOCKS = ['AAPL', 'MSFT', 'GOOGL']
HORIZONS = [1, 3, 7, 30, 252]
RESULTS_DIR = "./results/"
DATA_DIR = "./data/"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Helper Functions for Data Processing
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
    
    # Percentage Price Oscillator (PPO)
    df['PPO'] = ((df['EMA_10'] - df['EMA_20']) / df['EMA_20']) * 100
    
    # Stochastic Oscillator
    df['Lowest_14'] = df['Low'].rolling(window=14).min()
    df['Highest_14'] = df['High'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - df['Lowest_14']) / (df['Highest_14'] - df['Lowest_14']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Commodity Channel Index (CCI)
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_SMA_20'] = df['TP'].rolling(window=20).mean()
    df['TP_Deviation'] = df['TP'] - df['TP_SMA_20']
    df['TP_Deviation_SMA_20'] = 0.015 * df['TP_Deviation'].abs().rolling(window=20).mean()
    df['CCI'] = df['TP_Deviation'] / df['TP_Deviation_SMA_20']
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(periods=lag)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

def add_correlation_features(stock_dfs, stock_names):
    """
    Add correlation-based features between stocks.
    
    Args:
        stock_dfs: Dictionary of DataFrames with stock data
        stock_names: List of stock names
        
    Returns:
        Dictionary of DataFrames with added correlation features
    """
    # Create a copy to avoid modifying the original dataframes
    result_dfs = {stock: df.copy() for stock, df in stock_dfs.items()}
    
    # Ensure all dataframes have the same index
    common_dates = set.intersection(*[set(df.index) for df in stock_dfs.values()])
    for stock in stock_names:
        result_dfs[stock] = result_dfs[stock].loc[result_dfs[stock].index.isin(common_dates)]
    
    # Calculate correlation features
    for stock in stock_names:
        # Add correlation features with other stocks
        for other_stock in stock_names:
            if other_stock != stock:
                # Price correlation
                result_dfs[stock][f'Corr_{other_stock}_Price_5d'] = result_dfs[stock]['Close'].rolling(5).corr(
                    stock_dfs[other_stock]['Close'].reindex(result_dfs[stock].index))
                result_dfs[stock][f'Corr_{other_stock}_Price_10d'] = result_dfs[stock]['Close'].rolling(10).corr(
                    stock_dfs[other_stock]['Close'].reindex(result_dfs[stock].index))
                result_dfs[stock][f'Corr_{other_stock}_Price_20d'] = result_dfs[stock]['Close'].rolling(20).corr(
                    stock_dfs[other_stock]['Close'].reindex(result_dfs[stock].index))
                
                # Return correlation
                result_dfs[stock][f'Corr_{other_stock}_Return_5d'] = result_dfs[stock]['Return_Lag_1'].rolling(5).corr(
                    stock_dfs[other_stock]['Return_Lag_1'].reindex(result_dfs[stock].index))
                result_dfs[stock][f'Corr_{other_stock}_Return_10d'] = result_dfs[stock]['Return_Lag_1'].rolling(10).corr(
                    stock_dfs[other_stock]['Return_Lag_1'].reindex(result_dfs[stock].index))
                
                # Volume correlation
                result_dfs[stock][f'Corr_{other_stock}_Volume_5d'] = result_dfs[stock]['Volume'].rolling(5).corr(
                    stock_dfs[other_stock]['Volume'].reindex(result_dfs[stock].index))
                
                # Relative performance
                result_dfs[stock][f'RelPerf_{other_stock}_1d'] = (
                    result_dfs[stock]['Close'] / result_dfs[stock]['Close'].shift(1)
                ) / (
                    stock_dfs[other_stock]['Close'].reindex(result_dfs[stock].index) / 
                    stock_dfs[other_stock]['Close'].reindex(result_dfs[stock].index).shift(1)
                )
                
                result_dfs[stock][f'RelPerf_{other_stock}_5d'] = (
                    result_dfs[stock]['Close'] / result_dfs[stock]['Close'].shift(5)
                ) / (
                    stock_dfs[other_stock]['Close'].reindex(result_dfs[stock].index) / 
                    stock_dfs[other_stock]['Close'].reindex(result_dfs[stock].index).shift(5)
                )
    
    # Drop rows with NaN values
    for stock in stock_names:
        result_dfs[stock] = result_dfs[stock].dropna()
    
    return result_dfs

def create_target_variable(df, horizon, multiclass=False):
    """
    Create target variable for prediction.
    
    Args:
        df: DataFrame with stock price data
        horizon: Prediction horizon in days
        multiclass: Whether to create multiclass target (True) or binary target (False)
        
    Returns:
        DataFrame with added target variable
    """
    # Calculate future return
    df[f'Future_Return_{horizon}d'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
    
    if multiclass:
        # Create multiclass target based on percentage ranges
        conditions = [
            (df[f'Future_Return_{horizon}d'] < -0.05),                  # Down >5%
            (df[f'Future_Return_{horizon}d'] >= -0.05) & (df[f'Future_Return_{horizon}d'] < -0.03),  # Down 3-5%
            (df[f'Future_Return_{horizon}d'] >= -0.03) & (df[f'Future_Return_{horizon}d'] < -0.01),  # Down 1-3%
            (df[f'Future_Return_{horizon}d'] >= -0.01) & (df[f'Future_Return_{horizon}d'] < 0),      # Down <1%
            (df[f'Future_Return_{horizon}d'] >= 0) & (df[f'Future_Return_{horizon}d'] < 0.01),       # Up <1%
            (df[f'Future_Return_{horizon}d'] >= 0.01) & (df[f'Future_Return_{horizon}d'] < 0.03),    # Up 1-3%
            (df[f'Future_Return_{horizon}d'] >= 0.03) & (df[f'Future_Return_{horizon}d'] < 0.05),    # Up 3-5%
            (df[f'Future_Return_{horizon}d'] >= 0.05)                   # Up >5%
        ]
        
        values = [0, 1, 2, 3, 4, 5, 6, 7]  # Class labels
        df[f'Target_{horizon}d'] = np.select(conditions, values)
    else:
        # Create binary target (1 if price goes up, 0 if price goes down)
        df[f'Target_{horizon}d'] = (df[f'Future_Return_{horizon}d'] > 0).astype(int)
    
    # Drop rows with NaN in target
    df = df.dropna(subset=[f'Target_{horizon}d'])
    
    return df

def process_stock_data(stock, horizon, window_size=90, step_size=30, multiclass=False):
    """
    Process stock data for a given stock and horizon.
    
    Args:
        stock: Stock symbol
        horizon: Prediction horizon in days
        window_size: Size of the rolling window
        step_size: Step size for the rolling window
        multiclass: Whether to create multiclass target
        
    Returns:
        Processed DataFrame
    """
    print(f"Processing {stock} for horizon {horizon} with window_size {window_size} and step_size {step_size}...")
    
    # Check if processed data already exists
    processed_file = f"{DATA_DIR}{stock}_processed_h{horizon}_mc{multiclass}.csv"
    if os.path.exists(processed_file):
        print(f"✅ Processed data already exists. Loading from {processed_file}")
        return pd.read_csv(processed_file, index_col='Date', parse_dates=True)
    
    # Load data
    try:
        # In a real scenario, you would load from the Kaggle dataset
        # For this example, we'll create some dummy data
        dates = pd.date_range(start='2010-01-01', end='2017-12-31')
        np.random.seed(42)  # For reproducibility
        
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
        
        print(f"✅ Data loaded for {stock}")
    except Exception as e:
        print(f"❌ Error loading data for {stock}: {e}")
        return None
    
    # Add technical indicators
    df = add_technical_indicators(df)
    print(f"✅ Technical indicators added successfully.")
    
    # Create target variable
    df = create_target_variable(df, horizon, multiclass)
    
    # Print target distribution
    target_col = f'Target_{horizon}d'
    if multiclass:
        print(f"Target distribution for {stock} at horizon {horizon}:")
        print(df[target_col].value_counts(normalize=True))
    else:
        threshold = 0.005  # 0.5% threshold for binary classification
        print(f"Adjusted threshold for {stock} at horizon {horizon}: {threshold:.4f}")
        print(f"Unique target values for {stock} at horizon {horizon}: {np.unique(df[target_col])}")
    
    # Save processed data
    df.to_csv(processed_file)
    
    return df

def train_model_rolling_window(X, y, window_size, step_size, params=None):
    """
    Train XGBoost model using rolling window cross-validation.
    
    Args:
        X: Feature matrix
        y: Target vector
        window_size: Size of the rolling window
        step_size: Step size for the rolling window
        params: XGBoost parameters
        
    Returns:
        Dictionary with predictions, accuracies, and models
    """
    if params is None:
        params = {
            'objective': 'binary:logistic' if len(np.unique(y)) <= 2 else 'multi:softprob',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'alpha': 0,
            'lambda': 1,
            'num_class': len(np.unique(y)) if len(np.unique(y)) > 2 else None
        }
        
        # Remove num_class for binary classification
        if len(np.unique(y)) <= 2:
            params.pop('num_class', None)
    
    n_samples = len(X)
    predictions = np.zeros(n_samples) if len(np.unique(y)) <= 2 else np.zeros((n_samples, len(np.unique(y))))
    accuracies = []
    models = []
    
    # Determine the number of windows
    n_windows = (n_samples - window_size) // step_size + 1
    
    for i in range(n_windows):
        # Define window indices
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Skip if we've reached the end of the data
        if end_idx >= n_samples:
            break
        
        # Split data into train and test
        X_train = X.iloc[start_idx:end_idx]
        y_train = y.iloc[start_idx:end_idx]
        
        # Define test indices
        test_start_idx = end_idx
        test_end_idx = min(test_start_idx + step_size, n_samples)
        
        # Skip if test set is empty
        if test_start_idx >= n_samples or test_end_idx <= test_start_idx:
            break
        
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]
        
        # Skip if training data has only one class
        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {start_idx} due to single class in y_train: {np.unique(y_train)}")
            continue
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], 
                          early_stopping_rounds=20, verbose_eval=False)
        
        # Make predictions
        if len(np.unique(y)) <= 2:
            # Binary classification
            pred_probs = model.predict(dtest)
            preds = (pred_probs > 0.5).astype(int)
            predictions[test_start_idx:test_end_idx] = preds
        else:
            # Multiclass classification
            pred_probs = model.predict(dtest)
            preds = np.argmax(pred_probs, axis=1)
            predictions[test_start_idx:test_end_idx] = preds
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, preds)
        accuracies.append(accuracy)
        models.append(model)
        
        print(f"Window {i+1}/{n_windows}: Accuracy = {accuracy:.4f}")
    
    return {
        'predictions': predictions,
        'accuracies': accuracies,
        'models': models
    }

def train_model_expanding_window(X, y, initial_window_size, step_size, params=None):
    """
    Train XGBoost model using expanding window approach.
    
    Args:
        X: Feature matrix
        y: Target vector
        initial_window_size: Initial size of the window
        step_size: Step size for expanding the window
        params: XGBoost parameters
        
    Returns:
        Dictionary with predictions, accuracies, and models
    """
    if params is None:
        params = {
            'objective': 'binary:logistic' if len(np.unique(y)) <= 2 else 'multi:softprob',
            'eval_metric': 'logloss',
            'eta': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'alpha': 0,
            'lambda': 1,
            'num_class': len(np.unique(y)) if len(np.unique(y)) > 2 else None
        }
        
        # Remove num_class for binary classification
        if len(np.unique(y)) <= 2:
            params.pop('num_class', None)
    
    n_samples = len(X)
    predictions = np.zeros(n_samples) if len(np.unique(y)) <= 2 else np.zeros((n_samples, len(np.unique(y))))
    accuracies = []
    models = []
    
    # Determine the number of windows
    n_windows = (n_samples - initial_window_size) // step_size + 1
    
    for i in range(n_windows):
        # Define window indices
        end_idx = initial_window_size + i * step_size
        
        # Skip if we've reached the end of the data
        if end_idx >= n_samples:
            break
        
        # Split data into train and test
        X_train = X.iloc[:end_idx]
        y_train = y.iloc[:end_idx]
        
        # Define test indices
        test_start_idx = end_idx
        test_end_idx = min(test_start_idx + step_size, n_samples)
        
        # Skip if test set is empty
        if test_start_idx >= n_samples or test_end_idx <= test_start_idx:
            break
        
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]
        
        # Skip if training data has only one class
        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {end_idx} due to single class in y_train: {np.unique(y_train)}")
            continue
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], 
                          early_stopping_rounds=20, verbose_eval=False)
        
        # Make predictions
        if len(np.unique(y)) <= 2:
            # Binary classification
            pred_probs = model.predict(dtest)
            preds = (pred_probs > 0.5).astype(int)
            predictions[test_start_idx:test_end_idx] = preds
        else:
            # Multiclass classification
            pred_probs = model.predict(dtest)
            preds = np.argmax(pred_probs, axis=1)
            predictions[test_start_idx:test_end_idx] = preds
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, preds)
        accuracies.append(accuracy)
        models.append(model)
        
        print(f"Expanding Window {i+1}/{n_windows}: Train size = {len(X_train)}, Accuracy = {accuracy:.4f}")
    
    return {
        'predictions': predictions,
        'accuracies': accuracies,
        'models': models
    }

def optimize_hyperparameters(X_train, y_train, X_val, y_val, is_multiclass=False):
    """
    Optimize XGBoost hyperparameters using grid search.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        is_multiclass: Whether it's a multiclass problem
        
    Returns:
        Dictionary with best parameters
    """
    print("Optimizing XGBoost hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0.1, 1, 10]
    }
    
    # Create XGBoost classifier
    if is_multiclass:
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            num_class=len(np.unique(y_train))
        )
    else:
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    # Use a smaller subset of the parameter grid for faster optimization
    # In a real scenario, you would use the full grid
    small_param_grid = {
        'max_depth': [3, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [50, 100],
        'subsample': [0.8, 1.0]
    }
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=small_param_grid,
        scoring='accuracy',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy with best parameters: {accuracy:.4f}")
    
    return best_params

def evaluate_model(y_true, y_pred, is_multiclass=False):
    """
    Evaluate model performance.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        is_multiclass: Whether it's a multiclass problem
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate baseline accuracy (majority class)
    baseline_accuracy = max(np.bincount(y_true)) / len(y_true)
    
    if is_multiclass:
        # For multiclass, calculate macro F1-score
        f1 = f1_score(y_true, y_pred, average='macro')
        
        # No direct ROC-AUC for multiclass, so we'll skip it
        roc_auc = None
        
        # Generate classification report
        report = classification_report(y_true, y_pred)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
    else:
        # For binary, calculate F1-score and ROC-AUC
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred)
        
        # Generate classification report
        report = classification_report(y_true, y_pred)
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'baseline_accuracy': baseline_accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

def compute_shap_values(model, X, is_multiclass=False):
    """
    Compute SHAP values for feature importance.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix
        is_multiclass: Whether it's a multiclass problem
        
    Returns:
        SHAP values
    """
    # Convert to DMatrix for XGBoost
    dmatrix = xgb.DMatrix(X)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X)
    
    return shap_values

def plot_shap_summary(shap_values, X, is_multiclass=False, class_idx=None, output_file=None):
    """
    Plot SHAP summary.
    
    Args:
        shap_values: SHAP values
        X: Feature matrix
        is_multiclass: Whether it's a multiclass problem
        class_idx: Class index for multiclass problems
        output_file: Output file path
    """
    plt.figure(figsize=(12, 8))
    
    if is_multiclass:
        if class_idx is not None:
            # Plot for specific class
            shap.summary_plot(shap_values[class_idx], X, show=False)
            plt.title(f'SHAP Summary Plot for Class {class_idx}')
        else:
            # Plot for all classes
            shap.summary_plot(shap_values, X, show=False)
            plt.title('SHAP Summary Plot for All Classes')
    else:
        # Binary classification
        shap.summary_plot(shap_values, X, show=False)
        plt.title('SHAP Summary Plot')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Saved SHAP plot to {output_file}")
    
    plt.close()

def process_horizon(horizon, window_size=90, step_size=30, multiclass=False, use_expanding_window=False):
    """
    Process all stocks for a given horizon.
    
    Args:
        horizon: Prediction horizon in days
        window_size: Size of the rolling/initial window
        step_size: Step size for the window
        multiclass: Whether to use multiclass prediction
        use_expanding_window: Whether to use expanding window approach
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*50}")
    print(f"Processing horizon: {horizon} days")
    print(f"{'='*50}")
    
    results = {}
    all_stock_data = {}
    
    # Process each stock
    for stock in STOCKS:
        # Process stock data
        df = process_stock_data(stock, horizon, window_size, step_size, multiclass)
        if df is None:
            print(f"❌ Skipping {stock} due to data processing error.")
            continue
        
        all_stock_data[stock] = df
        
        # Prepare features and target
        target_col = f'Target_{horizon}d'
        feature_cols = [col for col in df.columns if col not in [target_col, f'Future_Return_{horizon}d']]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Split data for hyperparameter optimization
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)
        
        # Optimize hyperparameters
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, is_multiclass=multiclass)
        
        # Prepare parameters for XGBoost
        params = {
            'objective': 'binary:logistic' if not multiclass else 'multi:softprob',
            'eval_metric': 'logloss' if not multiclass else 'mlogloss',
            'eta': best_params['learning_rate'],
            'max_depth': best_params['max_depth'],
            'subsample': best_params['subsample'],
            'colsample_bytree': best_params.get('colsample_bytree', 0.8),
            'min_child_weight': best_params.get('min_child_weight', 1),
            'gamma': best_params.get('gamma', 0),
            'alpha': best_params.get('reg_alpha', 0),
            'lambda': best_params.get('reg_lambda', 1),
        }
        
        if multiclass:
            params['num_class'] = len(np.unique(y))
        
        # Train model with rolling or expanding window
        if use_expanding_window:
            print(f"\nTraining expanding window model for {stock} at horizon {horizon}...")
            model_results = train_model_expanding_window(X, y, window_size, step_size, params)
        else:
            print(f"\nTraining rolling window model for {stock} at horizon {horizon}...")
            model_results = train_model_rolling_window(X, y, window_size, step_size, params)
        
        # Evaluate model
        y_pred = model_results['predictions']
        
        # For multiclass, convert predictions to class labels
        if multiclass and len(y_pred.shape) > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        # Filter out indices where predictions were not made
        valid_indices = ~np.isnan(y_pred)
        y_true_valid = y[valid_indices].values
        y_pred_valid = y_pred[valid_indices]
        
        # Evaluate model
        eval_results = evaluate_model(y_true_valid, y_pred_valid, is_multiclass=multiclass)
        
        # Print evaluation results
        window_type = "Expanding" if use_expanding_window else "Rolling"
        print(f"\n{stock} {window_type} - Accuracy: {eval_results['accuracy']:.4f}, "
              f"F1-Score: {eval_results['f1_score']:.4f}, "
              f"Baseline Accuracy: {eval_results['baseline_accuracy']:.4f}")
        
        if not multiclass and eval_results['roc_auc'] is not None:
            print(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        print(eval_results['classification_report'])
        
        # Compute SHAP values for the last model
        if model_results['models']:
            last_model = model_results['models'][-1]
            
            # Use a subset of data for SHAP analysis to speed up computation
            X_subset = X.iloc[-500:]
            
            print(f"\nComputing SHAP values for {stock} at horizon {horizon}...")
            shap_values = compute_shap_values(last_model, X_subset, is_multiclass=multiclass)
            
            # Plot SHAP summary
            output_file = f"{RESULTS_DIR}shap_summary_horizon_{horizon}_{stock}_{window_type.lower()}.png"
            plot_shap_summary(shap_values, X_subset, is_multiclass=multiclass, output_file=output_file)
        
        # Store results
        results[stock] = {
            'predictions': y_pred,
            'accuracies': model_results['accuracies'],
            'evaluation': eval_results,
            'params': params
        }
        
        # Save results to CSV
        window_type_str = "expanding" if use_expanding_window else "rolling"
        multiclass_str = "multiclass" if multiclass else "binary"
        results_file = f"{RESULTS_DIR}results_horizon_{horizon}_{stock}_{window_type_str}_{multiclass_str}.csv"
        
        # Create DataFrame with predictions
        results_df = pd.DataFrame({
            'Date': df.index[valid_indices],
            'Actual': y_true_valid,
            'Predicted': y_pred_valid,
            'Future_Return': df[f'Future_Return_{horizon}d'].values[valid_indices]
        })
        
        # Save to CSV
        results_df.to_csv(results_file)
        print(f"Saved results to {results_file}")
    
    # Add correlation features if we have data for multiple stocks
    if len(all_stock_data) > 1:
        print("\nAdding correlation features between stocks...")
        all_stock_data_with_corr = add_correlation_features(all_stock_data, list(all_stock_data.keys()))
        
        # Process each stock with correlation features
        for stock in all_stock_data_with_corr:
            df = all_stock_data_with_corr[stock]
            
            # Prepare features and target
            target_col = f'Target_{horizon}d'
            feature_cols = [col for col in df.columns if col not in [target_col, f'Future_Return_{horizon}d']]
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Split data for hyperparameter optimization
            X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, shuffle=False)
            
            # Optimize hyperparameters
            best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, is_multiclass=multiclass)
            
            # Prepare parameters for XGBoost
            params = {
                'objective': 'binary:logistic' if not multiclass else 'multi:softprob',
                'eval_metric': 'logloss' if not multiclass else 'mlogloss',
                'eta': best_params['learning_rate'],
                'max_depth': best_params['max_depth'],
                'subsample': best_params['subsample'],
                'colsample_bytree': best_params.get('colsample_bytree', 0.8),
                'min_child_weight': best_params.get('min_child_weight', 1),
                'gamma': best_params.get('gamma', 0),
                'alpha': best_params.get('reg_alpha', 0),
                'lambda': best_params.get('reg_lambda', 1),
            }
            
            if multiclass:
                params['num_class'] = len(np.unique(y))
            
            # Train model with rolling or expanding window
            if use_expanding_window:
                print(f"\nTraining expanding window model with correlation features for {stock} at horizon {horizon}...")
                model_results = train_model_expanding_window(X, y, window_size, step_size, params)
            else:
                print(f"\nTraining rolling window model with correlation features for {stock} at horizon {horizon}...")
                model_results = train_model_rolling_window(X, y, window_size, step_size, params)
            
            # Evaluate model
            y_pred = model_results['predictions']
            
            # For multiclass, convert predictions to class labels
            if multiclass and len(y_pred.shape) > 1:
                y_pred = np.argmax(y_pred, axis=1)
            
            # Filter out indices where predictions were not made
            valid_indices = ~np.isnan(y_pred)
            y_true_valid = y[valid_indices].values
            y_pred_valid = y_pred[valid_indices]
            
            # Evaluate model
            eval_results = evaluate_model(y_true_valid, y_pred_valid, is_multiclass=multiclass)
            
            # Print evaluation results
            window_type = "Expanding" if use_expanding_window else "Rolling"
            print(f"\n{stock} {window_type} with Correlation - Accuracy: {eval_results['accuracy']:.4f}, "
                  f"F1-Score: {eval_results['f1_score']:.4f}, "
                  f"Baseline Accuracy: {eval_results['baseline_accuracy']:.4f}")
            
            if not multiclass and eval_results['roc_auc'] is not None:
                print(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
            
            print("\nClassification Report:")
            print(eval_results['classification_report'])
            
            # Compute SHAP values for the last model
            if model_results['models']:
                last_model = model_results['models'][-1]
                
                # Use a subset of data for SHAP analysis to speed up computation
                X_subset = X.iloc[-500:]
                
                print(f"\nComputing SHAP values for {stock} with correlation features at horizon {horizon}...")
                shap_values = compute_shap_values(last_model, X_subset, is_multiclass=multiclass)
                
                # Plot SHAP summary
                output_file = f"{RESULTS_DIR}shap_summary_horizon_{horizon}_{stock}_{window_type.lower()}_with_corr.png"
                plot_shap_summary(shap_values, X_subset, is_multiclass=multiclass, output_file=output_file)
            
            # Store results
            results[f"{stock}_with_corr"] = {
                'predictions': y_pred,
                'accuracies': model_results['accuracies'],
                'evaluation': eval_results,
                'params': params
            }
            
            # Save results to CSV
            window_type_str = "expanding" if use_expanding_window else "rolling"
            multiclass_str = "multiclass" if multiclass else "binary"
            results_file = f"{RESULTS_DIR}results_horizon_{horizon}_{stock}_{window_type_str}_{multiclass_str}_with_corr.csv"
            
            # Create DataFrame with predictions
            results_df = pd.DataFrame({
                'Date': df.index[valid_indices],
                'Actual': y_true_valid,
                'Predicted': y_pred_valid,
                'Future_Return': df[f'Future_Return_{horizon}d'].values[valid_indices]
            })
            
            # Save to CSV
            results_df.to_csv(results_file)
            print(f"Saved results to {results_file}")
    
    return results

def main():
    """
    Main function to run the XGBoost optimization.
    """
    start_time = time.time()
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Process each horizon with different configurations
    all_results = {}
    
    # Configuration combinations
    configs = [
        {'multiclass': False, 'use_expanding_window': False, 'name': 'binary_rolling'},
        {'multiclass': False, 'use_expanding_window': True, 'name': 'binary_expanding'},
        {'multiclass': True, 'use_expanding_window': False, 'name': 'multiclass_rolling'},
        {'multiclass': True, 'use_expanding_window': True, 'name': 'multiclass_expanding'}
    ]
    
    # Process each horizon with each configuration
    for horizon in HORIZONS:
        all_results[horizon] = {}
        
        for config in configs:
            print(f"\n{'='*80}")
            print(f"Processing horizon {horizon} with {config['name']} configuration")
            print(f"{'='*80}")
            
            results = process_horizon(
                horizon=horizon,
                window_size=90,
                step_size=30,
                multiclass=config['multiclass'],
                use_expanding_window=config['use_expanding_window']
            )
            
            all_results[horizon][config['name']] = results
    
    # Summarize results
    print("\n\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    
    for horizon in HORIZONS:
        print(f"\nHorizon: {horizon} Day(s)")
        print("-" * 40)
        
        for config_name, results in all_results[horizon].items():
            print(f"\nConfiguration: {config_name}")
            print("-" * 30)
            
            for stock, result in results.items():
                if 'evaluation' in result:
                    eval_results = result['evaluation']
                    print(f"{stock} - Accuracy: {eval_results['accuracy']:.4f}, "
                          f"F1-Score: {eval_results['f1_score']:.4f}, "
                          f"Baseline Accuracy: {eval_results['baseline_accuracy']:.4f}")
                    
                    if not config_name.startswith('multiclass') and eval_results['roc_auc'] is not None:
                        print(f"ROC-AUC: {eval_results['roc_auc']:.4f}")
    
    # Create comparison plots
    print("\n\nCreating comparison plots...")
    
    # Accuracy comparison across horizons
    plt.figure(figsize=(15, 10))
    
    for config_name in [config['name'] for config in configs]:
        accuracies = []
        
        for horizon in HORIZONS:
            if horizon in all_results and config_name in all_results[horizon]:
                # Calculate average accuracy across stocks
                avg_accuracy = np.mean([
                    result['evaluation']['accuracy']
                    for stock, result in all_results[horizon][config_name].items()
                    if 'evaluation' in result
                ])
                
                accuracies.append(avg_accuracy)
        
        plt.plot(HORIZONS, accuracies, marker='o', label=config_name)
    
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Average Accuracy')
    plt.title('Accuracy Comparison Across Prediction Horizons')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}accuracy_comparison.png")
    print(f"Saved accuracy comparison plot to {RESULTS_DIR}accuracy_comparison.png")
    
    # F1-score comparison across horizons
    plt.figure(figsize=(15, 10))
    
    for config_name in [config['name'] for config in configs]:
        f1_scores = []
        
        for horizon in HORIZONS:
            if horizon in all_results and config_name in all_results[horizon]:
                # Calculate average F1-score across stocks
                avg_f1 = np.mean([
                    result['evaluation']['f1_score']
                    for stock, result in all_results[horizon][config_name].items()
                    if 'evaluation' in result
                ])
                
                f1_scores.append(avg_f1)
        
        plt.plot(HORIZONS, f1_scores, marker='o', label=config_name)
    
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Average F1-Score')
    plt.title('F1-Score Comparison Across Prediction Horizons')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{RESULTS_DIR}f1_comparison.png")
    print(f"Saved F1-score comparison plot to {RESULTS_DIR}f1_comparison.png")
    
    # End time
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()
