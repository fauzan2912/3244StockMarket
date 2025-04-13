"""
Optimized version of xgboost_optimization_all_in_one.py with improved performance,
memory management, and progress tracking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
import gc  # Added for memory management
warnings.filterwarnings('ignore')

# Constants with relative paths - reduced scope for faster testing
STOCKS = ['AAPL']  # Reduced to single stock for testing
HORIZONS = [1, 7]  # Reduced to fewer horizons
RESULTS_DIR = "./results/"
DATA_DIR = "./data/"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def load_stock_data(stocks, data_dir):
    """
    Load stock data for multiple stocks.
    
    Args:
        stocks: List of stock symbols
        data_dir: Directory containing stock data files
        
    Returns:
        Dictionary of DataFrames with stock data
    """
    print(f"Loading stock data at {time.strftime('%H:%M:%S')}")
    stock_dfs = {}
    
    for stock in stocks:
        try:
            # Try to load from actual files
            file_path = os.path.join(data_dir, f"{stock}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                stock_dfs[stock] = df
                print(f"✅ Loaded data for {stock} from file")
            else:
                # Create synthetic data if file doesn't exist
                print(f"File not found for {stock}, creating synthetic data")
                dates = pd.date_range(start='2010-01-01', end='2017-12-31')
                np.random.seed(hash(stock) % 2**32)  # Different seed for each stock
                
                # Generate random price data with some correlation structure
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
                stock_dfs[stock] = df
                print(f"✅ Created synthetic data for {stock}")
        except Exception as e:
            print(f"❌ Error loading data for {stock}: {e}")
    
    return stock_dfs

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        DataFrame with added technical indicators
    """
    print(f"Adding technical indicators at {time.strftime('%H:%M:%S')}")
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
    
    # Lagged returns
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(periods=lag)
    
    return df

def add_volatility_features(df):
    """
    Add enhanced volatility-based features.
    
    Args:
        df: DataFrame with stock price data
        
    Returns:
        DataFrame with added volatility features
    """
    print(f"Adding volatility features at {time.strftime('%H:%M:%S')}")
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
    
    # Exponentially weighted volatility
    df['EWMA_Vol_10'] = df['Returns'].ewm(span=10).std() * np.sqrt(252)
    df['EWMA_Vol_20'] = df['Returns'].ewm(span=20).std() * np.sqrt(252)
    
    # Volatility Ratio features
    df['Vol_Ratio_5_20'] = df['HV_5'] / df['HV_20']
    df['Vol_Ratio_10_50'] = df['HV_10'] / df['HV_50']
    
    return df

def create_correlation_features(stock_dfs, target_stock, window_sizes=[5, 10, 20, 50]):
    """
    Create correlation-based features for a target stock.
    
    Args:
        stock_dfs: Dictionary of DataFrames with stock data
        target_stock: Target stock symbol
        window_sizes: List of rolling window sizes
        
    Returns:
        DataFrame with correlation features
    """
    print(f"Creating correlation features for {target_stock} at {time.strftime('%H:%M:%S')}")
    # Get target stock data
    target_df = stock_dfs[target_stock].copy()
    
    # Calculate returns for all stocks
    returns = {}
    for stock, df in stock_dfs.items():
        returns[stock] = df['Close'].pct_change()
    
    # Create correlation features
    for stock in stock_dfs:
        if stock != target_stock:
            # Price correlation
            for window in window_sizes:
                target_df[f'Corr_{stock}_Price_{window}d'] = target_df['Close'].rolling(window).corr(
                    stock_dfs[stock]['Close'])
            
            # Return correlation
            for window in window_sizes:
                target_df[f'Corr_{stock}_Return_{window}d'] = returns[target_stock].rolling(window).corr(
                    returns[stock])
            
            # Beta (systematic risk)
            for window in window_sizes:
                # Calculate covariance
                cov = returns[target_stock].rolling(window).cov(returns[stock])
                
                # Calculate variance of other stock
                var = returns[stock].rolling(window).var()
                
                # Calculate beta with error handling for division by zero
                target_df[f'Beta_{stock}_{window}d'] = cov / var.replace(0, np.nan)
    
    return target_df

def create_multiclass_target(df, horizon):
    """
    Create multiclass target variable based on future returns.
    
    Args:
        df: DataFrame with stock price data
        horizon: Prediction horizon in days
        
    Returns:
        DataFrame with multiclass target variable
    """
    print(f"Creating multiclass target for horizon {horizon} at {time.strftime('%H:%M:%S')}")
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate future return
    df[f'Future_Return_{horizon}d'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
    
    # Define thresholds for multiclass prediction
    thresholds = [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]
    
    # Create multiclass target
    conditions = [
        df[f'Future_Return_{horizon}d'] <= thresholds[0],  # Down >5%
        (df[f'Future_Return_{horizon}d'] > thresholds[0]) & (df[f'Future_Return_{horizon}d'] <= thresholds[1]),  # Down 3-5%
        (df[f'Future_Return_{horizon}d'] > thresholds[1]) & (df[f'Future_Return_{horizon}d'] <= thresholds[2]),  # Down 1-3%
        (df[f'Future_Return_{horizon}d'] > thresholds[2]) & (df[f'Future_Return_{horizon}d'] <= thresholds[3]),  # Down <1% to Up <1%
        (df[f'Future_Return_{horizon}d'] > thresholds[3]) & (df[f'Future_Return_{horizon}d'] <= thresholds[4]),  # Up 1-3%
        (df[f'Future_Return_{horizon}d'] > thresholds[4]) & (df[f'Future_Return_{horizon}d'] <= thresholds[5]),  # Up 3-5%
        df[f'Future_Return_{horizon}d'] > thresholds[5]  # Up >5%
    ]
    
    choices = [0, 1, 2, 3, 4, 5, 6]
    df[f'Target_MC_{horizon}d'] = np.select(conditions, choices, default=3)
    
    # Print class distribution
    print("\nClass distribution for horizon {}:".format(horizon))
    class_counts = df[f'Target_MC_{horizon}d'].value_counts().sort_index()
    total_samples = len(df)
    
    class_names = ["Down >5%", "Down 3-5%", "Down 1-3%", "Down <1% to Up <1%", "Up 1-3%", "Up 3-5%", "Up >5%"]
    
    for class_idx, count in class_counts.items():
        if class_idx < len(class_names):
            print(f"{class_names[class_idx]}: {count} samples ({count/total_samples*100:.2f}%)")
    
    return df

def optimize_hyperparameters(X_train, y_train, X_val, y_val, is_multiclass=False):
    """
    Optimize XGBoost hyperparameters using a simple grid search.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        is_multiclass: Whether it's a multiclass problem
        
    Returns:
        Dictionary with best parameters
    """
    print(f"Optimizing hyperparameters at {time.strftime('%H:%M:%S')}")
    # Define parameter grid - reduced for faster execution
    param_grid = {
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.2],
        'subsample': [0.8],
        'colsample_bytree': [0.8],
        'min_child_weight': [1],
        'gamma': [0]
    }
    
    # Set objective and evaluation metric based on problem type
    if is_multiclass:
        objective = 'multi:softprob'
        eval_metric = 'mlogloss'
        num_class = len(np.unique(y_train))
    else:
        objective = 'binary:logistic'
        eval_metric = 'logloss'
        num_class = None
    
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Initialize best score and parameters
    best_score = float('inf')
    best_params = None
    
    # Simple grid search
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for subsample in param_grid['subsample']:
                for colsample_bytree in param_grid['colsample_bytree']:
                    for min_child_weight in param_grid['min_child_weight']:
                        for gamma in param_grid['gamma']:
                            # Set parameters
                            params = {
                                'objective': objective,
                                'eval_metric': eval_metric,
                                'max_depth': max_depth,
                                'eta': learning_rate,
                                'subsample': subsample,
                                'colsample_bytree': colsample_bytree,
                                'min_child_weight': min_child_weight,
                                'gamma': gamma
                            }
                            
                            if is_multiclass:
                                params['num_class'] = num_class
                            
                            # Train model
                            model = xgb.train(
                                params,
                                dtrain,
                                num_boost_round=50,  # Reduced for faster execution
                                evals=[(dval, 'eval')],
                                early_stopping_rounds=10,  # Reduced for faster execution
                                verbose_eval=False
                            )
                            
                            # Get validation score
                            score = model.best_score
                            
                            # Update best parameters if better score
                            if score < best_score:
                                best_score = score
                                best_params = params
    
    return best_params

def train_model_rolling_window(X, y, future_returns, window_size=90, step_size=30, multiclass=False, optimize_hyperparams=True):
    """
    Train XGBoost model using rolling window approach.
    
    Args:
        X: Feature matrix
        y: Target vector
        future_returns: Future returns for financial impact analysis
        window_size: Size of the rolling window
        step_size: Step size for the rolling window
        multiclass: Whether it's a multiclass problem
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Training rolling window model at {time.strftime('%H:%M:%S')}")
    n_samples = len(X)
    predictions = np.zeros(n_samples) if not multiclass else np.zeros((n_samples, len(np.unique(y))))
    actual_returns = np.zeros(n_samples)
    valid_indices = np.zeros(n_samples, dtype=bool)
    models = []
    
    # Determine the number of windows
    n_windows = (n_samples - window_size) // step_size + 1
    
    # Limit number of windows for faster execution
    n_windows = min(n_windows, 3)
    
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
        returns_test = future_returns.iloc[test_start_idx:test_end_idx]
        
        # Skip if training data has only one class
        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {start_idx} due to single class in y_train: {np.unique(y_train)}")
            continue
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            # Split training data for validation
            train_size = int(0.8 * len(X_train))
            X_train_subset, X_val_subset = X_train.iloc[:train_size], X_train.iloc[train_size:]
            y_train_subset, y_val_subset = y_train.iloc[:train_size], y_train.iloc[train_size:]
            
            # Optimize hyperparameters
            best_params = optimize_hyperparameters(
                X_train_subset, y_train_subset, X_val_subset, y_val_subset, is_multiclass=multiclass
            )
        else:
            # Use default parameters
            if multiclass:
                best_params = {
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'eta': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'num_class': len(np.unique(y))
                }
            else:
                best_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'eta': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0
                }
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(best_params, dtrain, num_boost_round=50, evals=[(dtest, 'eval')], 
                          early_stopping_rounds=10, verbose_eval=False)
        
        # Make predictions
        if multiclass:
            # Multiclass classification
            pred_probs = model.predict(dtest)
            preds = np.argmax(pred_probs, axis=1)
            predictions[test_start_idx:test_end_idx] = preds
        else:
            # Binary classification
            pred_probs = model.predict(dtest)
            preds = (pred_probs > 0.5).astype(int)
            predictions[test_start_idx:test_end_idx] = preds
        
        # Store actual returns and mark valid indices
        actual_returns[test_start_idx:test_end_idx] = returns_test.values
        valid_indices[test_start_idx:test_end_idx] = True
        
        # Store model
        models.append(model)
        
        print(f"Window {i+1}/{n_windows}: Trained model with {len(X_train)} samples, tested on {len(X_test)} samples")
        
        # Clean up memory
        gc.collect()
    
    # Filter valid predictions and returns
    valid_predictions = predictions[valid_indices]
    valid_returns = actual_returns[valid_indices]
    valid_y = y.iloc[valid_indices].values
    
    # Evaluate predictions
    if multiclass:
        # Multiclass evaluation
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_macro': f1_score(valid_y, valid_predictions, average='macro'),
            'f1_weighted': f1_score(valid_y, valid_predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        
        # Financial impact for multiclass
        # Assuming class meanings: 0=Down>5%, 1=Down3-5%, 2=Down1-3%, 3=Flat, 4=Up1-3%, 5=Up3-5%, 6=Up>5%
        class_returns = {
            0: -0.07,  # Down >5%
            1: -0.04,  # Down 3-5%
            2: -0.02,  # Down 1-3%
            3: 0.0,    # Flat
            4: 0.02,   # Up 1-3%
            5: 0.04,   # Up 3-5%
            6: 0.07    # Up >5%
        }
        
        # Calculate strategy returns based on predicted class
        strategy_returns = np.array([class_returns.get(pred, 0) for pred in valid_predictions])
        
        financial_impact = {
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
    else:
        # Binary evaluation
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_score': f1_score(valid_y, valid_predictions),
            'roc_auc': roc_auc_score(valid_y, valid_predictions) if len(np.unique(valid_y)) > 1 else None,
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        
        # Simple financial impact for binary
        long_returns = valid_returns[valid_predictions == 1].mean() if any(valid_predictions == 1) else 0
        short_returns = -valid_returns[valid_predictions == 0].mean() if any(valid_predictions == 0) else 0
        strategy_returns = np.where(valid_predictions == 1, valid_returns, -valid_returns)
        
        financial_impact = {
            'long_returns': long_returns,
            'short_returns': short_returns,
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
    
    # Print evaluation results
    print("\nEvaluation results:")
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"F1-Score: {eval_metrics['f1_score'] if not multiclass else eval_metrics['f1_macro']:.4f}")
    
    if not multiclass and eval_metrics['roc_auc'] is not None:
        print(f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    
    print("\nFinancial impact:")
    if multiclass:
        print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
        print(f"Strategy Sharpe: {financial_impact['strategy_sharpe']:.4f}")
    else:
        print(f"Long Returns: {financial_impact['long_returns']:.4f}")
        print(f"Short Returns: {financial_impact['short_returns']:.4f}")
        print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
        print(f"Strategy Sharpe: {financial_impact['strategy_sharpe']:.4f}")
    
    # Return results
    return {
        'predictions': valid_predictions,
        'actual_returns': valid_returns,
        'true_labels': valid_y,
        'eval_metrics': eval_metrics,
        'financial_impact': financial_impact,
        'models': models,
        'params': best_params if optimize_hyperparams else None
    }

def train_model_expanding_window(X, y, future_returns, initial_window_size=90, step_size=30, multiclass=False, optimize_hyperparams=True):
    """
    Train XGBoost model using expanding window approach.
    
    Args:
        X: Feature matrix
        y: Target vector
        future_returns: Future returns for financial impact analysis
        initial_window_size: Initial size of the window
        step_size: Step size for expanding the window
        multiclass: Whether it's a multiclass problem
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"Training expanding window model at {time.strftime('%H:%M:%S')}")
    n_samples = len(X)
    predictions = np.zeros(n_samples) if not multiclass else np.zeros((n_samples, len(np.unique(y))))
    actual_returns = np.zeros(n_samples)
    valid_indices = np.zeros(n_samples, dtype=bool)
    models = []
    
    # Determine the number of windows
    n_windows = (n_samples - initial_window_size) // step_size + 1
    
    # Limit number of windows for faster execution
    n_windows = min(n_windows, 3)
    
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
        returns_test = future_returns.iloc[test_start_idx:test_end_idx]
        
        # Skip if training data has only one class
        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {end_idx} due to single class in y_train: {np.unique(y_train)}")
            continue
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            # Split training data for validation
            train_size = int(0.8 * len(X_train))
            X_train_subset, X_val_subset = X_train.iloc[:train_size], X_train.iloc[train_size:]
            y_train_subset, y_val_subset = y_train.iloc[:train_size], y_train.iloc[train_size:]
            
            # Optimize hyperparameters
            best_params = optimize_hyperparameters(
                X_train_subset, y_train_subset, X_val_subset, y_val_subset, is_multiclass=multiclass
            )
        else:
            # Use default parameters
            if multiclass:
                best_params = {
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'eta': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'num_class': len(np.unique(y))
                }
            else:
                best_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'eta': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0
                }
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(best_params, dtrain, num_boost_round=50, evals=[(dtest, 'eval')], 
                          early_stopping_rounds=10, verbose_eval=False)
        
        # Make predictions
        if multiclass:
            # Multiclass classification
            pred_probs = model.predict(dtest)
            preds = np.argmax(pred_probs, axis=1)
            predictions[test_start_idx:test_end_idx] = preds
        else:
            # Binary classification
            pred_probs = model.predict(dtest)
            preds = (pred_probs > 0.5).astype(int)
            predictions[test_start_idx:test_end_idx] = preds
        
        # Store actual returns and mark valid indices
        actual_returns[test_start_idx:test_end_idx] = returns_test.values
        valid_indices[test_start_idx:test_end_idx] = True
        
        # Store model
        models.append(model)
        
        print(f"Expanding Window {i+1}/{n_windows}: Trained model with {len(X_train)} samples, tested on {len(X_test)} samples")
        
        # Clean up memory
        gc.collect()
    
    # Filter valid predictions and returns
    valid_predictions = predictions[valid_indices]
    valid_returns = actual_returns[valid_indices]
    valid_y = y.iloc[valid_indices].values
    
    # Evaluate predictions
    if multiclass:
        # Multiclass evaluation
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_macro': f1_score(valid_y, valid_predictions, average='macro'),
            'f1_weighted': f1_score(valid_y, valid_predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        
        # Financial impact for multiclass
        # Assuming class meanings: 0=Down>5%, 1=Down3-5%, 2=Down1-3%, 3=Flat, 4=Up1-3%, 5=Up3-5%, 6=Up>5%
        class_returns = {
            0: -0.07,  # Down >5%
            1: -0.04,  # Down 3-5%
            2: -0.02,  # Down 1-3%
            3: 0.0,    # Flat
            4: 0.02,   # Up 1-3%
            5: 0.04,   # Up 3-5%
            6: 0.07    # Up >5%
        }
        
        # Calculate strategy returns based on predicted class
        strategy_returns = np.array([class_returns.get(pred, 0) for pred in valid_predictions])
        
        financial_impact = {
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
    else:
        # Binary evaluation
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_score': f1_score(valid_y, valid_predictions),
            'roc_auc': roc_auc_score(valid_y, valid_predictions) if len(np.unique(valid_y)) > 1 else None,
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        
        # Simple financial impact for binary
        long_returns = valid_returns[valid_predictions == 1].mean() if any(valid_predictions == 1) else 0
        short_returns = -valid_returns[valid_predictions == 0].mean() if any(valid_predictions == 0) else 0
        strategy_returns = np.where(valid_predictions == 1, valid_returns, -valid_returns)
        
        financial_impact = {
            'long_returns': long_returns,
            'short_returns': short_returns,
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
    
    # Print evaluation results
    print("\nEvaluation results:")
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"F1-Score: {eval_metrics['f1_score'] if not multiclass else eval_metrics['f1_macro']:.4f}")
    
    if not multiclass and eval_metrics['roc_auc'] is not None:
        print(f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    
    print("\nFinancial impact:")
    if multiclass:
        print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
        print(f"Strategy Sharpe: {financial_impact['strategy_sharpe']:.4f}")
    else:
        print(f"Long Returns: {financial_impact['long_returns']:.4f}")
        print(f"Short Returns: {financial_impact['short_returns']:.4f}")
        print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
        print(f"Strategy Sharpe: {financial_impact['strategy_sharpe']:.4f}")
    
    # Return results
    return {
        'predictions': valid_predictions,
        'actual_returns': valid_returns,
        'true_labels': valid_y,
        'eval_metrics': eval_metrics,
        'financial_impact': financial_impact,
        'models': models,
        'params': best_params if optimize_hyperparams else None
    }

def prepare_data_for_modeling(stock, horizon, multiclass=False):
    """
    Prepare data for modeling for a specific stock and horizon.
    
    Args:
        stock: Stock symbol
        horizon: Prediction horizon
        multiclass: Whether to create multiclass target
        
    Returns:
        X, y, future_returns
    """
    print(f"Preparing data for {stock} with horizon {horizon} at {time.strftime('%H:%M:%S')}")
    # Load stock data
    stock_dfs = load_stock_data([stock], DATA_DIR)
    
    if stock not in stock_dfs:
        print(f"Error: Could not load data for {stock}")
        return None, None, None
    
    # Get stock data
    df = stock_dfs[stock]
    
    # Add technical indicators
    df = add_technical_indicators(df)
    
    # Add volatility features
    df = add_volatility_features(df)
    
    # Add correlation features if more than one stock
    if len(stock_dfs) > 1:
        df = create_correlation_features(stock_dfs, stock)
    
    # Create target variable
    if multiclass:
        df = create_multiclass_target(df, horizon)
        target_col = f'Target_MC_{horizon}d'
    else:
        df[f'Future_Return_{horizon}d'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
        df[f'Target_{horizon}d'] = (df[f'Future_Return_{horizon}d'] > 0).astype(int)
        target_col = f'Target_{horizon}d'
    
    # Drop rows with NaN
    df = df.dropna()
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in [target_col, f'Future_Return_{horizon}d']]
    X = df[feature_cols]
    y = df[target_col]
    future_returns = df[f'Future_Return_{horizon}d']
    
    return X, y, future_returns

def run_model_comparison(stock, horizon, window_strategy='rolling', multiclass=False):
    """
    Run model comparison for a specific stock and horizon.
    
    Args:
        stock: Stock symbol
        horizon: Prediction horizon
        window_strategy: 'rolling' or 'expanding'
        multiclass: Whether to use multiclass prediction
        
    Returns:
        Dictionary with results
    """
    print(f"\nRunning model comparison for {stock} with horizon {horizon}...")
    print(f"Window strategy: {window_strategy}")
    print(f"Prediction type: {'Multiclass' if multiclass else 'Binary'}")
    
    # Prepare data
    X, y, future_returns = prepare_data_for_modeling(stock, horizon, multiclass)
    
    if X is None:
        print(f"Error: Could not prepare data for {stock}")
        return None
    
    # Train and evaluate model
    if window_strategy == 'rolling':
        results = train_model_rolling_window(X, y, future_returns, multiclass=multiclass)
    else:  # expanding
        results = train_model_expanding_window(X, y, future_returns, multiclass=multiclass)
    
    # Clean up memory
    gc.collect()
    
    return results

def main():
    """
    Main function to run the XGBoost optimization.
    """
    start_time = time.time()
    print(f"XGBoost Stock Market Prediction Optimization - Started at {time.strftime('%H:%M:%S')}")
    print("===========================================")
    
    # Create directories
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Define stocks to analyze
    stocks = STOCKS
    
    # Define horizons to predict
    horizons = HORIZONS
    
    # Run comparison for each stock and horizon
    results = {}
    
    for stock in stocks:
        stock_results = {}
        
        for horizon in horizons:
            print(f"\n{'='*80}")
            print(f"Processing {stock} for horizon {horizon} at {time.strftime('%H:%M:%S')}")
            print(f"{'='*80}")
            
            # Run rolling window with binary prediction
            rolling_binary = run_model_comparison(stock, horizon, window_strategy='rolling', multiclass=False)
            
            # Run expanding window with binary prediction
            expanding_binary = run_model_comparison(stock, horizon, window_strategy='expanding', multiclass=False)
            
            # Run rolling window with multiclass prediction
            rolling_multiclass = run_model_comparison(stock, horizon, window_strategy='rolling', multiclass=True)
            
            # Run expanding window with multiclass prediction
            expanding_multiclass = run_model_comparison(stock, horizon, window_strategy='expanding', multiclass=True)
            
            # Store results
            horizon_results = {
                'rolling_binary': rolling_binary,
                'expanding_binary': expanding_binary,
                'rolling_multiclass': rolling_multiclass,
                'expanding_multiclass': expanding_multiclass
            }
            
            stock_results[horizon] = horizon_results
            
            # Clean up memory
            gc.collect()
        
        results[stock] = stock_results
    
    # Compare results
    print("\n\nSummary of Results")
    print("=================")
    
    for stock in results:
        print(f"\nResults for {stock}:")
        
        for horizon in results[stock]:
            print(f"\n  Horizon: {horizon} days")
            
            for model_type, result in results[stock][horizon].items():
                if result is None:
                    continue
                
                print(f"    {model_type}:")
                print(f"      Accuracy: {result['eval_metrics']['accuracy']:.4f}")
                
                if 'multiclass' in model_type:
                    print(f"      F1 (macro): {result['eval_metrics']['f1_macro']:.4f}")
                else:
                    print(f"      F1: {result['eval_metrics']['f1_score']:.4f}")
                    if result['eval_metrics']['roc_auc'] is not None:
                        print(f"      ROC-AUC: {result['eval_metrics']['roc_auc']:.4f}")
                
                print(f"      Strategy Return: {result['financial_impact']['strategy_return']:.4f}")
                print(f"      Strategy Sharpe: {result['financial_impact']['strategy_sharpe']:.4f}")
    
    # End time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nOptimization complete at {time.strftime('%H:%M:%S')}. Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
