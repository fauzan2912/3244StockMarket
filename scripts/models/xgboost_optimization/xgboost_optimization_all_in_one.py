"""
XGBoost Optimization for Stock Market Prediction
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
warnings.filterwarnings('ignore')

# Constants
STOCKS = ['AAPL', 'MSFT', 'GOOGL']
HORIZONS = [1, 3, 7, 30, 252]
RESULTS_DIR = "./results/"
DATA_DIR = "./data/"
NUM_CLASSES_MULTICLASS = 7

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def load_stock_data(stocks, data_dir):
    """Load stock data."""
    stock_dfs = {}
    for stock in stocks:
        try:
            file_path = os.path.join(data_dir, f"{stock}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                stock_dfs[stock] = df
                print(f"✅ Loaded data for {stock} from file")
            else:
                dates = pd.date_range(start='2010-01-01', end='2017-12-31')
                np.random.seed(hash(stock) % 2**32)
                n = len(dates)
                daily_returns = np.random.normal(loc=0, scale=0.02, size=n)
                prices = 100 * np.exp(np.cumsum(daily_returns))
                df = pd.DataFrame({
                    'Date': dates,
                    'Open': prices * (1 + np.random.normal(0, 0.01, n)),
                    'High': prices * (1 + np.random.normal(0.01, 0.01, n)),
                    'Low': prices * (1 - np.random.normal(0.01, 0.01, n)),
                    'Close': prices,
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
    """Add technical indicators."""
    df = df.copy()
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['BB_Std']
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['BB_Std']
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['ROC_5'] = df['Close'].pct_change(periods=5) * 100
    df['ROC_10'] = df['Close'].pct_change(periods=10) * 100
    df['ROC_20'] = df['Close'].pct_change(periods=20) * 100
    for lag in [1, 2, 3, 5, 10]:
        df[f'Return_Lag_{lag}'] = df['Close'].pct_change(periods=lag)
    return df

def add_volatility_features(df):
    """Add volatility features."""
    df = df.copy()
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change()
    for window in [5, 10, 20, 50]:
        df[f'HV_{window}'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
    df['Vol_of_Vol_10'] = df['HV_10'].rolling(window=10).std()
    df['EWMA_Vol_10'] = df['Returns'].ewm(span=10).std() * np.sqrt(252)
    df['EWMA_Vol_20'] = df['Returns'].ewm(span=20).std() * np.sqrt(252)
    df['Vol_Ratio_5_20'] = df['HV_5'] / df['HV_20']
    df['Vol_Ratio_10_50'] = df['HV_10'] / df['HV_50']
    return df

def create_correlation_features(stock_dfs, target_stock, window_sizes=[5, 10, 20, 50]):
    """Create correlation features."""
    target_df = stock_dfs[target_stock].copy()
    returns = {stock: df['Close'].pct_change() for stock, df in stock_dfs.items()}
    for stock in stock_dfs:
        if stock != target_stock:
            for window in window_sizes:
                target_df[f'Corr_{stock}_Price_{window}d'] = target_df['Close'].rolling(window).corr(stock_dfs[stock]['Close'])
                target_df[f'Corr_{stock}_Return_{window}d'] = returns[target_stock].rolling(window).corr(returns[stock])
                cov = returns[target_stock].rolling(window).cov(returns[stock])
                var = returns[stock].rolling(window).var()
                target_df[f'Beta_{stock}_{window}d'] = cov / var.replace(0, np.nan)
    return target_df

def create_multiclass_target(df, horizon):
    """Create multiclass target."""
    df = df.copy()
    df[f'Future_Return_{horizon}d'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
    thresholds = [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]
    conditions = [
        df[f'Future_Return_{horizon}d'] <= thresholds[0],
        (df[f'Future_Return_{horizon}d'] > thresholds[0]) & (df[f'Future_Return_{horizon}d'] <= thresholds[1]),
        (df[f'Future_Return_{horizon}d'] > thresholds[1]) & (df[f'Future_Return_{horizon}d'] <= thresholds[2]),
        (df[f'Future_Return_{horizon}d'] > thresholds[2]) & (df[f'Future_Return_{horizon}d'] <= thresholds[3]),
        (df[f'Future_Return_{horizon}d'] > thresholds[3]) & (df[f'Future_Return_{horizon}d'] <= thresholds[4]),
        (df[f'Future_Return_{horizon}d'] > thresholds[4]) & (df[f'Future_Return_{horizon}d'] <= thresholds[5]),
        df[f'Future_Return_{horizon}d'] > thresholds[5]
    ]
    choices = [0, 1, 2, 3, 4, 5, 6]
    df[f'Target_MC_{horizon}d'] = np.select(conditions, choices, default=3)
    return df

def optimize_hyperparameters(X_train, y_train, X_val, y_val, is_multiclass=False):
    """Optimize hyperparameters with fixed num_class for multiclass."""
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2]
    }
    objective = 'multi:softprob' if is_multiclass else 'binary:logistic'
    eval_metric = 'mlogloss' if is_multiclass else 'logloss'
    num_class = NUM_CLASSES_MULTICLASS if is_multiclass else None
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    best_score = float('inf')
    best_params = None
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            for subsample in param_grid['subsample']:
                for colsample_bytree in param_grid['colsample_bytree']:
                    for min_child_weight in param_grid['min_child_weight']:
                        for gamma in param_grid['gamma']:
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
                            model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dval, 'eval')],
                                              early_stopping_rounds=20, verbose_eval=False)
                            score = model.best_score
                            if score < best_score:
                                best_score = score
                                best_params = params
    return best_params

def train_model_rolling_window(X, y, future_returns, stock, horizon, window_size=90, step_size=30, multiclass=False, optimize_hyperparams=True):
    """Train with rolling window, fixed broadcasting error, and added visualizations."""
    n_samples = len(X)
    predictions = np.zeros(n_samples)  # Fixed: 1D array for both binary and multiclass
    actual_returns = np.zeros(n_samples)
    valid_indices = np.zeros(n_samples, dtype=bool)
    models = []
    window_accuracies = []
    best_params = None

    n_windows = (n_samples - window_size) // step_size + 1

    for i in range(n_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size

        if end_idx >= n_samples:
            break

        X_train = X.iloc[start_idx:end_idx]
        y_train = y.iloc[start_idx:end_idx]

        test_start_idx = end_idx
        test_end_idx = min(test_start_idx + step_size, n_samples)

        if test_start_idx >= n_samples or test_end_idx <= test_start_idx:
            break

        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]
        returns_test = future_returns.iloc[test_start_idx:test_end_idx]

        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {start_idx} due to single class in y_train: {np.unique(y_train)}")
            continue

        if multiclass and (y_train.max() >= NUM_CLASSES_MULTICLASS or y_train.min() < 0):
            print(f"Invalid labels in window {i+1}: {np.unique(y_train)}")
            continue

        if optimize_hyperparams:
            train_size = int(0.8 * len(X_train))
            X_train_subset, X_val_subset = X_train.iloc[:train_size], X_train.iloc[train_size:]
            y_train_subset, y_val_subset = y_train.iloc[:train_size], y_train.iloc[train_size:]
            best_params = optimize_hyperparameters(X_train_subset, y_train_subset, X_val_subset, y_val_subset, multiclass)
        else:
            best_params = {
                'objective': 'multi:softprob' if multiclass else 'binary:logistic',
                'eval_metric': 'mlogloss' if multiclass else 'logloss',
                'eta': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0
            }
            if multiclass:
                best_params['num_class'] = NUM_CLASSES_MULTICLASS

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(best_params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')],
                          early_stopping_rounds=20, verbose_eval=False)

        pred_probs = model.predict(dtest)
        preds = np.argmax(pred_probs, axis=1) if multiclass else (pred_probs > 0.5).astype(int)
        predictions[test_start_idx:test_end_idx] = preds

        actual_returns[test_start_idx:test_end_idx] = returns_test.values
        valid_indices[test_start_idx:test_end_idx] = True
        models.append(model)

        window_acc = accuracy_score(y_test, preds)
        window_accuracies.append(window_acc)
        print(f"Window {i+1}/{n_windows}: Trained model with {len(X_train)} samples, tested on {len(X_test)} samples, Accuracy: {window_acc:.4f}")

    valid_predictions = predictions[valid_indices]
    valid_returns = actual_returns[valid_indices]
    valid_y = y.iloc[valid_indices].values

    if multiclass:
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_macro': f1_score(valid_y, valid_predictions, average='macro'),
            'f1_weighted': f1_score(valid_y, valid_predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        class_returns = {0: -0.07, 1: -0.04, 2: -0.02, 3: 0.0, 4: 0.02, 5: 0.04, 6: 0.07}
        strategy_returns = np.array([class_returns[pred] for pred in valid_predictions])
        financial_impact = {
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
    else:
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_score': f1_score(valid_y, valid_predictions),
            'roc_auc': roc_auc_score(valid_y, valid_predictions) if len(np.unique(valid_y)) > 1 else None,
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        long_returns = valid_returns[valid_predictions == 1].mean() if any(valid_predictions == 1) else 0
        short_returns = -valid_returns[valid_predictions == 0].mean() if any(valid_predictions == 0) else 0
        strategy_returns = np.where(valid_predictions == 1, valid_returns, -valid_returns)
        financial_impact = {
            'long_returns': long_returns,
            'short_returns': short_returns,
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }

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

    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(window_accuracies) + 1), window_accuracies, marker='o')
    plt.title(f"Rolling Window Accuracy Over Time ({stock}, Horizon {horizon})")
    plt.xlabel("Window Number")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"rolling_accuracy_{stock}_{horizon}_{'mc' if multiclass else 'bin'}.png"))
    plt.close()

    if multiclass:
        cm = confusion_matrix(valid_y, valid_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Rolling Confusion Matrix ({stock}, Horizon {horizon})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(RESULTS_DIR, f"rolling_cm_{stock}_{horizon}_mc.png"))
        plt.close()

    return {
        'predictions': valid_predictions,
        'actual_returns': valid_returns,
        'true_labels': valid_y,
        'eval_metrics': eval_metrics,
        'financial_impact': financial_impact,
        'models': models,
        'params': best_params if optimize_hyperparams else None
    }

def train_model_expanding_window(X, y, future_returns, stock, horizon, initial_window_size=90, step_size=30, multiclass=False, optimize_hyperparams=True):
    """Train with expanding window, fixed broadcasting error, and added visualizations."""
    n_samples = len(X)
    predictions = np.zeros(n_samples)  # Fixed: 1D array for both binary and multiclass
    actual_returns = np.zeros(n_samples)
    valid_indices = np.zeros(n_samples, dtype=bool)
    models = []
    window_accuracies = []
    best_params = None

    n_windows = (n_samples - initial_window_size) // step_size + 1

    for i in range(n_windows):
        end_idx = initial_window_size + i * step_size

        if end_idx >= n_samples:
            break

        X_train = X.iloc[:end_idx]
        y_train = y.iloc[:end_idx]

        test_start_idx = end_idx
        test_end_idx = min(test_start_idx + step_size, n_samples)

        if test_start_idx >= n_samples or test_end_idx <= test_start_idx:
            break

        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]
        returns_test = future_returns.iloc[test_start_idx:test_end_idx]

        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {end_idx} due to single class in y_train: {np.unique(y_train)}")
            continue

        if multiclass and (y_train.max() >= NUM_CLASSES_MULTICLASS or y_train.min() < 0):
            print(f"Invalid labels in window {i+1}: {np.unique(y_train)}")
            continue

        if optimize_hyperparams:
            train_size = int(0.8 * len(X_train))
            X_train_subset, X_val_subset = X_train.iloc[:train_size], X_train.iloc[train_size:]
            y_train_subset, y_val_subset = y_train.iloc[:train_size], y_train.iloc[train_size:]
            best_params = optimize_hyperparameters(X_train_subset, y_train_subset, X_val_subset, y_val_subset, multiclass)
        else:
            best_params = {
                'objective': 'multi:softprob' if multiclass else 'binary:logistic',
                'eval_metric': 'mlogloss' if multiclass else 'logloss',
                'eta': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0
            }
            if multiclass:
                best_params['num_class'] = NUM_CLASSES_MULTICLASS

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(best_params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')],
                          early_stopping_rounds=20, verbose_eval=False)

        pred_probs = model.predict(dtest)
        preds = np.argmax(pred_probs, axis=1) if multiclass else (pred_probs > 0.5).astype(int)
        predictions[test_start_idx:test_end_idx] = preds

        actual_returns[test_start_idx:test_end_idx] = returns_test.values
        valid_indices[test_start_idx:test_end_idx] = True
        models.append(model)

        window_acc = accuracy_score(y_test, preds)
        window_accuracies.append(window_acc)
        print(f"Expanding Window {i+1}/{n_windows}: Trained model with {len(X_train)} samples, tested on {len(X_test)} samples, Accuracy: {window_acc:.4f}")

    valid_predictions = predictions[valid_indices]
    valid_returns = actual_returns[valid_indices]
    valid_y = y.iloc[valid_indices].values

    if multiclass:
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_macro': f1_score(valid_y, valid_predictions, average='macro'),
            'f1_weighted': f1_score(valid_y, valid_predictions, average='weighted'),
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        class_returns = {0: -0.07, 1: -0.04, 2: -0.02, 3: 0.0, 4: 0.02, 5: 0.04, 6: 0.07}
        strategy_returns = np.array([class_returns[pred] for pred in valid_predictions])
        financial_impact = {
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
    else:
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_score': f1_score(valid_y, valid_predictions),
            'roc_auc': roc_auc_score(valid_y, valid_predictions) if len(np.unique(valid_y)) > 1 else None,
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        long_returns = valid_returns[valid_predictions == 1].mean() if any(valid_predictions == 1) else 0
        short_returns = -valid_returns[valid_predictions == 0].mean() if any(valid_predictions == 0) else 0
        strategy_returns = np.where(valid_predictions == 1, valid_returns, -valid_returns)
        financial_impact = {
            'long_returns': long_returns,
            'short_returns': short_returns,
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }

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

    # Visualizations
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(window_accuracies) + 1), window_accuracies, marker='o')
    plt.title(f"Expanding Window Accuracy Over Time ({stock}, Horizon {horizon})")
    plt.xlabel("Window Number")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, f"expanding_accuracy_{stock}_{horizon}_{'mc' if multiclass else 'bin'}.png"))
    plt.close()

    if multiclass:
        cm = confusion_matrix(valid_y, valid_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Expanding Confusion Matrix ({stock}, Horizon {horizon})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(os.path.join(RESULTS_DIR, f"expanding_cm_{stock}_{horizon}_mc.png"))
        plt.close()

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
    """Prepare data."""
    stock_dfs = load_stock_data([stock], DATA_DIR)
    if stock not in stock_dfs:
        print(f"Error: Could not load data for {stock}")
        return None, None, None
    df = stock_dfs[stock]
    df = add_technical_indicators(df)
    df = add_volatility_features(df)
    if len(stock_dfs) > 1:
        df = create_correlation_features(stock_dfs, stock)
    if multiclass:
        df = create_multiclass_target(df, horizon)
        target_col = f'Target_MC_{horizon}d'
    else:
        df[f'Future_Return_{horizon}d'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
        df[f'Target_{horizon}d'] = (df[f'Future_Return_{horizon}d'] > 0).astype(int)
        target_col = f'Target_{horizon}d'
    df = df.dropna()
    feature_cols = [col for col in df.columns if col not in [target_col, f'Future_Return_{horizon}d']]
    X = df[feature_cols]
    y = df[target_col]
    future_returns = df[f'Future_Return_{horizon}d']
    return X, y, future_returns

def run_model_comparison(stock, horizon, window_strategy='rolling', multiclass=False):
    """Run model comparison."""
    print(f"\nRunning model comparison for {stock} with horizon {horizon}...")
    print(f"Window strategy: {window_strategy}")
    print(f"Prediction type: {'Multiclass' if multiclass else 'Binary'}")
    X, y, future_returns = prepare_data_for_modeling(stock, horizon, multiclass)
    if X is None:
        print(f"Error: Could not prepare data for {stock}")
        return None
    if window_strategy == 'rolling':
        results = train_model_rolling_window(X, y, future_returns, stock, horizon, multiclass=multiclass)
    else:
        results = train_model_expanding_window(X, y, future_returns, stock, horizon, multiclass=multiclass)
    return results

def main():
    """Main function."""
    print("XGBoost Stock Market Prediction Optimization")
    print("===========================================")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    stocks = STOCKS
    horizons = HORIZONS
    results = {}
    for stock in stocks:
        stock_results = {}
        for horizon in horizons:
            print(f"\n{'='*80}")
            print(f"Processing {stock} for horizon {horizon}")
            print(f"{'='*80}")
            rolling_binary = run_model_comparison(stock, horizon, window_strategy='rolling', multiclass=False)
            expanding_binary = run_model_comparison(stock, horizon, window_strategy='expanding', multiclass=False)
            rolling_multiclass = run_model_comparison(stock, horizon, window_strategy='rolling', multiclass=True)
            expanding_multiclass = run_model_comparison(stock, horizon, window_strategy='expanding', multiclass=True)
            horizon_results = {
                'rolling_binary': rolling_binary,
                'expanding_binary': expanding_binary,
                'rolling_multiclass': rolling_multiclass,
                'expanding_multiclass': expanding_multiclass
            }
            stock_results[horizon] = horizon_results
        results[stock] = stock_results
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
    print("\nOptimization complete. Results and plots saved to:", RESULTS_DIR)

if __name__ == "__main__":
    main()