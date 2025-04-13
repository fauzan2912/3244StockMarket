#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data loader
from src.data_loader import get_stocks, get_technical_indicators
from evaluation.metrics import calculate_sharpe_ratio, calculate_returns
from 

# Define custom scorer for Sharpe ratio
def sharpe_ratio_scorer(y_true, y_pred, returns):
    """
    Custom scorer for GridSearchCV that calculates Sharpe ratio
    
    Args:
        y_true: True labels (not used, but required for scorer interface)
        y_pred: Predicted labels
        returns: Actual returns for calculating Sharpe ratio
    
    Returns:
        Sharpe ratio score
    """
    strategy_returns = calculate_returns(y_pred, returns)
    return calculate_sharpe_ratio(strategy_returns)

def prepare_data_for_stock(stock_symbol, start_date=None, end_date=None, test_size=0.2):
    """
    Prepare data for hyperparameter tuning for a single stock
    
    Args:
        stock_symbol: Stock symbol to include
        start_date: Start date for filtering data (str or datetime)
        end_date: End date for filtering data (str or datetime)
        test_size: Proportion of data to use for testing
        
    Returns:
        X_train, y_train, X_test, y_test, test_returns, feature_cols
    """
    print(f"\n--- Preparing Data for {stock_symbol} ---")
    
    # Load stock data with date filtering
    df = get_stocks(stock_symbol, start_date, end_date)
    
    # Add technical indicators
    df = get_technical_indicators(df)
    
    # Drop NaN values
    df = df.dropna()
    
    # Calculate returns for evaluation
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()  # Remove first row with NaN returns
    
    # Create feature columns
    feature_cols = [col for col in df.columns if col not in ['Stock', 'Date', 'Target', 'Returns']]
    
    print(f"--- Features selected: {len(feature_cols)} features")
    
    # Use time-based splitting to avoid lookahead bias
    df = df.sort_values('Date')
    
    # Calculate split index
    split_idx = int(len(df) * (1 - test_size))
    
    # Split data
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Debug info
    train_date_range = f"{train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')}"
    test_date_range = f"{test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}"
    print(f"--- Train date range: {train_date_range}")
    print(f"--- Test date range: {test_date_range}")
    
    # Prepare X and y
    X_train = train_df[feature_cols].values
    y_train = train_df['Target'].values
    
    X_test = test_df[feature_cols].values
    y_test = test_df['Target'].values
    
    test_returns = test_df['Returns'].values
    
    print(f"--- Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
    
    return X_train, y_train, X_test, y_test, test_returns, feature_cols

def tune_logistic_regression(X_train, y_train, X_test, y_test, test_returns):
    """
    Tune hyperparameters for logistic regression
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        test_returns: Returns for the test set
        
    Returns:
        Dictionary with best parameters and results
    """
    print("\n--- Tuning Logistic Regression Hyperparameters ---")
    
    # Define parameter grid
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear'],
        'max_iter': [1000],
        'random_state': [42]
    }
    
    # Define the model
    log_reg = LogisticRegression()
    
    # Create custom scorer that uses test returns for Sharpe ratio
    def custom_sharpe_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return sharpe_ratio_scorer(y, y_pred, test_returns)
    
    # Create grid search
    grid_search = GridSearchCV(
        estimator=log_reg,
        param_grid=param_grid,
        scoring=custom_sharpe_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    
    # Train model with best parameters
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate Sharpe ratio
    strategy_returns = calculate_returns(y_pred, test_returns)
    sharpe = calculate_sharpe_ratio(strategy_returns)
    
    # Create results dictionary
    results = {
        'best_params': best_params,
        'accuracy': accuracy,
        'sharpe_ratio': sharpe,
        'cv_results': {
            'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
            'params': [str(p) for p in grid_search.cv_results_['params']]
        }
    }
    
    print(f"--- Best Parameters: {best_params}")
    print(f"--- Test Accuracy: {accuracy:.4f}")
    print(f"--- Sharpe Ratio: {sharpe:.4f}")
    
    return results, best_model


def tune_random_forest(X_train, y_train, X_test, y_test, test_returns):
    """
    Tune hyperparameters for random forest
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        test_returns: Returns for the test set
        
    Returns:
        Dictionary with best parameters and results
    """
    print("\n--- Tuning Random Forest Hyperparameters ---")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'random_state': [42]
    }
    
    # Define the model
    rf = RandomForestRegressor()
    
    # Create custom scorer that uses test returns for Sharpe ratio
    scorer = make_scorer(r2_score)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit randomized search
    random_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = random_search.best_params_
    
    # Train model with best parameters
    best_model = RandomForestRegressor(**best_params)
    best_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate r2 score
    r2 = r2_score(y_test, y_pred)
    
    # Calculate Sharpe ratio
    strategy_returns = calculate_returns(y_pred, test_returns)
    sharpe = calculate_sharpe_ratio(strategy_returns)
    
    # Create results dictionary
    results = {
        'best_params': best_params,
        'r2 score': r2,
        'sharpe_ratio': sharpe,
        'cv_results': {
            'mean_test_score': random_search.cv_results_['mean_test_score'].tolist(),
            'params': [str(p) for p in random_search.cv_results_['params']]
        }
    }
    
    print(f"--- Best Parameters: {best_params}")
    print(f"--- R² Score on Test Set: {r2:.4f}")
    print(f"--- Sharpe Ratio: {sharpe:.4f}")
    
    return results, best_model
    

def tune_xgboost(X_train, y_train, X_test, y_test, test_returns):
    """
    Tune hyperparameters for XGBoost
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Testing features
        y_test: Testing target
        test_returns: Returns for the test set
        
    Returns:
        Dictionary with best parameters and results
    """
    print("\n--- Tuning XGBoost Hyperparameters ---")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2],
        'random_state': [42]
    }
    
    # Define the model
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    
    # Create custom scorer that uses test returns for Sharpe ratio
    def custom_sharpe_scorer(estimator, X, y):
        y_pred = estimator.predict(X)
        return sharpe_ratio_scorer(y, y_pred, test_returns)
    
    # Create randomized search for faster tuning
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=20,  # Number of parameter settings to try
        scoring=custom_sharpe_scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    # Fit randomized search
    random_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = random_search.best_params_
    
    # Train model with best parameters
    best_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')
    best_model.fit(X_train, y_train)
    
    # Make predictions on test set
    y_pred = best_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate Sharpe ratio
    strategy_returns = calculate_returns(y_pred, test_returns)
    sharpe = calculate_sharpe_ratio(strategy_returns)
    
    # Create results dictionary
    results = {
        'best_params': best_params,
        'accuracy': accuracy,
        'sharpe_ratio': sharpe,
        'cv_results': {
            'mean_test_score': random_search.cv_results_['mean_test_score'].tolist(),
            'params': [str(p) for p in random_search.cv_results_['params']]
        }
    }
    
    print(f"--- Best Parameters: {best_params}")
    print(f"--- Test Accuracy: {accuracy:.4f}")
    print(f"--- Sharpe Ratio: {sharpe:.4f}")
    
    return results, best_model

def save_params_to_json(params, model_type, stock_symbol):
    """
    Save model parameters to a JSON file in the appropriate directory
    
    Args:
        params: Dictionary with model parameters
        model_type: Type of model (logistic, rf, xgb)
        stock_symbol: Stock symbol
    """
    # Create config directory if it doesn't exist
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
    os.makedirs(config_dir, exist_ok=True)
    
    # Create stock-specific directory
    stock_dir = os.path.join(config_dir, stock_symbol)
    os.makedirs(stock_dir, exist_ok=True)
    
    # Create model-type-specific directory
    model_type_dir = os.path.join(stock_dir, model_type)
    os.makedirs(model_type_dir, exist_ok=True)
    
    # Save parameters to JSON file
    filepath = os.path.join(model_type_dir, "params.json")
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=4)
    
    # Also save to the stock-specific file in the config directory
    stock_filepath = os.path.join(config_dir, f"{stock_symbol}_{model_type}_params.json")
    with open(stock_filepath, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"--- Saved parameters for {stock_symbol} {model_type} to {filepath}")

def save_model(model, model_type, stock_symbol):
    """
    Save model to file in the appropriate directory
    
    Args:
        model: Trained model
        model_type: Type of model (logistic, rf, xgb)
        stock_symbol: Stock symbol
    """
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Create stock-specific directory
    stock_dir = os.path.join(models_dir, stock_symbol)
    os.makedirs(stock_dir, exist_ok=True)
    
    # Create model-type-specific directory
    model_type_dir = os.path.join(stock_dir, model_type)
    os.makedirs(model_type_dir, exist_ok=True)
    
    # Save model to file
    filepath = os.path.join(model_type_dir, "model.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"--- Saved {model_type} model for {stock_symbol} to {filepath}")

def save_results_to_json(results, model_type, stock_symbol):
    """
    Save tuning results to a JSON file in the appropriate directory
    
    Args:
        results: Dictionary with tuning results
        model_type: Type of model (logistic, rf, xgb)
        stock_symbol: Stock symbol
    """
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create stock-specific directory
    stock_dir = os.path.join(results_dir, stock_symbol)
    os.makedirs(stock_dir, exist_ok=True)
    
    # Create model-type-specific directory
    model_type_dir = os.path.join(stock_dir, model_type)
    os.makedirs(model_type_dir, exist_ok=True)
    
    # Save results to JSON file
    filepath = os.path.join(model_type_dir, "tuning_results.json")
    with open(filepath, 'w') as f:
        # Convert numpy values to Python native types for JSON serialization
        import json
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NpEncoder, self).default(obj)
        
        json.dump(results, f, indent=4, cls=NpEncoder)
    
    print(f"--- Saved tuning results for {stock_symbol} {model_type} to {filepath}")

def save_feature_cols(feature_cols, stock_symbol):
    """Save feature columns for a specific stock"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    # Create stock-specific directory
    stock_dir = os.path.join(models_dir, stock_symbol)
    os.makedirs(stock_dir, exist_ok=True)
    
    # Save feature columns
    feature_cols_path = os.path.join(stock_dir, "feature_cols.pkl")
    with open(feature_cols_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"--- Saved feature columns for {stock_symbol}")
    return feature_cols_path

def tune_model_for_stock(stock_symbol, model_type, start_date=None, end_date=None, test_size=0.2):
    """
    Tune hyperparameters for a specific stock and model type
    
    Args:
        stock_symbol: Stock symbol to tune for
        model_type: Type of model to tune ('logistic', 'rf', 'xgb')
        start_date: Start date for filtering data
        end_date: End date for filtering data
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary with tuning results
    """
    # Prepare data for this stock
    X_train, y_train, X_test, y_test, test_returns, feature_cols = prepare_data_for_stock(
        stock_symbol, start_date, end_date, test_size
    )
    
    # Save feature columns for this stock
    save_feature_cols(feature_cols, stock_symbol)
    
    # Tune model based on type
    if model_type == 'logistic':
        results, model = tune_logistic_regression(X_train, y_train, X_test, y_test, test_returns)
    elif model_type == 'rf':
        results, model = tune_random_forest(X_train, y_train, X_test, y_test, test_returns)
    elif model_type == 'xgb':
        results, model = tune_xgboost(X_train, y_train, X_test, y_test, test_returns)
    else:
        print(f"--- Unsupported model type: {model_type}")
        return None
    
    # Save results
    save_params_to_json(results['best_params'], model_type, stock_symbol)
    save_results_to_json(results, model_type, stock_symbol)
    save_model(model, model_type, stock_symbol)
    
    # Return results
    return {
        'stock': stock_symbol,
        'model_type': model_type,
        'best_params': results['best_params'],
        'accuracy': results['accuracy'],
        'sharpe_ratio': results['sharpe_ratio'],
        'train_samples': len(X_train),
        'test_samples': len(X_test)
    }

def main():
    """
    Main function for hyperparameter tuning
    """
    print("\n--- Starting Hyperparameter Tuning Pipeline ---\n")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Tune hyperparameters for stock price prediction models.')
    parser.add_argument('--model', type=str, default='logistic', choices=['logistic', 'rf', 'xgb', 'all'],
                        help='Model to tune (logistic, rf, xgb, or all)')
    parser.add_argument('--stocks', type=str, nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                        help='Stock symbols to use for tuning')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for data (format: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data (format: YYYY-MM-DD)')
    parser.add_argument('--year', type=int, default=None,
                        help='Specific year to use for data (e.g., 2016)')
    parser.add_argument('--tuning_method', type=str, default='random',
                        choices=['grid', 'random'],
                        help='Method for hyperparameter tuning')
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Number of iterations for randomized search')
    
    args = parser.parse_args()
    
    # If year is specified, set start_date and end_date for that year
    if args.year is not None:
        args.start_date = f"{args.year}-01-01"
        args.end_date = f"{args.year}-12-31"
        print(f"--- Using data for year {args.year} ({args.start_date} to {args.end_date})")
    
    # Determine which models to tune
    model_types = []
    if args.model == 'all':
        model_types = ['logistic', 'rf', 'xgb']
    else:
        model_types = [args.model]
    
    # Tune models for each stock and model type
    all_results = []
    
    for stock in args.stocks:
        for model_type in model_types:
            print(f"\n=== Tuning {model_type} model for {stock} ===")
            
            result = tune_model_for_stock(
                stock_symbol=stock,
                model_type=model_type,
                start_date=args.start_date,
                end_date=args.end_date,
                test_size=args.test_size
            )
            
            if result:
                all_results.append(result)
    
    # Create summary DataFrame
    if all_results:
        summary_df = pd.DataFrame(all_results)
        
        # Print summary
        print("\n--- Tuning Summary ---")
        print(summary_df[['stock', 'model_type', 'accuracy', 'sharpe_ratio']])
        
        # Save summary to CSV
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        os.makedirs(results_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(results_dir, "tuning_summary.csv"), index=False)
    
    print("\n--- Hyperparameter tuning completed successfully! ---\n")

if __name__ == "__main__":
    main()
