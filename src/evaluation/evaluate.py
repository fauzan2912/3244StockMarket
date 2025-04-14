#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import data loader
from data_loader import get_stocks, get_technical_indicators

# Import evaluation modules
from evaluation.metrics import evaluate_trading_strategy
from evaluation.visualizer import plot_cumulative_returns, plot_confusion_matrix

def load_training_info():
    """Load the overall training info"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    training_info_path = os.path.join(models_dir, "training_info.pkl")
    
    if os.path.exists(training_info_path):
        with open(training_info_path, 'rb') as f:
            return pickle.load(f)
    return None

def load_model(model_type, stock_symbol):
    """Load a model for a specific stock and model type"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    stock_dir = os.path.join(models_dir, stock_symbol)
    model_type_dir = os.path.join(stock_dir, model_type)
    model_path = os.path.join(model_type_dir, "model.pkl")
    
    if not os.path.exists(model_path):
        print(f"--- Model not found for {stock_symbol} at {model_path}")
        return None
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    return model

def load_feature_cols(stock_symbol):
    """Load feature columns for a specific stock"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    stock_dir = os.path.join(models_dir, stock_symbol)
    feature_cols_path = os.path.join(stock_dir, "feature_cols.pkl")
    
    if not os.path.exists(feature_cols_path):
        print(f"--- Feature columns not found for {stock_symbol}")
        return None
    
    with open(feature_cols_path, 'rb') as f:
        feature_cols = pickle.load(f)
    
    return feature_cols

def prepare_data_for_stock(stock_symbol, start_date=None, end_date=None, test_size=0.2):
    """
    Prepare data for evaluating a single stock model
    
    Args:
        stock_symbol: Stock symbol to evaluate
        start_date: Start date for filtering data (str or datetime)
        end_date: End date for filtering data (str or datetime)
        test_size: Proportion of data to use for testing
        
    Returns:
        X_test, y_test, test_dates, test_returns
    """
    print(f"\n--- Preparing Evaluation Data for {stock_symbol} ---")
    
    # Load stock data with date filtering
    df = get_stocks(stock_symbol, start_date, end_date)
    
    # Add technical indicators
    df = get_technical_indicators(df)
    
    # Drop NaN values
    df = df.dropna()
    
    # Calculate returns for evaluation
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()  # Remove first row with NaN returns
    
    # Load feature columns for this stock
    feature_cols = load_feature_cols(stock_symbol)
    if feature_cols is None:
        print(f"--- Cannot evaluate {stock_symbol} without feature columns")
        return None, None, None, None
    
    print(f"--- Features selected: {len(feature_cols)} features")
    
    # Use time-based splitting to avoid lookahead bias
    df = df.sort_values('Date')
    
    # Calculate split index
    split_idx = int(len(df) * (1 - test_size))
    
    # Get only the test data
    test_df = df.iloc[split_idx:]
    
    # Debug info
    test_date_range = f"{test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')}"
    print(f"--- Test date range: {test_date_range}")
    
    # Prepare X and y
    X_test = test_df[feature_cols].values
    y_test = test_df['Target'].values
    
    # Prepare dates and returns for evaluation
    test_dates = test_df['Date'].values
    test_returns = test_df['Returns'].values
    
    print(f"--- Test data prepared: {len(X_test)} testing samples")
    
    return X_test, y_test, test_dates, test_returns

def evaluate_model_for_stock(stock_symbol, model_type, start_date=None, end_date=None, test_size=0.2):
    """
    Evaluate a model for a specific stock
    
    Args:
        stock_symbol: Stock symbol to evaluate
        model_type: Type of model to evaluate ('logistic', 'rf', 'xgb', etc.)
        start_date: Start date for filtering data
        end_date: End date for filtering data
        test_size: Proportion of data to use for testing
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Prepare data for this stock
    if (model_type == "lstm"):
        X_test, y_test, test_dates, test_returns = prepare_data_for_stock(
        stock_symbol, start_date, end_date, test_size
        )
        y_test, test_dates, test_returns = y_test[90:], test_dates[90:], test_returns[90:]
    else:
        X_test, y_test, test_dates, test_returns = prepare_data_for_stock(
            stock_symbol, start_date, end_date, test_size
        )
    
    if X_test is None:
        return None
    
    # Load model for this stock
    print(f"\n--- Loading {model_type} model for {stock_symbol} ---")
    model = load_model(model_type, stock_symbol)
    
    if model is None:
        return None
    
    # Make predictions
    print(f"--- Making predictions for {stock_symbol} ---")
    predictions = model.predict(X_test)
    
    # Evaluate trading strategy
    print(f"--- Evaluating trading strategy for {stock_symbol} ---")
    metrics, strategy_returns = evaluate_trading_strategy(
        predictions, test_returns, test_dates
    )
    
    # Print metrics
    print(f"\n--- {stock_symbol} {model_type.capitalize()} Performance ---")
    print(f"Cumulative Return: {metrics['cumulative_return']:.4f}")
    print(f"Annualized Return: {metrics['annualized_return']:.4f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Maximum Drawdown: {metrics['max_drawdown']:.4f}")
    print(f"Win Rate: {metrics['win_rate']:.4f}")
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Create stock-specific directory
    stock_dir = os.path.join(results_dir, stock_symbol)
    os.makedirs(stock_dir, exist_ok=True)
    
    # Create model-type-specific directory
    model_type_dir = os.path.join(stock_dir, model_type)
    os.makedirs(model_type_dir, exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        y_test, predictions, f"{stock_symbol} {model_type.capitalize()}",
        save_path=os.path.join(model_type_dir, "confusion_matrix.png")
    )
    
    # Plot cumulative returns
    plot_cumulative_returns(
        {f"{stock_symbol} {model_type.capitalize()}": predictions}, 
        test_returns, test_dates,
        save_path=os.path.join(model_type_dir, "cumulative_returns.png")
    )
    
    # Save metrics to JSON
    metrics_path = os.path.join(model_type_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        import json
        json.dump(metrics, f, indent=4)
    
    # Return evaluation results
    return {
        'stock': stock_symbol,
        'model_type': model_type,
        'metrics': metrics,
        'test_samples': len(X_test),
        'test_date_range': f"{test_dates[0]} to {test_dates[-1]}",
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

def main():
    """Evaluate models for each stock separately"""
    print("\n--- Starting Model Evaluation Pipeline (One Model Per Stock) ---\n")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate stock price prediction models (one per stock).')
    parser.add_argument('--model', type=str, default='logistic', 
                        choices=['logistic', 'rf', 'xgb', 'lstm', 'attention', 'all'],
                        help='Model type to evaluate')
    parser.add_argument('--stocks', type=str, nargs='+', default=None,
                        help='Stock symbols to evaluate models for (default: use stocks from training)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for data (format: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data (format: YYYY-MM-DD)')
    parser.add_argument('--year', type=int, default=None,
                        help='Specific year to use for data (e.g., 2016)')
    parser.add_argument('--use_training_info', action='store_true', default=True,
                        help='Use training info for evaluation settings')
    
    args = parser.parse_args()
    
    # If year is specified, set start_date and end_date for that year
    if args.year is not None:
        args.start_date = f"{args.year}-01-01"
        args.end_date = f"{args.year}-12-31"
        print(f"--- Using data for year {args.year} ({args.start_date} to {args.end_date})")
        # If user specifies year, don't use training info
        args.use_training_info = False
    
    # Load training info if requested
    training_info = None
    if args.use_training_info:
        training_info = load_training_info()
        if training_info:
            print("--- Using settings from training info")
            if not args.stocks:
                args.stocks = training_info.get('stocks')
            if not args.start_date:
                args.start_date = training_info.get('start_date')
            if not args.end_date:
                args.end_date = training_info.get('end_date')
            if args.test_size == 0.2 and 'test_size' in training_info:  # Only if using default
                args.test_size = training_info.get('test_size')
                
            print(f"--- Using: stocks={args.stocks}, dates={args.start_date} to {args.end_date}")
    
    if not args.stocks:
        print("--- No stocks specified. Using default stocks.")
        args.stocks = ["BAC","AAPL","GE", "F", "MSFT", "SIRI", "INTC", "CSCO", "PFE", "HPQ"] # Top 10 median trading volume
    
    # Determine which models to evaluate
    model_types = []
    if args.model == 'all':
        if training_info and 'model_types' in training_info:
            model_types = training_info['model_types']
        else:
            model_types = ['logistic', 'rf', 'xgb', 'lstm', 'attention']
    else:
        model_types = [args.model]
    
    # Create results directory for storing overall results
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Evaluate models for each stock and model type
    all_results = {}
    
    for model_type in model_types:
        model_results = []
        
        for stock in args.stocks:
            result = evaluate_model_for_stock(
                stock_symbol=stock,
                model_type=model_type,
                start_date=args.start_date,
                end_date=args.end_date,
                test_size=args.test_size
            )
            
            if result:
                model_results.append(result)
        
        if model_results:
            all_results[model_type] = model_results
            
            # Create summary DataFrame for this model type
            metrics_df = pd.DataFrame([
                {
                    'Stock': r['stock'],
                    'Cumulative Return': r['metrics']['cumulative_return'],
                    'Annualized Return': r['metrics']['annualized_return'],
                    'Sharpe Ratio': r['metrics']['sharpe_ratio'],
                    'Max Drawdown': r['metrics']['max_drawdown'],
                    'Win Rate': r['metrics']['win_rate']
                }
                for r in model_results
            ])
            
            # Save metrics to CSV
            metrics_df.to_csv(os.path.join(results_dir, f"{model_type}_metrics.csv"), index=False)
            
            # Print summary
            print(f"\n--- {model_type.capitalize()} Model Summary ---")
            print(metrics_df[['Stock', 'Cumulative Return', 'Sharpe Ratio', 'Win Rate']])
            
            # Identify best stock by Sharpe ratio
            best_stock = metrics_df.loc[metrics_df['Sharpe Ratio'].idxmax()]
            print(f"\n--- Best stock by Sharpe ratio: {best_stock['Stock']} (Sharpe: {best_stock['Sharpe Ratio']:.4f}, Return: {best_stock['Cumulative Return']:.4f})")
    
    # Save overall evaluation results
    evaluation_info = {
        'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_types': model_types,
        'stocks': args.stocks,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'year': args.year,
        'test_size': args.test_size,
        'results': all_results
    }
    
    with open(os.path.join(results_dir, "evaluation_info.pkl"), 'wb') as f:
        pickle.dump(evaluation_info, f)
    
    print(f"\n--- Evaluation complete. Results saved to: {results_dir}")

if __name__ == "__main__":
    main()