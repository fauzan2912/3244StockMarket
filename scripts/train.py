#!/usr/bin/env python3

import os
import sys
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import models
from src.models.model_logistic import LogisticModel

# Import data loader
from src.data_loader import get_stocks, get_technical_indicators

def prepare_data_for_stock(stock_symbol, start_date=None, end_date=None, test_size=0.2, random_state=42):
    """
    Prepare data for a single stock model
    
    Args:
        stock_symbol: Stock symbol to include
        start_date: Start date for data filtering (str or datetime)
        end_date: End date for data filtering (str or datetime)
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, X_test, y_train, y_test, test_dates, test_returns, feature_cols
    """
    print(f"\n--- Preparing Data for {stock_symbol} ---")
    
    # Load stock data with date filtering
    df = get_stocks(stock_symbol, start_date, end_date)
    
    # Add technical indicators
    df = get_technical_indicators(df)
    
    # Drop NaN values
    df = df.dropna()
    
    # Calculate returns for evaluation (not the same as the Target column)
    df['Returns'] = df['Close'].pct_change()
    df = df.dropna()  # Remove first row with NaN returns
    
    # Create feature columns (all except Stock, Date, Target and Returns)
    feature_cols = [col for col in df.columns if col not in ['Stock', 'Date', 'Target', 'Returns']]
    
    print(f"--- Features selected: {len(feature_cols)} features")
    
    # Use time-based splitting to avoid lookahead bias
    # Sort by date
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
    
    # Prepare dates and returns for evaluation
    test_dates = test_df['Date'].values
    test_returns = test_df['Returns'].values
    
    print(f"--- Data split: {len(X_train)} training samples, {len(X_test)} testing samples")
    
    return X_train, X_test, y_train, y_test, test_dates, test_returns, feature_cols

def save_model(model, model_type, stock_symbol):
    """Save a model to the models directory with stock and model type folders"""
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    
    # Create stock-specific directory
    stock_dir = os.path.join(models_dir, stock_symbol)
    os.makedirs(stock_dir, exist_ok=True)
    
    # Create model-type-specific directory
    model_type_dir = os.path.join(stock_dir, model_type)
    os.makedirs(model_type_dir, exist_ok=True)
    
    # Save model with a standard name
    model_path = os.path.join(model_type_dir, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature columns in the model type directory
    if hasattr(model, 'feature_importance') and model.feature_importance is not None:
        # Also save feature importance
        feature_importance_path = os.path.join(model_type_dir, "feature_importance.pkl")
        with open(feature_importance_path, 'wb') as f:
            pickle.dump(model.feature_importance, f)
    
    print(f"--- Saved {model_type} model for {stock_symbol} to {model_path}")
    return model_path

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

def train_model_for_stock(stock_symbol, model_type, start_date=None, end_date=None, test_size=0.2, config_path=None):
    """
    Train a model for a specific stock
    
    Args:
        stock_symbol: Stock symbol to train for
        model_type: Type of model to train ('logistic', 'rf', 'xgb', etc.)
        start_date: Start date for data filtering
        end_date: End date for data filtering
        test_size: Proportion of data to use for testing
        config_path: Path to config file with parameters
        
    Returns:
        Dictionary with model info
    """
    # Prepare data for this stock
    X_train, X_test, y_train, y_test, test_dates, test_returns, feature_cols = prepare_data_for_stock(
        stock_symbol, start_date, end_date, test_size
    )
    
    # Save feature columns for this stock
    save_feature_cols(feature_cols, stock_symbol)
    
    # Train model based on type
    if model_type == 'logistic':
        print(f"\n--- Training Logistic Regression for {stock_symbol} ---")
        
        # Check if stock-specific config file exists
        stock_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", f"{stock_symbol}_logistic_params.json"
        )
        
        if os.path.exists(stock_config_path):
            print(f"--- Found stock-specific configuration file at {stock_config_path}")
            model = LogisticModel(config_file=stock_config_path)
        elif config_path and os.path.exists(config_path):
            print(f"--- Found global configuration file at {config_path}")
            model = LogisticModel(config_file=config_path)
        else:
            print("--- No configuration file found. Using default parameters.")
            model = LogisticModel()
        
        model.train(X_train, y_train)
        
        # Get feature importance
        importance_df = model.get_feature_importance(feature_cols)
        print(f"\nTop 10 most important features for {stock_symbol}:")
        print(importance_df.head(10))
        
        # Save model
        model_path = save_model(model, "logistic", stock_symbol)
        
        # Return model info
        return {
            'stock': stock_symbol,
            'model_type': 'logistic',
            'train_samples': len(X_train),
            'test_samples': len(X_test), 
            'feature_count': len(feature_cols),
            'model_path': model_path,
            'training_accuracy': model.model.score(X_train, y_train)
        }
    
    elif model_type == 'rf':
        # Add code for Random Forest model training
        print(f"\n--- Random Forest training not yet implemented for {stock_symbol} ---")
        return None
    
    elif model_type == 'xgb':
        # Add code for XGBoost model training
        print(f"\n--- XGBoost training not yet implemented for {stock_symbol} ---")
        return None
    
    elif model_type == 'lstm':
        # Add code for LSTM model training
        print(f"\n--- LSTM training not yet implemented for {stock_symbol} ---")
        return None
    
    elif model_type == 'attention':
        # Add code for Attention LSTM model training
        print(f"\n--- Attention LSTM training not yet implemented for {stock_symbol} ---")
        return None
    
    else:
        print(f"--- Unsupported model type: {model_type}")
        return None

def main():
    """Train models for each stock separately"""
    print("\n--- Starting Model Training Pipeline (One Model Per Stock) ---\n")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train stock price prediction models (one per stock).')
    parser.add_argument('--model', type=str, default='logistic', 
                        choices=['logistic', 'rf', 'xgb', 'lstm', 'attention', 'all'],
                        help='Model type to train')
    parser.add_argument('--stocks', type=str, nargs='+', 
                        default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        help='Stock symbols to train models for')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--start_date', type=str, default=None,
                        help='Start date for data (format: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                        help='End date for data (format: YYYY-MM-DD)')
    parser.add_argument('--year', type=int, default=None,
                        help='Specific year to use for data (e.g., 2016)')
    
    args = parser.parse_args()
    
    # If year is specified, set start_date and end_date for that year
    if args.year is not None:
        args.start_date = f"{args.year}-01-01"
        args.end_date = f"{args.year}-12-31"
        print(f"--- Using data for year {args.year} ({args.start_date} to {args.end_date})")
    
    # Create directory for storing global training info
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Check for global config file
    global_config_path = None
    if args.model == 'logistic':
        global_config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "logistic_params.json"
        )
    
    # Determine which models to train
    model_types = []
    if args.model == 'all':
        model_types = ['logistic', 'rf', 'xgb', 'lstm', 'attention']
    else:
        model_types = [args.model]
    
    # Train models for each stock and model type
    all_models_info = []
    
    for stock in args.stocks:
        stock_models_info = []
        
        for model_type in model_types:
            print(f"\n=== Training {model_type} model for {stock} ===")
            
            model_info = train_model_for_stock(
                stock_symbol=stock,
                model_type=model_type,
                start_date=args.start_date,
                end_date=args.end_date,
                test_size=args.test_size,
                config_path=global_config_path
            )
            
            if model_info:
                stock_models_info.append(model_info)
        
        if stock_models_info:
            all_models_info.extend(stock_models_info)
    
    # Save overall training info
    training_info = {
        'model_types': model_types,
        'stocks': args.stocks,
        'start_date': args.start_date,
        'end_date': args.end_date,
        'year': args.year,
        'test_size': args.test_size,
        'models': all_models_info,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    training_info_path = os.path.join(models_dir, "training_info.pkl")
    with open(training_info_path, 'wb') as f:
        pickle.dump(training_info, f)
    
    print("\n--- Training Summary ---")
    print(f"Model types: {model_types}")
    print(f"Stocks: {args.stocks}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Models trained: {len(all_models_info)}")
    
    print("\n--- Model training completed successfully! ---\n")

if __name__ == "__main__":
    main()