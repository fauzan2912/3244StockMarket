# src/core/evaluator.py

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from evaluation.metrics import evaluate_trading_strategy
from src.utils.io import save_metrics, load_model, load_feature_cols
import pandas as pd

def evaluate_model(model, feature_cols, stock_symbol, test_df, window_id, return_preds=False, model_type=None):
    """
    Evaluate a model on test data and optionally return predictions.
    
    Args:
        model: Trained model
        feature_cols: List of feature column names
        stock_symbol: Stock ticker symbol
        test_df: DataFrame of test data
        window_id: Identifier for the evaluation window (e.g., '2015_rolling')
        return_preds: Whether to return predictions and strategy returns
        model_type: Optional string for labeling/logging the model type
    """
    if test_df.empty:
        print(f"[SKIP] No test data for {stock_symbol} - {window_id}")
        return (None, None, None) if return_preds else None

    X_test = test_df[feature_cols].values
    y_test = test_df['Target'].values
    test_returns = test_df['Returns'].values
    test_dates = test_df['Date'].values

    preds = model.predict(X_test)
    metrics, strategy_returns = evaluate_trading_strategy(preds, test_returns, test_dates)

    label = f"{stock_symbol}"
    if model_type:
        label += f" [{model_type}]"
    label += f" - {window_id}"

    print(f"[{label}] Sharpe: {metrics['sharpe_ratio']:.3f}, Return: {metrics['cumulative_return']:.2%}")
    save_metrics(stock_symbol, metrics, model_type, suffix=window_id)

    if return_preds:
        return metrics, strategy_returns, preds
    return metrics

def evaluate_saved_model(stock_symbol, model_type, test_df, window_id):
    """
    Load model and features from file and evaluate on test data.
    """
    model = load_model(model_type, stock_symbol, suffix=window_id)
    feature_cols = load_feature_cols(stock_symbol, model_type, suffix=window_id)
    return evaluate_model(model, feature_cols, stock_symbol, test_df, window_id)
