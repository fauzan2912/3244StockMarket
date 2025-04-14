# src/core/trainer.py

import os
import sys
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.core.model_factory import get_model
from src.tuning.dispatcher import tune_model_dispatcher
from src.utils.io import save_model, save_feature_cols

def train_model(model_type, stock_symbol, train_df, save=True, meta=None):
    """
    Train a model using hyperparameter tuning and optionally save it.

    Args:
        model_type: str - 'svm', 'logistic', etc.
        stock_symbol: str - 'AAPL'
        train_df: DataFrame with features + Target
        save: bool - whether to save the model and features
        meta: tuple - (test_start_date, 'rolling'/'expanding')

    Returns:
        best_model, feature_cols
    """
    feature_cols = [col for col in train_df.columns if col not in ['Stock', 'Date', 'Target', 'Returns']]
    X = train_df[feature_cols].values
    y = train_df['Target'].values

    # Split for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    val_returns = train_df['Returns'].iloc[-len(y_val):].values

    # Tune model
    results, model_with_scaler = tune_model_dispatcher(model_type, X_train, y_train, X_val, y_val, val_returns)
    model = model_with_scaler['model']

    # Save using tagged suffix
    if save:
        suffix = f"{meta[0].strftime('%Y-%m')}_{meta[1]}" if meta else "tuned"
        save_model(model, model_type, stock_symbol, suffix=suffix)
        save_feature_cols(feature_cols, stock_symbol, model_type, suffix=suffix)

    return model, feature_cols
