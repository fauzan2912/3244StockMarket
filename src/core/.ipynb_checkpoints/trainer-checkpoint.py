# src/core/trainer.py

import os
import sys
from sklearn.model_selection import train_test_split
import gc
from keras import backend as K

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.core.model_factory import get_model
from src.tuning.dispatcher import tune_model_dispatcher
from src.utils.io import save_model, save_feature_cols

def train_model(model_type, stock_symbol, train_df, save=True, meta=None):
    """
    Tune and train a model on the training window.

    Returns:
        - best model (with scaler and params)
        - feature column list
    """
    feature_cols = [col for col in train_df.columns if col not in ['Stock', 'Date', 'Target', 'Returns']]
    X = train_df[feature_cols].values
    y = train_df['Target'].values

    # Validation split (time-order preserved)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    val_returns = train_df['Returns'].iloc[-len(y_val):].values

    # Clear GPU memory before tuning/training
    gc.collect()
    K.clear_session()

    # Call tuning
    best_params_dict, model_dict = tune_model_dispatcher(
        model_type,
        X_train,
        y_train,
        X_val,
        y_val,
        val_returns
    )

    # Extract and clean tuned parameters
    tuned_params = best_params_dict["best_params"]
    tuned_params_clean = {k.replace('model__', ''): v for k, v in tuned_params.items()}

    # Handle input_shape explicitly for RNN-based models
    if model_type in ["lstm", "attention_lstm", "deep_rnn"]:
        timesteps = 1
        n_features = X_train.shape[1]
        input_shape = (timesteps, n_features)
        tuned_params_clean['input_shape'] = input_shape

    # Create model object with cleaned params
    model_obj = get_model(model_type, **tuned_params_clean)

    # Assign model and scaler from tuning output
    if model_type == "rf":
        model_obj.model = model_dict
        model_obj.scaler = None
    else:
        model_obj.model = model_dict["model"]
        model_obj.scaler = model_dict["scaler"]

    if save:
        suffix = f"{meta[0].strftime('%Y-%m')}_{meta[1]}" if meta else "tuned"
        save_model(model_obj, model_type, stock_symbol, suffix=suffix)
        save_feature_cols(feature_cols, stock_symbol, model_type, suffix=suffix)

    return model_obj, feature_cols
