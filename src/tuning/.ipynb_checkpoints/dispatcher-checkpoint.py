# src/tuning/dispatcher.py

from src.tuning.svm import tune_svm
from src.tuning.rf import tune_random_forest
from src.tuning.lstm import tune_lstm
from src.tuning.attention_lstm import tune_attention_lstm

def tune_model_dispatcher(model_type, X_train, y_train, X_val, y_val, val_returns):
    """
    Dispatch tuning based on model type.

    Returns:
        - best params dictionary
        - model object (including scaler if applicable)
    """
    tuning_map = {
        "svm": tune_svm,
        "rf": tune_random_forest,
        "lstm": tune_lstm,
        "attention_lstm": tune_attention_lstm,
    }

    if model_type not in tuning_map:
        raise ValueError(f"Unsupported model type for tuning: {model_type}")

    return tuning_map[model_type](X_train, y_train, X_val, y_val, val_returns)
