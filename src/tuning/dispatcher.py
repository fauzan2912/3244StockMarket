# src/tuning/dispatcher.py

from src.tuning.svm import tune_svm
from src.tuning.logistic import tune_logistic
from src.tuning.rf import tune_random_forest
from src.tuning.lstm import tune_lstm
from src.tuning.attention_lstm import tune_attention_lstm
from src.tuning.deep_rnn import tune_deep_rnn

def tune_model_dispatcher(model_type, X_train, y_train, X_val, y_val, val_returns):
    """
    Dispatch tuning based on model type.

    Returns:
        - best params dictionary (or list of dicts if model_type == 'all')
        - model object(s)
    """
    tuning_map = {
        "svm": tune_svm,
        "logistic": tune_logistic,
        "rf": tune_random_forest,
        "lstm": tune_lstm,
        "attention_lstm": tune_attention_lstm,
        "deep_rnn": tune_deep_rnn,
    }

    if model_type == "all":
        results = {}
        for mtype, func in tuning_map.items():
            print(f"\n--- Tuning model: {mtype} ---")
            best_params, model = func(X_train, y_train, X_val, y_val, val_returns)
            results[mtype] = (best_params, model)
        return results  # You'll need to handle this result differently in calling code
    elif model_type not in tuning_map:
        raise ValueError(f"Unsupported model type for tuning: {model_type}")

    return tuning_map[model_type](X_train, y_train, X_val, y_val, val_returns)
