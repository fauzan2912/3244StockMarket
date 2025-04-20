# src/tuning/dispatcher.py

from .svm            import tune_svm
from .logistic       import tune_logistic
from .rf             import tune_random_forest
from .lstm           import tune_lstm
from .attention_lstm import tune_attention_lstm
from .deep_rnn       import tune_deep_rnn

_tuning_map = {
    "svm":             tune_svm,
    "logistic":        tune_logistic,
    "rf":              tune_random_forest,
    "lstm":            tune_lstm,
    "attention_lstm":  tune_attention_lstm,
    "deep_rnn":        tune_deep_rnn,
}

def tune_model_dispatcher(model_type: str,
                          X_train, y_train,
                          X_val,   y_val,
                          val_returns):
    """
    Dispatch to the appropriate tune_* function based on model_type.
    Returns whatever that function returns (best_params, model or model‐dict).
    """
    try:
        tuner = _tuning_map[model_type]
    except KeyError:
        valid = ", ".join(_tuning_map.keys())
        raise ValueError(f"Unknown model type '{model_type}'. Valid options are: {valid}")
    return tuner(X_train, y_train, X_val, y_val, val_returns)
