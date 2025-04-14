# src/tuning/dispatcher.py

from src.tuning.svm import tune_svm
# from src.tuning.logistic import tune_logistic
# from src.tuning.rf import tune_random_forest

def tune_model_dispatcher(model_type, X_train, y_train, X_val, y_val, val_returns):
    if model_type == 'svm':
        return tune_svm(X_train, y_train, X_val, y_val, val_returns)
    # elif model_type == 'logistic':
        # return tune_logistic(X_train, y_train, X_val, y_val, val_returns)
    elif model_type == 'rf':
        return tune_random_forest(X_train, y_train, X_val, y_val, val_returns)
    else:
        raise ValueError(f"No tuner implemented for model type: {model_type}")
