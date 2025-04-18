import numpy as np
import gc
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import accuracy_score
from evaluation.metrics import calculate_sharpe_ratio, calculate_returns
from models.xgboost import XgboostModel


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_returns: np.ndarray,
    n_iter: int = 20,
    random_state: int = 42
):
    """
    Manual randomized hyperparameter search for XgboostModel.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        X_val:   Validation features
        y_val:   Validation labels
        val_returns: Array of true returns for computing Sharpe
        n_iter: Number of random samples to draw
        random_state: Seed for reproducibility

    Returns:
        best_metrics: dict with 'best_params', 'accuracy', 'sharpe_ratio'
        model_dict:   dict with 'model' (the trained XGBClassifier) and 'scaler'
    """
    print("[TUNING] XGBoost")

    # Define distributions for sampling
    param_dist = {
        'n_estimators':    [50, 100, 200, 300],
        'max_depth':       [3, 5, 7, 9],
        'learning_rate':   [0.01, 0.05, 0.1, 0.2],
        'subsample':       [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'random_state':    [random_state]
    }

    sampler = ParameterSampler(
        param_dist,
        n_iter=n_iter,
        random_state=random_state
    )

    best_score = -np.inf
    best_params = None
    best_model = None

    for params in sampler:
        model = XgboostModel(**params)
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  ↳ skipping {params} → {e}")
            continue

        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)

        if acc > best_score:
            best_score = acc
            best_params = params
            best_model = model

        # Clean up between trials
        del model
        gc.collect()

    if best_model is None:
        raise RuntimeError("All hyperparameter configurations for XGBoost failed.")

    # Compute Sharpe on final predictions
    preds = best_model.predict(X_val)
    sharpe = calculate_sharpe_ratio(calculate_returns(preds, val_returns))

    best_metrics = {
        'best_params':  best_params,
        'accuracy':     best_score,
        'sharpe_ratio': sharpe
    }
    model_dict = {
        'model':  best_model.model,
        'scaler': best_model.scaler
    }

    return best_metrics, model_dict
