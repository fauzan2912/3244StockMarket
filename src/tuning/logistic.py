# src/tuning/logistic.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
from evaluation.metrics import calculate_sharpe_ratio, calculate_returns

def tune_logistic(X_train, y_train, X_val, y_val, val_returns):
    print("[TUNING] Logistic Regression")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    param_grid = [
        {'penalty': ['l2'], 'C': [0.1, 1, 10], 'solver': ['lbfgs', 'sag', 'newton-cholesky']},
        {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10], 'solver': ['liblinear']},
        {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10], 'solver': ['saga']},
        {'penalty': ['elasticnet'], 'C': [0.1, 1, 10], 'solver': ['saga'], 'l1_ratio': [0.5, 0.7, 0.9]}
    ]

    search = RandomizedSearchCV(
        estimator=LogisticRegression(),
        param_distributions=param_grid,
        n_iter=10,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train_scaled, y_train)
    best_params = search.best_params_

    # Filter out irrelevant params (like l1_ratio for non-elasticnet)
    filtered_params = {
        k: v for k, v in best_params.items() if k in LogisticRegression().get_params()
    }

    best_model = LogisticRegression(**filtered_params)
    best_model.fit(X_train_scaled, y_train)

    y_pred = best_model.predict(X_val_scaled)
    accuracy = accuracy_score(y_val, y_pred)
    sharpe = calculate_sharpe_ratio(calculate_returns(y_pred, val_returns))

    return {
        'best_params': best_params,
        'accuracy': accuracy,
        'sharpe_ratio': sharpe,
        'scaler': scaler
    }, {
        'model': best_model,
        'scaler': scaler
    }