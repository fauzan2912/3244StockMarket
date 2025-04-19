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

    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'solver': ['lbfgs', 'liblinear', 'newton-cholesky', 'sag', 'saga']
    }

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

    best_model = LogisticRegression(**best_params)
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
