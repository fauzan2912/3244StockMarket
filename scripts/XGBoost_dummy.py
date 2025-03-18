#!/usr/bin/env python3

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Assuming your friend's code is in a module named `data_processing.py`
from data_loader import get_stocks, get_technical_indicators

# Set random seed for reproducibility
np.random.seed(42)

def prepare_data(df, horizon=1):
    """Prepare features and target for a given prediction horizon."""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'Hist',
                'RSI', 'K', 'D', 'J', 'OSC', 'BOLL_Mid', 'BOLL_Upper', 'BOLL_Lower', 'BIAS']
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    df = df.dropna()  # Drop rows with NaN values due to shifting or indicators
    X = df[features]
    y = df['Target']
    return X, y

def rolling_window_train_predict(df, window_size=730, horizon=1):
    """Train and predict using a rolling window strategy."""
    X, y = prepare_data(df, horizon)
    predictions = []
    actuals = []

    for i in range(window_size, len(df) - horizon):
        train_data = df.iloc[i - window_size:i]
        test_data = df.iloc[i:i + horizon]

        X_train = train_data.drop(columns=['Stock', 'Date', 'Target'])
        y_train = train_data['Target']
        X_test = test_data.drop(columns=['Stock', 'Date', 'Target'])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        predictions.extend(y_pred)
        actuals.extend(test_data['Target'].values)

    return predictions, actuals

def expanding_window_train_predict(df, initial_window=730, horizon=1):
    """Train and predict using an expanding window strategy."""
    X, y = prepare_data(df, horizon)
    predictions = []
    actuals = []

    for i in range(initial_window, len(df) - horizon):
        train_data = df.iloc[:i]
        test_data = df.iloc[i:i + horizon]

        X_train = train_data.drop(columns=['Stock', 'Date', 'Target'])
        y_train = train_data['Target']
        X_test = test_data.drop(columns=['Stock', 'Date', 'Target'])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        predictions.extend(y_pred)
        actuals.extend(test_data['Target'].values)

    return predictions, actuals

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, f1, roc_auc

def compute_shap_values(model, X):
    """Compute SHAP values for feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def main():
    # Load and preprocess data (focusing on one stock for simplicity, e.g., 'AAPL')
    df = get_stocks('AAPL')
    df = get_technical_indicators(df)

    horizons = [1, 3, 7, 30, 90]  # 1 day, 3 days, 1 week, 1 month, 3 months

    for horizon in horizons:
        print(f"\n=== Prediction Horizon: {horizon} Day(s) ===")

        # Rolling Window
        print("Rolling Window Strategy:")
        roll_preds, roll_actuals = rolling_window_train_predict(df, window_size=730, horizon=horizon)
        roll_acc, roll_f1, roll_roc_auc = evaluate_model(roll_actuals, roll_preds)
        print(f"Accuracy: {roll_acc:.4f}, F1-Score: {roll_f1:.4f}, ROC-AUC: {roll_roc_auc:.4f}")

        # Expanding Window
        print("Expanding Window Strategy:")
        exp_preds, exp_actuals = expanding_window_train_predict(df, initial_window=730, horizon=horizon)
        exp_acc, exp_f1, exp_roc_auc = evaluate_model(exp_actuals, exp_preds)
        print(f"Accuracy: {exp_acc:.4f}, F1-Score: {exp_f1:.4f}, ROC-AUC: {exp_roc_auc:.4f}")

        # Train final model for SHAP analysis (using full data up to last horizon days)
        X, y = prepare_data(df, horizon)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train_scaled, y_train)

        # SHAP Analysis
        shap_values = compute_shap_values(model, X_test_scaled)
        shap.summary_plot(shap_values, X_test, feature_names=X.columns)
        plt.savefig(f'shap_summary_horizon_{horizon}.png')
        plt.close()

if __name__ == "__main__":
    main()