#!/usr/bin/env python3

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add the parent directory to sys.path to find data_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_loader import get_stocks, get_technical_indicators

# Set random seed for reproducibility
np.random.seed(42)

# Define the directory to save results relative to the project root
BASE_DIR = "/Users/derr/Documents/CS3244/Project/3244StockMarket"
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "XGBoost")

# Create the directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Define the horizon for this script
HORIZON = 252  # 1 year (252 trading days)

def prepare_data(df, horizon=HORIZON):
    """Prepare features and target for a given prediction horizon."""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'Hist',
                'RSI', 'K', 'D', 'J', 'OSC', 'BOLL_Mid', 'BOLL_Upper', 'BOLL_Lower', 'BIAS']
    df['Target'] = (df['Close'].shift(-horizon) > df['Close']).astype(int)
    df = df.dropna()
    X = df[features]
    y = df['Target']
    print(f"Unique target values in prepared data: {np.unique(y)}")  # Debug: Check unique values
    return X, y

def rolling_window_train_predict(df, window_size=730, horizon=HORIZON, step_size=30):
    """Train and predict using a rolling window strategy with a step size."""
    X, y = prepare_data(df, horizon)
    predictions = []
    actuals = []

    i = window_size
    while i < len(df) - horizon:
        train_data = df.iloc[i - window_size:i]
        test_end = min(i + step_size, len(df) - horizon)
        test_data = df.iloc[i:test_end]

        X_train = train_data.drop(columns=['Stock', 'Date', 'Target'])
        y_train = train_data['Target']
        X_test = test_data.drop(columns=['Stock', 'Date', 'Target'])

        # Check if y_train has at least two classes
        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {i} due to single class in y_train: {np.unique(y_train)}")
            i += step_size
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        predictions.extend(y_pred)
        actuals.extend(test_data['Target'].values)

        i += step_size

    return predictions, actuals

def expanding_window_train_predict(df, initial_window=730, horizon=HORIZON, step_size=30):
    """Train and predict using an expanding window strategy with a step size."""
    X, y = prepare_data(df, horizon)
    predictions = []
    actuals = []

    i = initial_window
    while i < len(df) - horizon:
        train_data = df.iloc[:i]
        test_end = min(i + step_size, len(df) - horizon)
        test_data = df.iloc[i:test_end]

        X_train = train_data.drop(columns=['Stock', 'Date', 'Target'])
        y_train = train_data['Target']
        X_test = test_data.drop(columns=['Stock', 'Date', 'Target'])

        # Check if y_train has at least two classes
        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {i} due to single class in y_train: {np.unique(y_train)}")
            i += step_size
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        predictions.extend(y_pred)
        actuals.extend(test_data['Target'].values)

        i += step_size

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

def save_results(horizon, strategy, y_true, y_pred, results_dir):
    """Save prediction results to a CSV file."""
    accuracy, f1, roc_auc = evaluate_model(y_true, y_pred)
    results_df = pd.DataFrame({
        'Horizon': [horizon],
        'Strategy': [strategy],
        'Accuracy': [accuracy],
        'F1-Score': [f1],
        'ROC-AUC': [roc_auc]
    })
    results_file = os.path.join(results_dir, f'results_horizon_{horizon}_{strategy}.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Saved results to {results_file}")
    return accuracy, f1, roc_auc

def main():
    # Load and preprocess data (focusing on one stock for simplicity, e.g., 'AAPL')
    df = get_stocks('AAPL')
    # df = df.tail(1000)  # Uncomment to test on a smaller dataset
    df = get_technical_indicators(df)

    print(f"\n=== Prediction Horizon: {HORIZON} Day(s) ===")

    # Rolling Window
    print("Rolling Window Strategy:")
    roll_preds, roll_actuals = rolling_window_train_predict(df, window_size=730, horizon=HORIZON, step_size=30)
    if roll_preds and roll_actuals:  # Check if predictions and actuals are non-empty
        roll_acc, roll_f1, roll_roc_auc = save_results(HORIZON, 'rolling', roll_actuals, roll_preds, RESULTS_DIR)
        print(f"Accuracy: {roll_acc:.4f}, F1-Score: {roll_f1:.4f}, ROC-AUC: {roll_roc_auc:.4f}")
    else:
        print("No valid predictions generated for Rolling Window Strategy.")

    # Expanding Window
    print("Expanding Window Strategy:")
    exp_preds, exp_actuals = expanding_window_train_predict(df, initial_window=730, horizon=HORIZON, step_size=30)
    if exp_preds and exp_actuals:  # Check if predictions and actuals are non-empty
        exp_acc, exp_f1, exp_roc_auc = save_results(HORIZON, 'expanding', exp_actuals, exp_preds, RESULTS_DIR)
        print(f"Accuracy: {exp_acc:.4f}, F1-Score: {exp_f1:.4f}, ROC-AUC: {exp_roc_auc:.4f}")
    else:
        print("No valid predictions generated for Expanding Window Strategy.")

    # Train final model for SHAP analysis (using full data up to last horizon days)
    X, y = prepare_data(df, HORIZON)
    if len(np.unique(y)) < 2:
        print(f"Warning: Only one class found in target variable: {np.unique(y)}. Skipping SHAP analysis.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            verbosity=0,
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1
        )
        model.fit(X_train_scaled, y_train)

        # SHAP Analysis
        shap_values = compute_shap_values(model, X_test_scaled)
        shap.summary_plot(shap_values, X_test, feature_names=X.columns)
        shap_plot_file = os.path.join(RESULTS_DIR, f'shap_summary_horizon_{HORIZON}.png')
        plt.savefig(shap_plot_file)
        plt.close()
        print(f"Saved SHAP plot to {shap_plot_file}")

if __name__ == "__main__":
    main()