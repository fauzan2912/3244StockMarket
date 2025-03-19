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
from scipy.optimize import minimize
from multiprocessing import Pool
import time

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

# Define stocks and horizons
STOCKS = ['AAPL', 'MSFT', 'GOOGL']
HORIZONS = [1, 3, 7, 30, 252]  # 1 day, 3 days, 1 week, 1 month, 1 year
THRESHOLD = 0.01  # 1% threshold for significant movement

def prepare_data(df, horizon=HORIZONS[-1], threshold=THRESHOLD):
    """Prepare features and target for a given prediction horizon with a threshold."""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'Hist',
                'RSI', 'K', 'D', 'J', 'OSC', 'BOLL_Mid', 'BOLL_Upper', 'BOLL_Lower', 'BIAS']
    df['Price_Change'] = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
    df['Target'] = ((df['Price_Change'] > threshold).astype(int) | (df['Price_Change'] < -threshold).astype(int))
    df = df.dropna()
    X = df[features]
    y = df['Target']
    print(f"Unique target values for {df['Stock'].iloc[0]} at horizon {horizon}: {np.unique(y)} (0 = no significant change, 1 = significant change)")
    return X, y, df['Price_Change']

def rolling_window_predict(df, window_size=730, horizon=HORIZONS[-1], step_size=30, threshold=THRESHOLD):
    """Train and predict using a rolling window strategy, returning future predictions."""
    X, y, price_changes = prepare_data(df, horizon, threshold)
    predictions = []
    actuals = []

    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size - horizon:]

    i = window_size
    while i < len(df_train) - horizon:
        train_data = df_train.iloc[i - window_size:i]
        test_data = df_train.iloc[i:i + step_size]

        X_train = train_data.drop(columns=['Stock', 'Date', 'Target', 'Price_Change'])
        y_train = train_data['Target']
        X_test = test_data.drop(columns=['Stock', 'Date', 'Target', 'Price_Change'])

        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {i} for {df['Stock'].iloc[0]} due to single class in y_train: {np.unique(y_train)}")
            i += step_size
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0, n_estimators=50, max_depth=3, learning_rate=0.1)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        predictions.extend(y_pred)
        actuals.extend(test_data['Target'].values)

        i += step_size

    # Predict on the "future" test set
    X_test_future = df_test.drop(columns=['Stock', 'Date', 'Target', 'Price_Change'])
    future_preds = []
    if len(X_test_future) > 0:
        X_test_future_scaled = scaler.transform(X_test_future)
        future_preds = model.predict(X_test_future_scaled)
        print(f"Future predictions for {df['Stock'].iloc[0]} at horizon {horizon} days (Rolling): {future_preds}")
    return predictions, actuals, future_preds, price_changes.iloc[train_size:]

def expanding_window_predict(df, initial_window=730, horizon=HORIZONS[-1], step_size=30, threshold=THRESHOLD):
    """Train and predict using an expanding window strategy, returning future predictions."""
    X, y, price_changes = prepare_data(df, horizon, threshold)
    predictions = []
    actuals = []

    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size - horizon:]

    i = initial_window
    while i < len(df_train) - horizon:
        train_data = df_train.iloc[:i]
        test_data = df_train.iloc[i:i + step_size]

        X_train = train_data.drop(columns=['Stock', 'Date', 'Target', 'Price_Change'])
        y_train = train_data['Target']
        X_test = test_data.drop(columns=['Stock', 'Date', 'Target', 'Price_Change'])

        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {i} for {df['Stock'].iloc[0]} due to single class in y_train: {np.unique(y_train)}")
            i += step_size
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0, n_estimators=50, max_depth=3, learning_rate=0.1)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        predictions.extend(y_pred)
        actuals.extend(test_data['Target'].values)

        i += step_size

    # Predict on the "future" test set
    X_test_future = df_test.drop(columns=['Stock', 'Date', 'Target', 'Price_Change'])
    future_preds = []
    if len(X_test_future) > 0:
        X_test_future_scaled = scaler.transform(X_test_future)
        future_preds = model.predict(X_test_future_scaled)
        print(f"Future predictions for {df['Stock'].iloc[0]} at horizon {horizon} days (Expanding): {future_preds}")
    return predictions, actuals, future_preds, price_changes.iloc[train_size:]

def compute_portfolio_variance(weights, variances, correlations):
    """Compute portfolio variance using the given formula."""
    n = len(weights)
    variance = 0
    for i in range(n):
        variance += weights[i]**2 * variances[i]
        for j in range(n):
            if i != j:
                variance += weights[i] * weights[j] * correlations[i][j] * np.sqrt(variances[i] * variances[j])
    return variance

def optimize_portfolio(expected_returns, variances, correlations, min_return=0.11):
    """Optimize portfolio weights to minimize variance subject to return constraint."""
    n = len(expected_returns)
    initial_weights = np.array([1.0 / n] * n)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: np.dot(expected_returns, w) - min_return}
    ]
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(
        lambda w: compute_portfolio_variance(w, variances, correlations),
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result.x if result.success else np.array([1.0 / n] * n)

def compute_expected_return(weights, returns):
    """Compute expected portfolio return."""
    return np.dot(weights, returns)

def evaluate_model(y_true, y_pred):
    """Calculate evaluation metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    return accuracy, f1, roc_auc

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

def process_horizon(horizon):
    """Process a single horizon for all stocks."""
    print(f"\n=== Prediction Horizon: {horizon} Day(s) ===")
    stock_predictions = {}
    stock_actuals = {}
    stock_future_preds = {}
    stock_price_changes = {}

    # Load and preprocess data for all stocks
    all_data = {}
    for stock in STOCKS:
        df = get_stocks(stock)
        df = get_technical_indicators(df)
        all_data[stock] = df

    # Process each stock
    for stock in STOCKS:
        print(f"Processing {stock}...")
        roll_preds, roll_actuals, future_roll_preds, price_changes = rolling_window_predict(all_data[stock], horizon=horizon)
        exp_preds, exp_actuals, future_exp_preds, _ = expanding_window_predict(all_data[stock], horizon=horizon)

        stock_predictions[stock] = {'rolling': roll_preds, 'expanding': exp_preds}
        stock_actuals[stock] = {'rolling': roll_actuals, 'expanding': exp_actuals}
        stock_future_preds[stock] = {'rolling': future_roll_preds, 'expanding': future_exp_preds}
        stock_price_changes[stock] = price_changes

        if roll_preds and roll_actuals:
            roll_acc, roll_f1, roll_roc_auc = save_results(horizon, f'{stock}_rolling', roll_actuals, roll_preds, RESULTS_DIR)
            print(f"{stock} Rolling - Accuracy: {roll_acc:.4f}, F1-Score: {roll_f1:.4f}, ROC-AUC: {roll_roc_auc:.4f}")
        else:
            print(f"No valid predictions generated for {stock} Rolling Window Strategy.")

        if exp_preds and exp_actuals:
            exp_acc, exp_f1, exp_roc_auc = save_results(horizon, f'{stock}_expanding', exp_actuals, exp_preds, RESULTS_DIR)
            print(f"{stock} Expanding - Accuracy: {exp_acc:.4f}, F1-Score: {exp_f1:.4f}, ROC-AUC: {exp_roc_auc:.4f}")
        else:
            print(f"No valid predictions generated for {stock} Expanding Window Strategy.")

    # Portfolio Optimization
    expected_returns = []
    variances = []
    correlations = np.zeros((len(STOCKS), len(STOCKS)))
    for i, stock in enumerate(STOCKS):
        future_preds = stock_future_preds[stock]['rolling']
        if future_preds:
            price_change = stock_price_changes[stock]
            expected_return = np.mean(price_change[future_preds == 1]) if any(future_preds == 1) else 0.0
            expected_returns.append(expected_return)
            variances.append(np.var(price_change))
        for j, other_stock in enumerate(STOCKS):
            if i != j:
                correlations[i][j] = all_data[stock]['Close'].pct_change().corr(all_data[other_stock]['Close'].pct_change())

    if expected_returns and variances:
        optimal_weights = optimize_portfolio(np.array(expected_returns), np.array(variances), correlations)
        portfolio_return = compute_expected_return(optimal_weights, np.array(expected_returns))
        portfolio_variance = compute_portfolio_variance(optimal_weights, np.array(variances), correlations)

        print(f"Optimal Weights for Horizon {horizon}: {dict(zip(STOCKS, optimal_weights))}")
        print(f"Expected Portfolio Return: {portfolio_return:.4f}")
        print(f"Portfolio Variance: {portfolio_variance:.4f}")

    return {
        'horizon': horizon,
        'predictions': stock_predictions,
        'actuals': stock_actuals,
        'future_preds': stock_future_preds,
        'weights': optimal_weights if 'optimal_weights' in locals() else None,
        'portfolio_return': portfolio_return if 'portfolio_return' in locals() else None,
        'portfolio_variance': portfolio_variance if 'portfolio_variance' in locals() else None
    }

def compute_shap_values(model, X):
    """Compute SHAP values for feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def main():
    start_time = time.time()

    # Process all horizons in parallel
    with Pool(processes=len(HORIZONS)) as pool:
        results = pool.map(process_horizon, HORIZONS)

    # Collect results
    all_results = {res['horizon']: res for res in results}

    # Summary of results
    print("\n=== Summary of Results ===")
    for horizon in HORIZONS:
        res = all_results[horizon]
        print(f"\nHorizon: {horizon} Day(s)")
        print(f"Optimal Weights: {res['weights']}")
        print(f"Expected Portfolio Return: {res['portfolio_return']:.4f}")
        print(f"Portfolio Variance: {res['portfolio_variance']:.4f}")
        for stock in STOCKS:
            roll_acc = evaluate_model(res['actuals'][stock]['rolling'], res['predictions'][stock]['rolling'])[0] if res['actuals'][stock]['rolling'] else "N/A"
            exp_acc = evaluate_model(res['actuals'][stock]['expanding'], res['predictions'][stock]['expanding'])[0] if res['actuals'][stock]['expanding'] else "N/A"
            print(f"{stock} Rolling Accuracy: {roll_acc}")
            print(f"{stock} Expanding Accuracy: {exp_acc}")

    # SHAP Analysis for the last horizon and first stock
    horizon = HORIZONS[-1]
    stock = STOCKS[0]
    df = get_stocks(stock)
    df = get_technical_indicators(df)
    X, y, _ = prepare_data(df, horizon)
    if len(np.unique(y)) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBClassifier(eval_metric='logloss', random_state=42, verbosity=0, n_estimators=50, max_depth=3, learning_rate=0.1)
        model.fit(X_train_scaled, y_train)

        shap_values = compute_shap_values(model, X_test_scaled)
        shap.summary_plot(shap_values, X_test, feature_names=X.columns)
        shap_plot_file = os.path.join(RESULTS_DIR, f'shap_summary_horizon_{horizon}_{stock}.png')
        plt.savefig(shap_plot_file)
        plt.close()
        print(f"Saved SHAP plot to {shap_plot_file}")
    else:
        print(f"Warning: Only one class found in target variable for {stock} at horizon {horizon}. Skipping SHAP analysis.")

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()