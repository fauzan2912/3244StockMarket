#!/usr/bin/env python3

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import os
import sys
from scipy.optimize import minimize
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
THRESHOLD = 0.01  # Initial 1% threshold for significant movement

def adjust_threshold(df, horizon, initial_threshold=THRESHOLD):
    """Dynamically adjust the threshold to balance target classes with stricter constraints."""
    df = df.copy()
    df['Price_Change'] = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
    threshold = initial_threshold
    step = 0.005
    max_attempts = 100
    attempt = 0

    while attempt < max_attempts:
        df.loc[:, 'Target'] = (df['Price_Change'] > threshold).astype(int)  # Only upward movements
        df = df.dropna()
        class_counts = df['Target'].value_counts()
        if len(class_counts) < 2:
            threshold += step
            attempt += 1
            continue
        class_ratio = class_counts[1] / (class_counts[0] + class_counts[1])
        # Stricter balance: aim for 40% to 60% for minority class
        if 0.40 <= class_ratio <= 0.60:
            break
        if class_ratio < 0.40:
            threshold -= step
        else:
            threshold += step
        attempt += 1
    if attempt == max_attempts:
        print(f"Warning: Could not balance classes for horizon {horizon}. Final class ratio: {class_ratio:.2f}")
        # If balancing fails, set a default threshold to ensure both classes exist
        if class_ratio == 0:
            threshold = df['Price_Change'].quantile(0.4)  # Ensure at least 40% are class 1
        elif class_ratio == 1:
            threshold = df['Price_Change'].quantile(0.6)  # Ensure at least 40% are class 0
        df.loc[:, 'Target'] = (df['Price_Change'] > threshold).astype(int)
    return df, threshold

def prepare_data(df, horizon=HORIZONS[-1], threshold=THRESHOLD):
    """Prepare features and target for a given prediction horizon with a threshold."""
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'Hist',
                'RSI', 'K', 'D', 'J', 'OSC', 'BOLL_Mid', 'BOLL_Upper', 'BOLL_Lower', 'BIAS']
    # Add lagged returns and volatility
    df['Lag1_Return'] = df['Close'].pct_change(1)
    df['Lag5_Return'] = df['Close'].pct_change(5)
    df['Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
    features.extend(['Lag1_Return', 'Lag5_Return', 'Volatility_20'])
    df, adjusted_threshold = adjust_threshold(df, horizon, threshold)
    df = df.dropna(subset=features + ['Target'])
    print(f"Adjusted threshold for {df['Stock'].iloc[0]} at horizon {horizon}: {adjusted_threshold:.4f}")
    print(f"Unique target values for {df['Stock'].iloc[0]} at horizon {horizon}: {np.unique(df['Target'])}")
    return df

def get_xgboost_model(X_train, y_train):
    """Return an XGBoost model with tuned parameters and early stopping."""
    class_counts = np.bincount(y_train)
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 and class_counts[1] > 0 else 1.0
    model = XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=10
    )
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_split, y_train_split, eval_set=[(X_val, y_val)], verbose=False)
    return grid_search.best_estimator_

def process_stock(stock, horizon, window_size=None, step_size=None, threshold=THRESHOLD):
    """Process a single stock for a given horizon using rolling window strategy with cross-validation."""
    if window_size is None:
        window_size = 90 if horizon <= 7 else 365 if horizon <= 30 else 730
    if step_size is None:
        step_size = 30 if horizon <= 7 else 60
    print(f"Processing {stock} for horizon {horizon} with window_size {window_size} and step_size {step_size}...")
    df = get_stocks(stock)
    df = get_technical_indicators(df)
    df = prepare_data(df, horizon, threshold)
    predictions = []
    actuals = []

    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size - horizon:]

    i = window_size
    scaler = StandardScaler()
    selector = SelectKBest(score_func=f_classif, k=10)

    columns_to_drop = ['Stock', 'Date', 'Target', 'Price_Change']
    X_train_full = df_train.drop(columns=columns_to_drop, errors='ignore')
    y_train_full = df_train['Target']
    combined_train = pd.concat([X_train_full, y_train_full], axis=1).dropna()
    X_train_full = combined_train.drop(columns=['Target'])
    y_train_full = combined_train['Target']
    X_train_full_scaled = scaler.fit_transform(X_train_full)
    X_train_full_selected = selector.fit_transform(X_train_full_scaled, y_train_full)

    model = None
    tscv = TimeSeriesSplit(n_splits=5)
    while i < len(df_train) - horizon:
        train_data = df_train.iloc[i - window_size:i]
        test_data = df_train.iloc[i:i + step_size]

        X_train = train_data.drop(columns=columns_to_drop, errors='ignore')
        y_train = train_data['Target']
        X_test = test_data.drop(columns=columns_to_drop, errors='ignore')

        combined_train = pd.concat([X_train, y_train], axis=1).dropna()
        if len(combined_train) == 0:
            print(f"Skipping window at index {i} for {stock} due to all NaNs in training data.")
            i += step_size
            continue
        X_train = combined_train.drop(columns=['Target'])
        y_train = combined_train['Target']

        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {i} for {stock} due to single class in y_train: {np.unique(y_train)}")
            if model is None:
                print(f"No model available to make predictions at window {i}. Skipping predictions.")
                i += step_size
                continue
            # Use the existing model to predict without retraining
            X_test_scaled = scaler.transform(X_test)
            X_test_selected = selector.transform(X_test_scaled)
            y_pred = model.predict(X_test_selected)
            predictions.extend(y_pred)
            actuals.extend(test_data['Target'].values)
            i += step_size
            continue

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_selected = selector.transform(X_train_scaled)
        X_test_selected = selector.transform(X_test_scaled)

        if model is None:
            model = get_xgboost_model(X_train_selected, y_train)
        else:
            # Use cross-validation to retrain the model
            scores = []
            for train_idx, val_idx in tscv.split(X_train_selected):
                X_train_cv, X_val_cv = X_train_selected[train_idx], X_train_selected[val_idx]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                if len(X_val_cv) == 0 or len(y_val_cv) == 0:
                    print(f"Skipping CV fold at index {i} for {stock} due to empty validation set.")
                    continue
                if len(np.unique(y_train_cv)) < 2:
                    print(f"Skipping CV fold at index {i} for {stock} due to single class in y_train_cv: {np.unique(y_train_cv)}")
                    continue
                # Fit with validation set for early stopping
                model.fit(X_train_cv, y_train_cv, eval_set=[(X_val_cv, y_val_cv)], verbose=False)
                y_pred_cv = model.predict(X_val_cv)
                scores.append(accuracy_score(y_val_cv, y_pred_cv))
            if scores:
                print(f"Cross-validation accuracy for {stock} at horizon {horizon}, window {i}: {np.mean(scores):.4f}")
                # Final fit on the full training data for this window
                model.fit(X_train_selected, y_train, eval_set=[(X_train_selected, y_train)], verbose=False)
            else:
                print(f"Skipping window at index {i} for {stock} due to no valid CV folds.")
                if model is None:
                    print(f"No model available to make predictions at window {i}. Skipping predictions.")
                    i += step_size
                    continue
                # Use the existing model to predict without retraining
                y_pred = model.predict(X_test_selected)
                predictions.extend(y_pred)
                actuals.extend(test_data['Target'].values)
                i += step_size
                continue

        y_pred = model.predict(X_test_selected)
        predictions.extend(y_pred)
        actuals.extend(test_data['Target'].values)

        i += step_size

    X_test_future = df_test.drop(columns=columns_to_drop, errors='ignore')
    future_preds = []
    if len(X_test_future) > 0:
        X_test_future = X_test_future.dropna()
        if len(X_test_future) > 0:
            X_test_future_scaled = scaler.transform(X_test_future)
            X_test_future_selected = selector.transform(X_test_future_scaled)
            future_preds = model.predict(X_test_future_selected) if model else np.zeros(len(X_test_future), dtype=int)
            print(f"Future predictions for {stock} at horizon {horizon} days: {future_preds}")

    return stock, predictions, actuals, future_preds, df_test['Price_Change'], df

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

def optimize_portfolio(expected_returns, variances, correlations, min_return=0.11, risk_free_rate=0.01):
    """Optimize portfolio weights to maximize Sharpe ratio subject to return constraint."""
    n = len(expected_returns)
    initial_weights = np.array([1.0 / n] * n)

    def objective(weights):
        portfolio_return = compute_expected_return(weights, expected_returns)
        portfolio_variance = compute_portfolio_variance(weights, variances, correlations)
        sharpe_ratio = (portfolio_return - risk_free_rate) / np.sqrt(portfolio_variance) if portfolio_variance > 0 else 0
        return -sharpe_ratio

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: np.dot(expected_returns, w) - min_return}
    ]
    bounds = [(0, 1) for _ in range(n)]

    result = minimize(
        objective,
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
    """Process a single horizon for all stocks sequentially and return combined data for SHAP."""
    print(f"\n=== Prediction Horizon: {horizon} Day(s) ===")
    stock_predictions = {}
    stock_actuals = {}
    stock_future_preds = {}
    stock_price_changes = {}
    all_data = {}

    # Process stocks sequentially
    results = [process_stock(stock, horizon) for stock in STOCKS]

    # Collect results and combine data
    combined_data = []
    for stock, roll_preds, roll_actuals, future_roll_preds, price_changes, df in results:
        stock_predictions[stock] = {'rolling': roll_preds}
        stock_actuals[stock] = {'rolling': roll_actuals}
        stock_future_preds[stock] = {'rolling': future_roll_preds}
        stock_price_changes[stock] = price_changes
        all_data[stock] = df
        combined_data.append(df)

        if roll_preds and roll_actuals:
            roll_acc, roll_f1, roll_roc_auc = save_results(horizon, f'{stock}_rolling', roll_actuals, roll_preds, RESULTS_DIR)
            # Baseline: Predict majority class
            majority_class = 1 if np.mean(roll_actuals) > 0.5 else 0
            baseline_preds = [majority_class] * len(roll_actuals)
            baseline_acc = accuracy_score(roll_actuals, baseline_preds)
            print(f"{stock} Rolling - Accuracy: {roll_acc:.4f}, F1-Score: {roll_f1:.4f}, ROC-AUC: {roll_roc_auc:.4f}, Baseline Accuracy: {baseline_acc:.4f}")
        else:
            print(f"No valid predictions generated for {stock} Rolling Window Strategy.")

    # Combine data for SHAP
    combined_df = pd.concat(combined_data, ignore_index=True)

    # Portfolio Optimization
    expected_returns = []
    variances = []
    correlations = np.zeros((len(STOCKS), len(STOCKS)))
    active_stocks = []
    columns_to_drop = ['Stock', 'Date', 'Target', 'Price_Change']
    scaler = StandardScaler()
    selector = SelectKBest(score_func=f_classif, k=10)
    model = None

    for stock in STOCKS:
        X_train_full = all_data[stock].drop(columns=columns_to_drop, errors='ignore')
        y_train_full = all_data[stock]['Target']
        combined_train = pd.concat([X_train_full, y_train_full], axis=1).dropna()
        X_train_full = combined_train.drop(columns=['Target'])
        y_train_full = combined_train['Target']
        X_train_full_scaled = scaler.fit_transform(X_train_full)
        X_train_full_selected = selector.fit_transform(X_train_full_scaled, y_train_full)
        if model is None:
            model = get_xgboost_model(X_train_full_selected, y_train_full)
        break

    for i, stock in enumerate(STOCKS):
        future_preds = stock_future_preds[stock]['rolling']
        if len(future_preds) > 0:
            price_change = stock_price_changes[stock]
            if len(np.unique(future_preds)) > 1:
                X_test_future = all_data[stock].iloc[-len(future_preds):].drop(columns=columns_to_drop, errors='ignore')
                X_test_future_scaled = scaler.transform(X_test_future)
                X_test_future_selected = selector.transform(X_test_future_scaled)
                probs = model.predict_proba(X_test_future_selected)[:, 1]
                expected_return = np.sum(price_change * probs) / np.sum(probs) if np.sum(probs) > 0 else 0.0
            else:
                print(f"Warning: Single-class predictions for {stock} at horizon {horizon}: {np.unique(future_preds)}. Using historical average return.")
                expected_return = np.mean(price_change)
            if expected_return > 0:
                expected_returns.append(expected_return)
                variance = np.var(price_change)
                variances.append(variance if not np.isnan(variance) and variance > 0 else 1e-6)
                active_stocks.append(stock)
            else:
                print(f"Excluding {stock} from portfolio at horizon {horizon} due to negative expected return: {expected_return:.4f}")

    optimal_weights = None
    portfolio_return = None
    portfolio_variance = None
    if len(active_stocks) > 0:
        correlations = np.zeros((len(active_stocks), len(active_stocks)))
        for i, stock in enumerate(active_stocks):
            for j, other_stock in enumerate(active_stocks):
                if i != j:
                    corr = all_data[stock]['Close'].pct_change().corr(all_data[other_stock]['Close'].pct_change())
                    correlations[i][j] = corr if not np.isnan(corr) else 0.0

        min_return = 0.001 if horizon <= 7 else 0.01 if horizon <= 30 else 0.11
        optimal_weights = optimize_portfolio(np.array(expected_returns), np.array(variances), correlations, min_return=min_return)
        portfolio_return = compute_expected_return(optimal_weights, np.array(expected_returns))
        portfolio_variance = compute_portfolio_variance(optimal_weights, np.array(variances), correlations)

        full_weights = np.zeros(len(STOCKS))
        for idx, stock in enumerate(active_stocks):
            full_weights[STOCKS.index(stock)] = optimal_weights[idx]

        print(f"Optimal Weights for Horizon {horizon}: {dict(zip(STOCKS, full_weights))}")
        print(f"Expected Portfolio Return: {portfolio_return:.4f}")
        print(f"Portfolio Variance: {portfolio_variance:.4f}")
    else:
        print(f"Skipping portfolio optimization for horizon {horizon} due to no stocks with positive expected returns.")
        full_weights = np.array([1.0 / len(STOCKS)] * len(STOCKS))
        portfolio_return = None
        portfolio_variance = None

    return {
        'horizon': horizon,
        'predictions': stock_predictions,
        'actuals': stock_actuals,
        'future_preds': stock_future_preds,
        'weights': full_weights,
        'portfolio_return': portfolio_return,
        'portfolio_variance': portfolio_variance,
        'combined_data': combined_df
    }

def compute_shap_values(model, X):
    """Compute SHAP values for feature importance."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values

def main():
    start_time = time.time()

    # Process all horizons sequentially
    results = []
    for horizon in HORIZONS:
        result = process_horizon(horizon)
        results.append(result)

    # Collect results
    all_results = {res['horizon']: res for res in results}

    # Summary of results
    print("\n=== Summary of Results ===")
    for horizon in HORIZONS:
        res = all_results[horizon]
        print(f"\nHorizon: {horizon} Day(s)")
        print(f"Optimal Weights: {res['weights']}")
        if res['portfolio_return'] is not None:
            print(f"Expected Portfolio Return: {res['portfolio_return']:.4f}")
        else:
            print("Expected Portfolio Return: N/A (Portfolio optimization skipped)")
        if res['portfolio_variance'] is not None and not np.isnan(res['portfolio_variance']):
            print(f"Portfolio Variance: {res['portfolio_variance']:.4f}")
        else:
            print("Portfolio Variance: N/A (Portfolio optimization skipped or invalid variance)")
        for stock in STOCKS:
            roll_acc = evaluate_model(res['actuals'][stock]['rolling'], res['predictions'][stock]['rolling'])[0] if res['actuals'][stock]['rolling'] else "N/A"
            print(f"{stock} Rolling Accuracy: {roll_acc}")

    # SHAP Analysis for combined data at each horizon
    for horizon in HORIZONS:
        print(f"\nComputing SHAP values for all stocks at horizon {horizon}...")
        combined_df = all_results[horizon]['combined_data']
        if len(np.unique(combined_df['Target'])) >= 2:
            X = combined_df.drop(columns=['Stock', 'Date', 'Target', 'Price_Change'], errors='ignore')
            y = combined_df['Target']
            # Split data into training and validation sets for early stopping
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            selector = SelectKBest(score_func=f_classif, k=10)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)
            X_test_selected = selector.transform(X_test_scaled)

            # Subset the test data for SHAP analysis
            X_test_selected = X_test_selected[:500]
            y_test_subset = y_test[:500]  # Subset y_test to match X_test_selected
            X_test_subset = X_test.iloc[:500, selector.get_support()]

            model = get_xgboost_model(X_train_selected, y_train)
            # Fit the model with a validation set for early stopping
            model.fit(X_train_selected, y_train, eval_set=[(X_test_selected, y_test_subset)], verbose=False)

            shap_values = compute_shap_values(model, X_test_selected)
            shap.summary_plot(shap_values, X_test_subset, feature_names=X.columns[selector.get_support()], show=False)
            shap_plot_file = os.path.join(RESULTS_DIR, f'shap_summary_horizon_{horizon}_all_stocks.png')
            plt.savefig(shap_plot_file)
            plt.close()
            print(f"Saved SHAP plot to {shap_plot_file}")
        else:
            print(f"Warning: Only one class found in target variable for all stocks at horizon {horizon}. Skipping SHAP analysis.")

    end_time = time.time()
    print(f"\nTotal runtime: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()