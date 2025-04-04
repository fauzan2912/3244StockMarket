import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, mean_squared_error, mean_absolute_percentage_error, r2_score

from data_loader import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#finding stocks with the highest average trading volume
def top_liquid_stocks(n=10):
    df = get_stocks()  # Load all stock data
    avg_vol = df.groupby("Stock")["Volume"].mean()
    top_stocks = avg_vol.sort_values(ascending=False).head(n)
    return top_stocks

def data_prep(stock, horizon, threshold, k_features):
    df = get_stocks(stock)
    df = get_technical_indicators(df)
    df = df[df['Date'] >= '2009-01-01']
    df["Future_Close"] = df["Close"].shift(-horizon) # closing price after horizon days
    df["Price_Change"] = (df["Future_Close"] - df["Close"]) / df["Close"] 
    df["Target"] = (df["Price_Change"] > threshold).astype(int) # binary classification, did it reach the threshold?
    df["Lag1_Return"] = df["Close"].pct_change(1) # percentage change from previous day; helps measure momentum
    df["Lag5_Return"] = df["Close"].pct_change(5) # percentage change from 5 trading days ago (1 week); weekly trend
    df["Volatility"] = df["Close"].rolling(window=20).std() # standard deviation of closing price from previous 20 trading days (1 month); recent risk
    df.dropna(inplace=True)

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'MACD', 'Signal', 'Hist', 'RSI', 'K', 'D', 'J',
        'OSC', 'BOLL_Mid', 'BOLL_Upper', 'BOLL_Lower', 'BIAS',
        'Lag1_Return', 'Lag5_Return', 'Volatility'
    ]
    X = df[features]
    y = df['Target']
    y_reg = df['Future_Close']
    df = df.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_reg = y_reg.reset_index(drop=True)

    X_scaled = StandardScaler().fit_transform(X)
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_selected = selector.fit_transform(X_scaled, y)
    selected_features = X.columns[selector.get_support()]
    return df, X_selected, y, y_reg, selected_features

def plot_importance(X_selected, y, selected_features, stock):
    clf_final = RandomForestClassifier(n_estimators=150, max_depth=8, class_weight="balanced", random_state=42)
    clf_final.fit(X_selected, y)
    importances = clf_final.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": selected_features,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    # bar plot
    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df, palette="YlGnBu")
    plt.title(f"{stock} Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


#baseline models for comparison: linear regression
def train_baseline_linear(stock, horizon, k_features):
    df, X_selected, _, y_reg, selected_features = data_prep(stock, horizon, threshold=0.01, k_features=k_features)
    train_mask = df['Date'] < '2017-01-01'
    test_mask = df['Date'] >= '2017-01-01'

    X_train = X_selected[train_mask]
    y_train = y_reg[train_mask]
    X_test = X_selected[test_mask]
    y_test = y_reg[test_mask]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Baseline Linear Regression (Predicting Future Close) ===")
    print(f"RMSE: {np.sqrt(mse):.4f}, MAPE: {mape:.4f}, R²: {r2:.4f}")
    return model

#to further help with hyperparameter tuning -- to change later to improve accuracy
def tune_random_forest(X, y):
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=5,
        class_weight="balanced",
        random_state=42,
        oob_score=True  
    )
    model.fit(X, y)
    print("OOB Score (estimate of validation accuracy):", model.oob_score_) 
    return model

#random forest model (fixed window)
def train_randforest_FY(stock, horizon, threshold, k_features):
    df, X_selected, y, _, selected_features = prepare_data(stock, horizon, threshold, k_features)
    train_mask = df['Date'] < '2017-01-01'
    test_mask = df['Date'] >= '2017-01-01'
    X_train = X_selected[train_mask]
    y_train = y[train_mask]
    X_test = X_selected[test_mask]
    y_test = y[test_mask]

    model = tune_random_forest(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Fixed Training on 2009–2016, Test on 2017 ===")
    print(classification_report(y_test, y_pred))

#random forest model (rolling window)
def train_randforest_RW(stock, horizon, threshold, k_features, window_years=3):
    df, X_selected, y, _, selected_features = prepare_data(stock, horizon, threshold, k_features)
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    years = sorted(df["Year"].unique())
    print("\n=== Rolling Window by Year Report ===")

    for i in range(len(years) - window_years):
        train_years = years[i:i + window_years]
        test_year = years[i + window_years]
        train_mask = df["Year"].isin(train_years)
        test_mask = df["Year"] == test_year

        X_train = X_selected[train_mask]
        y_train = y[train_mask]
        X_test = X_selected[test_mask]
        y_test = y[test_mask]

        model = tune_random_forest(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n{train_years} -> {test_year}")
        print(classification_report(y_test, y_pred))

#random forest model (expanding window)
def train_randforest_EW(stock, horizon, threshold, k_features):
    df, X_selected, y, _, selected_features = prepare_data(stock, horizon, threshold, k_features)
    df["Year"] = pd.to_datetime(df["Date"]).dt.year
    years = sorted(df["Year"].unique())
    print("\n=== Expanding Window by Year Report ===")

    for i in range(1, len(years)):
        train_years = years[:i]
        test_year = years[i]
        train_mask = df["Year"].isin(train_years)
        test_mask = df["Year"] == test_year

        X_train = X_selected[train_mask]
        y_train = y[train_mask]
        X_test = X_selected[test_mask]
        y_test = y[test_mask]

        model = tune_random_forest(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n{train_years} -> {test_year}")
        print(classification_report(y_test, y_pred))



