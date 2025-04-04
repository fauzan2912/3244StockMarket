# import lib
import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc
from datetime import datetime

# set up the directory
sys.path.append("../../")

# get the customised data_loader
from data_loader import get_stocks, get_etfs, get_technical_indicators

# define global constant
STOCKS = ['AAPL', 'MSFT', 'GOOGL']
HORIZONS = [1, 3, 7, 30, 90] # 1 day, 3 days, 1 week, 1 month, 3 months
THRESHOLD = 0.01 # initial threshold

# from Darren's XGBoost
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

def prepare_data(stock, seq_length=720, horizon=1, threshold=0.01):
    # load the corresponding stock
    df = get_stocks(stock)
    df = get_technical_indicators(df)
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MACD', 'Signal', 'Hist',
                'RSI', 'K', 'D', 'J', 'OSC', 'BOLL_Mid', 'BOLL_Upper', 'BOLL_Lower', 'BIAS']


    # set up the data
    df, adjusted_threshold = adjust_threshold(df, horizon, threshold)
    # drop the irrevalent variables for training set
    X = df[features]
    y = df['Target'].values

    # normalise the vector
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    return X, y, scaler

def get_sequences(X, y, seq_length):
    """Create properly aligned sequences where:
    - Each X sequence contains seq_length past observations
    - Each y sequence contains corresponding targets
    - Both have exactly the same length"""
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])  # Past seq_length observations
            y_seq.append(y[i])   # Corresponding targets
    return np.array(X_seq), np.array(y_seq)

# build lstm model
def get_lstm(seq_len, num_param):
    model = Sequential([
        Input(shape=(seq_len, num_param)),
        LSTM(50, return_sequences=True),
        LSTM(25),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def get_enhanced_lstm(seq_len, num_features):
    model = Sequential([
        Input(shape=(seq_len, num_features)),
        LSTM(128, return_sequences=True,
             kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
             recurrent_dropout=0.2),
        BatchNormalization(),
        LSTM(64, return_sequences=True,
             kernel_regularizer=L1L2(l1=1e-5, l2=1e-4)),
        Dropout(0.3),
        LSTM(32),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    optimizer = Adam(learning_rate=0.001)
    # model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def evaluate_classification(y_true, y_pred_probs, threshold=0.5):
    """
    Compute comprehensive classification metrics
    Args:
        y_true: True binary labels (0 or 1)
        y_pred_probs: Predicted probabilities (0 to 1)
        threshold: Cutoff for binary classification
    Returns:
        Dictionary of metrics and plots
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_probs >= threshold).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_probs),
    }

    return metrics

def save_results(stock, model, horizon, window_type, window_size, sample_size, metrics, filepath='../../../Results/LSTM/full_lstm_results.csv'):
    """
    Save experiment results with metadata to CSV

    Args:
        stock: Stock ticker (e.g., 'AAPL')
        horizon: Prediction horizon in days (e.g., 5)
        window_type: 'rolling' or 'expanding'
        metrics: Dictionary of evaluation metrics
        params: Dictionary of model parameters
        filepath: Output CSV path
    """
    # Create results dictionary with metadata
    result_record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'stock': stock,
        'model': model,
        'horizon': horizon,
        'window_type': window_type,
        'window_size': window_size,
        'sample_size': sample_size,
        **metrics,  # Unpack metrics dictionary
    }

    # Convert to DataFrame
    results_df = pd.DataFrame([result_record])

    try:
        existing_df = pd.read_csv(filepath)
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        updated_df.to_csv(filepath, index=False)
    except FileNotFoundError:
        results_df.to_csv(filepath, index=False)

# train the model by rolling window and predict the outcome
def rolling_window_train(stock, horizon=1, train_ratio=0.8, window_size=None, epochs=5, batch_size=32, step_size=None):
    # set up the window_size & step_size
    if window_size is None:
        window_size = 90 if horizon <= 7 else 365 if horizon <= 30 else 730
    if step_size is None:
        step_size = 30 if horizon <= 7 else 60

    predictions = []
    actuals = []
    probability_predictions = []

    # prepare data for the seq_length
    X, y, scaler = prepare_data(stock, seq_length=window_size, horizon=horizon)
    test_start = int(len(X)*train_ratio)

    # get sequence data
    X, y = get_sequences(X, y, window_size)

    i = test_start
    while (i < len(X)):
        i_seq = i-window_size
        # reset the model
        model = get_enhanced_lstm(window_size, X.shape[2])

        # set up the variables for training
        X_window, y_window = X[i_seq-window_size:i_seq], y[i_seq-window_size:i_seq]

        # fit the model with rolling window
        model.fit(X_window,
                  y_window,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.2,
                  verbose=0,
                  callbacks=[
                      EarlyStopping(
                        monitor='val_loss',
                        patience=10,  # Longer for noisy financial data
                        min_delta=0.001,  # Minimum change to qualify as improvement
                        restore_best_weights=True  # Keeps the best model found
                        )
                  ])

        # Predict next value
        pred = model.predict(X[i_seq:i_seq+1], verbose=0)
        binary_pred = 1 if pred > 0.5 else 0

        predictions.append(binary_pred)
        actuals.append(y[i_seq])
        probability_predictions.append(pred)

        i += step_size

    return np.array(predictions).flatten(), np.array(actuals).flatten(), np.array(probability_predictions).flatten()

def expanding_window_train(stock, horizon, train_ratio=0.8, window_size=None, epochs=5, batch_size=32, step_size=30):
    # set up the window_size & step_size
    if window_size is None:
        window_size = 90 if horizon <= 7 else 365 if horizon <= 30 else 730
    if step_size is None:
        step_size = 30 if horizon <= 7 else 60

    predictions = []
    actuals = []
    probability_predictions = []
    X, y, scaler = prepare_data(stock, seq_length=window_size, horizon=horizon)
    test_start = int(len(X)*train_ratio)

    i = test_start
    while (i < len(X)):
        print(f"i: {i}")
        i_seq = i-window_size
        X_seq, y_seq = get_sequences(X, y, window_size)
        # reset the model
        model = get_lstm(window_size, X_seq.shape[2])

        # set up the variables for training
        X_window, y_window = X_seq[i_seq-window_size:i_seq], y_seq[i_seq-window_size:i_seq]

        # fit the model with rolling window
        model.fit(X_window,
                  y_window,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_split=0.2,
                  verbose=0,
                  callbacks=[
                      EarlyStopping(
                        monitor='val_loss',
                        patience=10,  # Longer for noisy financial data
                        min_delta=0.001,  # Minimum change to qualify as improvement
                        restore_best_weights=True  # Keeps the best model found
                      )
                  ])

        # Predict next value
        pred = model.predict(X_seq[i_seq:i_seq+1], verbose=0)
        binary_pred = 1 if pred > 0.5 else 0

        predictions.append(binary_pred)
        actuals.append(y_seq[i_seq])
        probability_predictions.append(pred)

        i += step_size
        window_size += step_size

    return np.array(predictions).flatten(), np.array(actuals).flatten(), np.array(probability_predictions).flatten()

def main():
    for i in range(len(STOCKS)):
      for j in range(len(HORIZONS)):
        # rolling window
        print(f"training & predicting stock: {STOCKS[i]} with horizon: {HORIZONS[i]} by rolling window\n")
        roll_predictions, roll_actuals, roll_probability_predictions = rolling_window_train(stock=STOCKS[i], horizon=HORIZONS[j])

        roll_matrics = evaluate_classification(roll_actuals, roll_probability_predictions, threshold=0.5) # evaulate the model performance
        save_results(STOCKS[i], "enhanced", HORIZONS[j], "roll", 730, "full", roll_matrics) # save results to a file
        print(f"roll_matrics:\n{roll_matrics}\n")

        # expanding window
        print(f"training & predicting stock: {STOCKS[i]} with horizon: {HORIZONS[j]} by expanding window\n")
        expand_predictions, expand_actuals, expand_probability_predictions = expanding_window_train(STOCKS[i], HORIZONS[j])
        expand_matrics = evaluate_classification(expand_actuals, expand_probability_predictions, threshold=0.5) # evaulate the model performance
        save_results(STOCKS[i], "simple", HORIZONS[j], "expand", 730, "full", expand_matrics) # save results to a file
        print(f"expand_matrics:\n{expand_matrics}")

if __name__ == "__main__":
  main()