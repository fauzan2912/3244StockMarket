#!/usr/bin/env python3
import os
import gc
import sys
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import calculate_sharpe_ratio, calculate_returns
from scikeras.wrappers import KerasClassifier as _KerasClassifier
from tensorflow.keras import backend as K

# Custom wrapper to mark the estimator as a classifier
class FixedKerasClassifier(_KerasClassifier):
    _estimator_type = "classifier"

def create_sequences(X, y, seq_length):
    """
    Create temporal sequences of length `seq_length`.
    For each sample i (starting from seq_length), the sequence consists of X[i-seq_length:i],
    and the corresponding target is y[i].
    """
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i-seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def tune_lstm(X_train, y_train, X_val, y_val, val_returns, seq_length=10):
    print("\n--- Tuning LSTM Hyperparameters with Temporal Sequences ---")

    # If the inputs are 2D, create temporal sequences using a sliding window.
    # Otherwise, we assume the data is already sequential (3D)
    if len(X_train.shape) == 2:
        # Scale data first (scaling on 2D is easier)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = StandardScaler().fit_transform(X_val)  # Alternatively, use same scaler if appropriate
        # Create sequences using sliding window
        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, seq_length)
        X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val, seq_length)
    else:
        # Assume inputs already have temporal information
        X_train_seq, y_train_seq = X_train, y_train
        X_val_seq, y_val_seq = X_val, y_val
        scaler = None

    n_samples, timesteps, n_features = X_train_seq.shape
    print(f"--- Data transformed: {n_samples} training sequences of length {timesteps} with {n_features} features.")

    # For consistency, scale the sequences by reshaping to 2D, applying scaler, then back to 3D
    # (Only needed if not done already; here we assume scaling was done before sequence creation)
    # Uncomment below if you need additional scaling:
    # X_train_seq = scaler.fit_transform(X_train_seq.reshape(-1, n_features)).reshape(X_train_seq.shape)
    # X_val_seq = scaler.transform(X_val_seq.reshape(-1, n_features)).reshape(X_val_seq.shape)

    # Enable GPU memory growth if using GPU
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Model builder using Functional API to avoid input_shape warnings.
    def create_model(lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
        from tensorflow.keras.optimizers import Adam
        inputs = Input(shape=(timesteps, n_features))
        x = LSTM(lstm_units)(inputs)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
        return model

    clf = FixedKerasClassifier(model=create_model, verbose=0)

    # Note the use of the `model__` prefix so scikeras can pass these parameters correctly.
    param_grid = {
        'model__lstm_units': [50, 100],
        'model__dropout_rate': [0.2, 0.5],
        'model__learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32],
        'epochs': [10, 20]
    }

    gc.collect()
    K.clear_session()

    search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        n_iter=5,
        scoring='accuracy',
        cv=3,
        n_jobs=1,  # Sequential execution to save GPU memory
        verbose=1,
        random_state=42
    )

    search.fit(X_train_seq, y_train_seq)
    best_params = search.best_params_
    print("--- Best Parameters:", best_params)

    # Remove the 'model__' prefix for model creation.
    model_params = {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}
    best_model = create_model(**model_params)

    gc.collect()
    K.clear_session()

    best_model.fit(
        X_train_seq,
        y_train_seq,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=0
    )

    y_pred = (best_model.predict(X_val_seq) > 0.5).astype("int32")
    accuracy = accuracy_score(y_val_seq, y_pred)
    # Convert predictions to trading signals: e.g., 0 -> -1 (sell/short), 1 -> +1 (buy)
    signals = np.where(y_pred == 0, -1, 1)
    sharpe = calculate_sharpe_ratio(calculate_returns(signals, val_returns))
    print("--- Accuracy:", accuracy)

    results = {
        'best_params': best_params,
        'accuracy': accuracy,
        'sharpe_ratio': sharpe,
        'cv_results': {
            'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
            'params': [str(p) for p in search.cv_results_['params']]
        }
    }

    # Return tuned parameters including the input shape needed by the model.
    tuned_params = best_params.copy()
    tuned_params['input_shape'] = (timesteps, n_features)
    print("Tuned parameters returned:", tuned_params)
    
    return results, {'model': best_model, 'scaler': scaler, 'tuned_params': tuned_params}
    
# Example usage:
if __name__ == "__main__":
    # Dummy data for testing purposes; replace these with your actual data.
    X_train_dummy = np.random.rand(200, 20)  # 200 samples, 20 features
    y_train_dummy = np.random.randint(0, 2, 200)
    X_val_dummy = np.random.rand(50, 20)
    y_val_dummy = np.random.randint(0, 2, 50)
    val_returns_dummy = np.random.rand(50) * 0.02  # dummy returns

    results, tuned = tune_lstm(X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy, val_returns_dummy, seq_length=10)
    print("\nFinal tuning results:")
    print(results)
