#!/usr/bin/env python3
# src/tuning/attention_lstm.py

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from evaluation.metrics import calculate_sharpe_ratio, calculate_returns
from scikeras.wrappers import KerasClassifier as _KerasClassifier
import tensorflow as tf
from keras import backend as K
import gc

# Custom wrapper to mark the estimator as a classifier
class FixedKerasClassifier(_KerasClassifier):
    _estimator_type = "classifier"

def create_sequences(X, y, seq_length):
    """
    Create temporal sequences of length `seq_length`.
    For each sample i (starting from seq_length), the sequence consists of X[i-seq_length:i]
    and the corresponding target is y[i].
    """
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
        X_seq.append(X[i - seq_length:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def tune_attention_lstm(X_train, y_train, X_val, y_val, val_returns, seq_length=10):
    print("\n--- Tuning Attention LSTM Hyperparameters ---")

    # If inputs are 2D, convert them to sequences using a sliding window.
    if len(X_train.shape) == 2:
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
    else:
        X_train_seq, y_train_seq = X_train, y_train

    if len(X_val.shape) == 2:
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, seq_length)
    else:
        X_val_seq, y_val_seq = X_val, y_val

    n_samples, timesteps, n_features = X_train_seq.shape
    print(f"Data transformed: {n_samples} training sequences of length {timesteps} with {n_features} features.")

    # Optionally adjust validation returns to align with the sequence targets.
    # For example, if you use a sliding window, the first 'seq_length' samples are lost.
    if len(val_returns) == X_val.shape[0]:
        val_returns_seq = val_returns[seq_length:]
    else:
        val_returns_seq = val_returns

    # Scale each sequence by reshaping into 2D, scaling, then reshaping back to 3D.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_seq.reshape(-1, n_features)).reshape(n_samples, timesteps, n_features)
    X_val_scaled = scaler.transform(X_val_seq.reshape(-1, n_features)).reshape(X_val_seq.shape[0], timesteps, n_features)

    # Enable GPU memory growth if using GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Model builder using the Functional API.
    def create_attention_lstm_model(lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Attention
        from tensorflow.keras.optimizers import Adam

        inputs = Input(shape=(timesteps, n_features))
        # The LSTM layer returns sequences which are fed to the Attention layer.
        lstm_out, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(inputs)
        # Apply built-in Attention layer on the LSTM output.
        attention_output = Attention()([lstm_out, lstm_out])
        # Use the last timestep from the attention output.
        x = Dropout(dropout_rate)(attention_output[:, -1, :])
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=learning_rate),
            metrics=['accuracy']
        )
        return model

    clf = FixedKerasClassifier(model=create_attention_lstm_model, verbose=0)

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

    search.fit(X_train_scaled, y_train_seq)
    best_params = search.best_params_
    print(f"--- Best Parameters: {best_params}")

    # Remove the 'model__' prefix for model creation.
    model_params = {k.replace('model__', ''): v for k, v in best_params.items() if 'model__' in k}
    best_model = create_attention_lstm_model(**model_params)

    gc.collect()
    K.clear_session()

    best_model.fit(
        X_train_scaled,
        y_train_seq,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=0
    )

    y_pred = (best_model.predict(X_val_scaled) > 0.5).astype("int32").flatten()
    accuracy = accuracy_score(y_val_seq, y_pred)
    signals = np.where(y_pred == 0, -1, 1)
    sharpe = calculate_sharpe_ratio(calculate_returns(signals, val_returns_seq))

    print(f"--- Accuracy: {accuracy:.4f}")

    results = {
        'best_params': best_params,
        'accuracy': accuracy,
        'sharpe_ratio': sharpe,
        'cv_results': {
            'mean_test_score': search.cv_results_['mean_test_score'].tolist(),
            'params': [str(p) for p in search.cv_results_['params']]
        }
    }

    # Return the tuned parameters including the sequence input shape.
    tuned_params = best_params.copy()
    tuned_params['input_shape'] = (timesteps, n_features)
    print("Tuned parameters returned:", tuned_params)
    
    return results, {'model': best_model, 'scaler': scaler, 'tuned_params': tuned_params}

# Example usage:
if __name__ == "__main__":
    # Dummy data for testing purposes; replace with your actual data.
    X_train_dummy = np.random.rand(200, 20)  # 200 samples, 20 features
    y_train_dummy = np.random.randint(0, 2, 200)
    X_val_dummy = np.random.rand(50, 20)
    y_val_dummy = np.random.randint(0, 2, 50)
    val_returns_dummy = np.random.rand(50) * 0.02  # Dummy returns

    results, tuned = tune_attention_lstm(X_train_dummy, y_train_dummy, X_val_dummy, y_val_dummy, val_returns_dummy, seq_length=10)
    print("\nFinal tuning results:")
    print(results)
