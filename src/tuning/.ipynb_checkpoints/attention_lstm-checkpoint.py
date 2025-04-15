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

class FixedKerasClassifier(_KerasClassifier):
    _estimator_type = "classifier"

def tune_attention_lstm(X_train, y_train, X_val, y_val, val_returns):
    print("\n--- Tuning Attention LSTM Hyperparameters ---")

    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    if len(X_val.shape) == 2:
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

    n_samples, timesteps, n_features = X_train.shape

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], timesteps, n_features)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    def create_attention_lstm_model(lstm_units=50, dropout_rate=0.2, learning_rate=0.001):
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Attention
        from tensorflow.keras.optimizers import Adam

        inputs = Input(shape=(timesteps, n_features))
        lstm_out, state_h, state_c = LSTM(lstm_units, return_sequences=True, return_state=True)(inputs)
        attention_output = Attention()([lstm_out, lstm_out])
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
        n_jobs=1,
        verbose=1,
        random_state=42
    )

    search.fit(X_train_scaled, y_train)
    best_params = search.best_params_

    model_params = {k.replace('model__', ''): v for k, v in best_params.items() if 'model__' in k}

    best_model = create_attention_lstm_model(**model_params)

    gc.collect()
    K.clear_session()

    best_model.fit(
        X_train_scaled,
        y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=0
    )

    y_pred = (best_model.predict(X_val_scaled) > 0.5).astype("int32").flatten()
    accuracy = accuracy_score(y_val, y_pred)
    signals = np.where(y_pred == 0, -1, 1)
    sharpe = calculate_sharpe_ratio(calculate_returns(signals, val_returns))

    print(f"--- Best Parameters: {best_params}")
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

    return results, {'model': best_model, 'scaler': scaler}
