# src/tuning/lstm.py
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from evaluation.metrics import calculate_sharpe_ratio, calculate_returns
from scikeras.wrappers import KerasClassifier as _KerasClassifier
import tensorflow as tf
from tensorflow.keras import backend as K
import gc

# Custom wrapper to mark the estimator as a classifier
class FixedKerasClassifier(_KerasClassifier):
    _estimator_type = "classifier"

def tune_lstm(X_train, y_train, X_val, y_val, val_returns):
    print("\n--- Tuning LSTM Hyperparameters ---")

    # Ensure inputs are 3D. If they are 2D, add a time dimension.
    if len(X_train.shape) == 2:
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    if len(X_val.shape) == 2:
        X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

    n_samples, timesteps, n_features = X_train.shape

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, timesteps, n_features)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape[0], timesteps, n_features)

    # Enable GPU memory growth
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Model builder using Functional API (avoids input_shape warning)
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

    # Use model__ prefix so that scikeras knows these go to the model builder function.
    param_grid = {
        'model__lstm_units': [50, 100],
        'model__dropout_rate': [0.2, 0.5],
        'model__learning_rate': [0.001, 0.0001],
        'batch_size': [16, 32],  # smaller batch sizes to reduce memory usage
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
        n_jobs=1,  # sequential execution to save GPU memory
        verbose=1,
        random_state=42
    )

    search.fit(X_train_scaled, y_train)
    best_params = search.best_params_
    print("--- Best Parameters:", best_params)

    # Remove the 'model__' prefix for model creation.
    model_params = {k.replace('model__', ''): v for k, v in best_params.items() if k.startswith('model__')}
    best_model = create_model(**model_params)

    gc.collect()
    K.clear_session()

    best_model.fit(
        X_train_scaled,
        y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=0
    )

    y_pred = (best_model.predict(X_val_scaled) > 0.5).astype("int32")
    accuracy = accuracy_score(y_val, y_pred)
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

    # Ensure that the tuned parameters include the 'input_shape' required by LSTMModel.
    tuned_params = best_params.copy()
    tuned_params['input_shape'] = (timesteps, n_features)
    print("Tuned parameters returned:", tuned_params)
    
    return results, {'model': best_model, 'scaler': scaler, 'tuned_params': tuned_params}
