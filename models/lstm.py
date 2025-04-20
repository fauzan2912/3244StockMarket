# src/tuning/lstm.py

import numpy as np
import pandas as pd
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score

from scikeras.wrappers import KerasClassifier as _KerasClassifier
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam


class LSTMModel:
    def __init__(
        self,
        input_shape,
        lstm_units=50,
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=20,
        batch_size=32,
        **kwargs
    ):
        self.input_shape = input_shape  # (timesteps, n_features)
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        x = LSTM(self.lstm_units)(inputs)
        x = Dropout(self.dropout_rate)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return model

    def train(self, X, y):
        # scale and reshape
        n, t, f = X.shape
        flat = X.reshape(-1, f)
        self.scaler.fit(flat)
        Xs = self.scaler.transform(flat).reshape(n, t, f)

        K.clear_session()
        gc.collect()

        self.model.fit(
            Xs, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self

    def predict(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        n, t, f = X.shape
        flat = X.reshape(-1, f)
        Xs = self.scaler.transform(flat).reshape(n, t, f)
        preds = self.model.predict(Xs, verbose=0)
        return (preds > 0.5).astype("int32").flatten()

    def predict_proba(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        n, t, f = X.shape
        flat = X.reshape(-1, f)
        Xs = self.scaler.transform(flat).reshape(n, t, f)
        return self.model.predict(Xs, verbose=0).flatten()

    def get_feature_importance(self, feature_names=None):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.input_shape[-1])]
        return pd.DataFrame({'Feature': feature_names, 'Importance': [0]*len(feature_names)})

    def get_params(self, deep=True):
        return {
            'input_shape': self.input_shape,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }


class FixedKerasClassifier(_KerasClassifier):
    _estimator_type = "classifier"

    @classmethod
    def __sklearn_tags__(cls, *args, **kwargs):
        """
        Override sklearn tag resolution so that RandomizedSearchCV
        never calls super().__sklearn_tags__()
        """
        return {
            'requires_fit': True,
            'binary_only': False,
            'multioutput': False,
            'allow_nan': True,
            'X_types': ['2darray'],
            'no_validation': False
        }


def tune_lstm(X_train, y_train, X_val, y_val, val_returns):
    """
    Tune an LSTM over temporal sequences using RandomizedSearchCV and TimeSeriesSplit.
    Returns: ({'best_params': dict}, {'model': keras.Model, 'scaler': StandardScaler})
    """
    print("\n--- Tuning LSTM Hyperparameters with Temporal Sequences ---")

    # 1) Build sequences
    seq_length = 10
    Xs, ys = [], []
    for i in range(seq_length, len(X_train)):
        Xs.append(X_train[i-seq_length:i])
        ys.append(y_train[i])
    Xs = np.array(Xs)
    ys = np.array(ys)

    n, t, f = Xs.shape
    print(f"--- Data transformed: {n} training sequences of length {t} with {f} features.")

    # 2) Scale
    scaler = StandardScaler()
    flat = Xs.reshape(-1, f)
    flat_s = scaler.fit_transform(flat)
    Xs_s = flat_s.reshape(n, t, f)

    # 3) Hyperparameter space
    param_dist = {
        'model__lstm_units':    [50, 100, 150],
        'model__dropout_rate':  [0.2, 0.3],
        'model__learning_rate': [1e-3, 1e-4],
        'epochs':               [10, 20],
        'batch_size':           [32, 64]
    }

    # 4) Wrap in our fixed KerasClassifier
    keras_clf = FixedKerasClassifier(
        model=LSTMModel,
        input_shape=(t, f),
        verbose=0
    )

    # 5) Time-series CV
    cv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(
        estimator=keras_clf,
        param_distributions=param_dist,
        n_iter=5,
        scoring='accuracy',
        cv=cv,
        n_jobs=1,
        random_state=42
    )

    # 6) Clear and fit
    K.clear_session()
    gc.collect()
    search.fit(Xs_s, ys)

    best = search.best_params_
    est  = search.best_estimator_
    mdl  = est.model_

    return {'best_params': best}, {'model': mdl, 'scaler': scaler}
