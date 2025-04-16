import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

class DeepRNNModel:
    def __init__(self, input_shape, rnn_units=50, layers=2, dropout_rate=0.2, learning_rate=0.001, epochs=10, batch_size=32, **kwargs):
        self.input_shape = input_shape
        self.rnn_units = [rnn_units] * layers if isinstance(rnn_units, int) else rnn_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))

        # Add multiple RNN layers with dropout
        for units in self.rnn_units[:-1]:
            model.add(SimpleRNN(units, return_sequences=True))
            model.add(Dropout(self.dropout_rate))

        # Last RNN layer without return_sequences
        model.add(SimpleRNN(self.rnn_units[-1]))
        model.add(Dropout(self.dropout_rate))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=self.learning_rate),
            metrics=['accuracy']
        )
        return model

    def train(self, X, y):
        n_samples, timesteps, n_features = X.shape
        X_scaled = self.scaler.fit_transform(X.reshape(-1, n_features)).reshape(n_samples, timesteps, n_features)
        self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self

    def predict(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        n_samples, timesteps, n_features = X.shape
        X_scaled = self.scaler.transform(X.reshape(-1, n_features)).reshape(n_samples, timesteps, n_features)
        return (self.model.predict(X_scaled) > 0.5).astype("int32").flatten()

    def predict_proba(self, X):
        if X.ndim == 2:
            X = X.reshape(X.shape[0], 1, X.shape[1])
        n_samples, timesteps, n_features = X.shape
        X_scaled = self.scaler.transform(X.reshape(-1, n_features)).reshape(n_samples, timesteps, n_features)
        return self.model.predict(X_scaled).flatten()

    def get_feature_importance(self, feature_names=None):
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.input_shape[-1])]
        return pd.DataFrame({'Feature': feature_names, 'Importance': [0]*len(feature_names)})

    def get_params(self):
        return {
            'input_shape': self.input_shape,
            'rnn_units': self.rnn_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
