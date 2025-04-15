import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

class LSTMModel:
    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2, learning_rate=0.001, epochs=20, batch_size=32, **kwargs):
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
        n_samples, timesteps, n_features = X.shape
        self.scaler.fit(X.reshape(-1, n_features))
        X_scaled = self.scaler.transform(X.reshape(-1, n_features)).reshape(n_samples, timesteps, n_features)
        self.model.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0
        )
        return self

    def predict(self, X):
        if X.ndim == 2:
            # Reshape to 3D: (samples, timesteps=1, n_features)
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
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
