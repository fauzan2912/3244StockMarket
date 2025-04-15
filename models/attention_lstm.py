import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Multiply, Lambda, Activation
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import tensorflow.keras.backend as K

class AttentionLSTMModel:
    def __init__(self, input_shape, lstm_units=50, dropout_rate=0.2,
                 learning_rate=0.001, epochs=20, batch_size=32, **kwargs):
        # input_shape expected as (timesteps, n_features)
        self.input_shape = input_shape  
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = StandardScaler()
        self.model = self._build_model()

    def attention_layer(self, inputs):
        # Compute raw attention scores: shape (batch, timesteps, 1)
        attention_scores = Dense(1, activation='tanh')(inputs)
        # Squeeze the last dimension so shape becomes (batch, timesteps)
        attention_scores = Lambda(lambda x: K.squeeze(x, axis=-1))(attention_scores)
    
        # Define a function that chooses between ones or softmax based on the dynamic number of timesteps
        def compute_attention_weights(scores):
            # Determine the number of timesteps dynamically
            dynamic_timesteps = tf.shape(scores)[1]
            # If there is only one timestep, return ones; otherwise apply softmax
            return tf.cond(
                tf.equal(dynamic_timesteps, 1),
                lambda: tf.ones_like(scores),
                lambda: tf.nn.softmax(scores, axis=-1)
            )
    
        # Wrap the conditional logic in a Lambda layer
        attention_weights = Lambda(compute_attention_weights)(attention_scores)
    
        # Expand dimensions so that attention_weights have shape (batch, timesteps, 1)
        attention_weights = Lambda(lambda x: K.expand_dims(x, axis=-1))(attention_weights)
    
        # Compute the context vector as the weighted sum of inputs along the timesteps axis
        context_vector = Multiply()([inputs, attention_weights])
        context_vector = Lambda(lambda x: K.sum(x, axis=1))(context_vector)
    
        return context_vector


    def _build_model(self):
        inputs = Input(shape=self.input_shape)
        lstm_out = LSTM(self.lstm_units, return_sequences=True)(inputs)
        attention_out = self.attention_layer(lstm_out)
        x = Dropout(self.dropout_rate)(attention_out)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(learning_rate=self.learning_rate),
                      metrics=['accuracy'])
        return model

    def train(self, X, y):
        n_samples, timesteps, n_features = X.shape
        # Fit scaler on reshaped data and then transform to preserve shape.
        self.scaler.fit(X.reshape(-1, n_features))
        X_scaled = self.scaler.transform(X.reshape(-1, n_features)).reshape(n_samples, timesteps, n_features)
        self.model.fit(X_scaled, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
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
        return pd.DataFrame({'Feature': feature_names, 'Importance': [0] * len(feature_names)})

    def get_params(self):
        return {
            'input_shape': self.input_shape,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }
