import pandas as pd
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, BatchNormalization, Dropout, Bidirectional, Conv1D, Attention
from keras.regularizers import L1L2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler

class LstmModel():
    def __init__(self, num_params):
        self.seq_length = 90
        self.model = self.get_lstm(self.seq_length, num_params)
        self.scaler = StandardScaler()

    def get_sequences_X(self, X, seq_length):
        X_seq = []
        for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])
        return np.array(X_seq).astype('float32')
    
    def get_sequences(self, X, y, seq_length):
        """
        Create properly aligned sequences where:
        - Each X sequence contains seq_length past observations
        - Each y sequence contains corresponding targets
        - Both have exactly the same length
        """
        X_seq, y_seq = [], []
        for i in range(self.seq_length, len(X)):
                X_seq.append(X[i-seq_length:i])  # Past seq_length observations
                y_seq.append(y[i])   # Corresponding targets
        return np.array(X_seq).astype('float32'), np.array(y_seq).astype('float32')

    def prepare_data(self, X, y):
        """
        Normalise and sequentialise the data
        """
        X = self.scaler.fit_transform(X)
        return self.get_sequences(X, y, self.seq_length)

    def prepare_data_X(self, X):
        X = self.scaler.fit_transform(X)
        return self.get_sequences_X(X, self.seq_length)

    def get_lstm(self, seq_length, num_param):
        """
        Return compiled lstm model
        """
        model = Sequential([
            Input(shape=(seq_length, num_param)),
            LSTM(128, 
                 return_sequences=True,
                 kernel_regularizer=L1L2(l1=1e-5, l2=1e-4),
                 recurrent_dropout=0.2),
            Dropout(0.3),
            LSTM(64),
            BatchNormalization(),
            Dense(64, activation='relu'),
            Dense(2, activation='softmax')
        ])
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train):
        X_train_seq, y_train_seq = self.prepare_data(X_train, y_train)
        
        history = self.model.fit(X_train_seq,
                  y_train_seq,
                  epochs=5,
                  batch_size=32,
                  validation_split=0.2,
                  verbose=0,
                  callbacks=[
                      EarlyStopping(
                        monitor='val_loss',
                        patience=10,  # Longer for noisy financial data
                        min_delta=0.001,  # Minimum change to qualify as improvement
                        restore_best_weights=True  # Keeps the best model found
                        ),
                      ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                  ])

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        print(f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
        self.accuracy = train_acc
        
        return self

    def predict(self, X_test):
        X_test_seq = self.prepare_data_X(X_test)        
        pred_prob = self.model.predict(X_test_seq, verbose=0)

        return np.argmax(pred_prob, axis=1)
        
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, filepath):
        """
        Load the model from a file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)       
