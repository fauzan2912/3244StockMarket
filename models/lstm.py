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
    def __init__(self, **kwargs):
        # params for building the lstm
        self.seq_length = kwargs.get('seq_length')
        self.num_features = kwargs.get('num_features')
        self.lstm_units = kwargs.get('lstm_units')
        self.dense_units = kwargs.get('dense_units')
        self.l1_reg = kwargs.get('l1_reg', 1e-5)
        self.l2_reg = kwargs.get('l2_reg', 1e-4)
        self.rec_drop = kwargs.get('rec_drop', 0.2)
        self.drop = kwargs.get('drop', 0.3)
        self.eta = kwargs.get('eta', 0.001)

        # params to train the lstm
        self.batch_size = kwargs.get('batch_size', 32)
        self.patience = kwargs.get('patience', 10)

        self.params = {}
        self.params.update(kwargs)
        
        self.model = None
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
        
    def build_model(self):
        """
        Build and compiled lstm model
        """
        # print('[DEBUG] build start')
        # print('[DEBUG] check for all params')
        # print(f'self.seq_length:{self.seq_length}')
        # print(f'self.num_features:{self.num_features}')
        # print(f'self.lstm_units:{self.lstm_units}')
        # print(f'self.l1_reg:{self.l1_reg}')
        # print(f'self.l2_reg:{self.l2_reg}')
        # print(f'self.rec_drop:{self.rec_drop}')
        # print(f'self.drop:{self.drop}')
        # print(f'self.dense_units:{self.dense_units}')
        # print(f'self.eta:{self.eta}')
        
        model = Sequential([
            Input(shape=(self.seq_length, self.num_features)),
            LSTM(self.lstm_units, 
                 return_sequences=True,
                 kernel_regularizer=L1L2(
                     l1=self.l1_reg, 
                     l2=self.l2_reg),
                 recurrent_dropout=self.rec_drop),
            Dropout(self.drop),
            LSTM(self.lstm_units // 2),
            BatchNormalization(),
            Dense(self.dense_units, activation='relu'),
            Dense(2, activation='softmax')
        ])
        optimizer = Adam(learning_rate=self.eta)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        # print('[DEBUG] build end')

    def fit(self, X_train, y_train):
        # print('[DEBUG] fit start')
        # print(f'[DEBUG] len(X_train), len(y_train), self.seq_length = {len(X_train), len(y_train), self.seq_length}')
        if self.model == None:
            self.build_model()
            
        X_train_seq, y_train_seq = self.prepare_data(X_train, y_train)
        # print(f'[DEBUG] len(X_train_seq), len(y_train_seq), self.seq_length = {len(X_train_seq), len(y_train_seq), self.seq_length}')
        
        history = self.model.fit(X_train_seq,
                  y_train_seq,
                  epochs=100,
                  batch_size=self.batch_size,
                  validation_split=0.2,
                  verbose=0,
                  callbacks=[
                      EarlyStopping(
                        monitor='val_loss',
                        patience=self.patience,  # Longer for noisy financial data
                        min_delta=0.001,  # Minimum change to qualify as improvement
                        restore_best_weights=True  # Keeps the best model found
                        ),
                      ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
                  ])

        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        print(f"Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")
        self.accuracy = train_acc

    def predict(self, X_test):
        # print('[DEBUG] start predict')
        # print('[Checking params]')
        # print(f'len(X_test):{len(X_test)}')
        X_test_seq = self.prepare_data_X(X_test)
        # print(f'X_test_seq.shape:{X_test_seq.shape}, self.seq_length:{self.seq_length}')
        pred_prob = self.model.predict(X_test_seq, verbose=0)
        # print(f'pred_prob.shape{pred_prob.shape}')
        pred = np.argmax(pred_prob, axis=1)
        # print('[DEBUG] Finish prediction')
        return pred

    def get_params(self, deep=True):
        return {
            'seq_length': self.seq_length,
            'num_features': self.num_features,
            'seq_length': self.seq_length,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'l1_reg': self.l1_reg,
            'l2_reg': self.l2_reg,
            'rec_drop': self.rec_drop,
            'drop': self.drop,
            'eta': self.eta,
            'batch_size': self.batch_size,
            'patience': self.patience
        }
        
    def set_params(self, **parameters):
        # print('[DEBUG] setting up params')
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
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
