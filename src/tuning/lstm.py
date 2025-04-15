import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler
from evaluation.metrics import calculate_sharpe_ratio, calculate_returns
from models.lstm import LstmModel
from functools import partial

def sequence_accuracy_score(y_true, y_pred, seq_length):
    """Calculate accuracy for sequence-based predictions"""
    # Compare only the predictable targets
    return accuracy_score(y_true[seq_length:], y_pred)

def get_sequences(X, y, seq_length):
    """
    Create properly aligned sequences where:
    - Each X sequence contains seq_length past observations
    - Each y sequence contains corresponding targets
    - Both have exactly the same length
    """
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X)):
            X_seq.append(X[i-seq_length:i])  # Past seq_length observations
            y_seq.append(y[i])   # Corresponding targets
    return np.array(X_seq).astype('float32'), np.array(y_seq).astype('float32')
        
def tune_lstm(X_train, y_train, X_val, y_val, val_returns):
    print("[TUNING] LSTM")
    # print(f'[DEBUG] len(X_train): {len(X_train)}, len(X_val): {len(X_val)}')
    val_size = len(X_val)
    if (val_size < 50):
        seq_length_grid = [10, 20]
    elif (val_size < 100):
        seq_length_grid = [30, 40]
    elif (val_size < 200):
        seq_length_grid = [50, 60]
    else:
        seq_length_grid = [60, 90]

    num_features = X_train.shape[1]
    
    best_score = -np.inf
    best_params = None
    best_model = None
    best_sharpe = 0

    for seq_length in seq_length_grid:  
        # print(f'[Sequence Length: {seq_length}]')
        # Params to be tuned
        param_grid = {
            'seq_length': [seq_length],
            'num_features': [num_features],
            'lstm_units': [128, 256],
            'dense_units': [32, 64, 128],
            'l1_reg': [1e-5, 1e-6],
            'l2_reg': [1e-4, 1e-5],
            'rec_drop': [0.1, 0.2, 0.3],
            'drop': [0.2, 0.3, 0.4],
            'eta': [0.001, 0.005, 0.01],
            'batch_size': [32, 64],
            'patience': [5, 10]
        }

        seq_scorer = make_scorer(
            partial(sequence_accuracy_score, seq_length=seq_length),  # Or use best seq_length
            needs_proba=False
        )
        random_search = RandomizedSearchCV(
            estimator=LstmModel(),
            param_distributions=param_grid,
            n_iter=5,
            scoring=seq_scorer,
            cv=2,
            n_jobs=1,
            verbose=2,
            random_state=42,
            error_score='raise'  # Important for debugging
        )
    
        random_search.fit(X_train, y_train)
        
        # print('[DEBUG] finish the training')

        cur_best_params = random_search.best_params_
        cur_best_model = random_search.best_estimator_

        # print('[DEBUG] get the current best param and models')

        y_pred = cur_best_model.predict(X_val)
        # print(f'len(y_pred): {len(y_pred)}')
        score = accuracy_score(y_val[seq_length:], y_pred)

        if score > best_score:
            best_score = score
            best_sharpe = calculate_sharpe_ratio(calculate_returns(y_pred, val_returns[seq_length:]))
            best_params = cur_best_params
            best_model = cur_best_model 

    return {
        'best_params': best_params,
        'accuracy': best_score,
        'sharpe_ratio': best_sharpe,
    }, {
        'model': best_model.model,
        'scaler': best_model.scaler
    }

