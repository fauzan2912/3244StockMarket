import pickle
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

class XgboostModel:
    """
    A scikit-learn-compatible wrapper for XGBoost classification,
    with optional feature scaling and persistence.
    """
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        random_state: int = 42,
        **kwargs
    ):
        # Model hyperparameters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        # Any additional XGBClassifier params
        self.extra_params = kwargs

        # Internal objects
        self.scaler = StandardScaler()
        self.model = None

    def build_model(self):
        """
        Instantiate the XGBClassifier with current hyperparameters.
        """
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        params.update(self.extra_params)

        self.model = XGBClassifier(**params)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the XGBoost model on the provided data.

        Args:
            X: 2D feature array (n_samples, n_features)
            y: 1D label array (n_samples,)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Build and train
        self.build_model()
        self.model.fit(X_scaled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: 2D feature array (n_samples, n_features)

        Returns:
            1D array of predicted class labels
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_params(self, deep: bool = True) -> dict:
        """
        For compatibility with scikit-learn's hyperparameter search.
        """
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'random_state': self.random_state,
        }
        params.update(self.extra_params)
        return params

    def set_params(self, **parameters):
        """
        Set hyperparameters (for scikit-learn compatibility).
        """
        for name, value in parameters.items():
            if hasattr(self, name):
                setattr(self, name, value)
            else:
                self.extra_params[name] = value
        return self

    def save(self, filepath: str):
        """
        Persist the entire wrapper (scaler + model) to disk.
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> 'XgboostModel':
        """
        Load a persisted XgboostModel from disk.
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
