# models/svm.py

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

class SVMModel:
    def __init__(self, **kwargs):
        self.params = {
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': 'scale',
            'probability': True,
            'random_state': 42
        }
        self.params.update(kwargs)
        self.model = SVC(**self.params)
        self.scaler = StandardScaler()
        self.feature_importance = None  # SVM doesn’t have true importance

    def train(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def get_feature_importance(self, feature_names):
        if self.model.kernel == 'linear':
            importance = np.abs(self.model.coef_[0])
            return pd.DataFrame({'Feature': feature_names, 'Importance': importance}).sort_values(by='Importance', ascending=False)
        else:
            return pd.DataFrame({'Feature': feature_names, 'Importance': 0})