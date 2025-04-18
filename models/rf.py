import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

class RandomForestModel:
    def __init__(self, **kwargs):
        self.model = RandomForestClassifier(**kwargs)
        self.feature_importance = None
        self.scaler = None
        self.params = kwargs  

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        self.feature_importance = self.model.feature_importances_

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_importance(self, feature_names=None):
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet")

        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importance
        })

        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        return importance_df

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)


    



    
