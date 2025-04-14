import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

class LogisticModel:
    """
    Logistic Regression model for stock price direction prediction
    """
    
    def __init__(self, C=1.0, max_iter=1000, random_state=42, penalty='l2', solver='liblinear', config_file=None):
        """
        Initialize the logistic regression model
        
        Args:
            C: Inverse of regularization strength
            max_iter: Maximum number of iterations
            random_state: Random seed for reproducibility
            penalty: Penalty type ('l1' or 'l2')
            solver: Algorithm to use ('liblinear', 'saga', etc.)
            config_file: Path to config file with parameters (overrides other args)
        """
        # If config file is provided, load parameters from it
        if config_file:
            import json
            import os
            
            # If config_file is just a name, assume it's in the config directory
            if not os.path.isabs(config_file) and not os.path.exists(config_file):
                config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config")
                config_file = os.path.join(config_dir, config_file)
                
                # Add .json extension if not present
                if not config_file.endswith('.json'):
                    config_file += '.json'
            
            with open(config_file, 'r') as f:
                params = json.load(f)
            
            print(f"--- Loaded parameters from {config_file}")
            
            # Set parameters from config file
            C = params.get('C', C)
            max_iter = params.get('max_iter', max_iter)
            random_state = params.get('random_state', random_state)
            penalty = params.get('penalty', penalty)
            solver = params.get('solver', solver)
        
        self.model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            penalty=penalty,
            solver=solver
        )
        self.feature_importance = None
    
    def train(self, X_train, y_train):
        """
        Train the logistic regression model
        
        Args:
            X_train: Training features
            y_train: Training target (binary)
        """
        # Fit the model
        self.model.fit(X_train, y_train)
        
        # Store feature importance (coefficients)
        self.feature_importance = self.model.coef_[0]
        
        # Print training accuracy
        train_accuracy = self.model.score(X_train, y_train)
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions
        
        Args:
            X: Features
            
        Returns:
            Binary predictions (0 or 1)
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Args:
            X: Features
            
        Returns:
            Probability of positive class (1)
        """
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance
        
        Args:
            feature_names: Names of features (optional)
            
        Returns:
            DataFrame with feature importance
        """
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': self.feature_importance
        })
        
        # Sort by absolute importance
        importance_df['Abs_Importance'] = importance_df['Importance'].abs()
        importance_df = importance_df.sort_values('Abs_Importance', ascending=False)
        importance_df = importance_df.drop('Abs_Importance', axis=1)
        
        return importance_df
    
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