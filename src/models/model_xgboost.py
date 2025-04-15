#!/usr/bin/env python3
"""
XGBoost Stock Market Prediction - Complete Implementation with Hyperparameter Tuning

This file contains a complete implementation of XGBoost for stock market prediction,
including the model class, evaluation metrics, hyperparameter tuning, and testing functionality.

"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from datetime import datetime
import pickle
import json
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Check if xgboost is installed, if not provide installation instructions
try:
    import xgboost as xgb
except ImportError:
    print("XGBoost is not installed. Please install it using:")
    print("pip install xgboost")
    print("\nThen run this script again.")
    sys.exit(1)

#######################
# XGBoost Model Class #
#######################

class XGBoostModel:
    """
    XGBoost model for stock price direction prediction
    """
    
    def __init__(self, learning_rate=0.1, max_depth=6, n_estimators=100, 
                 subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
                 gamma=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 random_state=42, early_stopping_rounds=20, config_file=None):
        """
        Initialize the XGBoost model
        
        Args:
            learning_rate: Step size shrinkage used to prevent overfitting
            max_depth: Maximum depth of a tree
            n_estimators: Number of boosting rounds
            subsample: Subsample ratio of the training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            min_child_weight: Minimum sum of instance weight needed in a child
            gamma: Minimum loss reduction required to make a further partition
            reg_alpha: L1 regularization term on weights
            reg_lambda: L2 regularization term on weights
            scale_pos_weight: Balance positive and negative weights
            random_state: Random seed for reproducibility
            early_stopping_rounds: Validation metric needs to improve at least once in 
                                  every early_stopping_rounds round(s) to continue training
            config_file: Path to config file with parameters (overrides other args)
        """
        # If config file is provided, load parameters from it
        if config_file:
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
            learning_rate = params.get('learning_rate', learning_rate)
            max_depth = params.get('max_depth', max_depth)
            n_estimators = params.get('n_estimators', n_estimators)
            subsample = params.get('subsample', subsample)
            colsample_bytree = params.get('colsample_bytree', colsample_bytree)
            min_child_weight = params.get('min_child_weight', min_child_weight)
            gamma = params.get('gamma', gamma)
            reg_alpha = params.get('reg_alpha', reg_alpha)
            reg_lambda = params.get('reg_lambda', reg_lambda)
            scale_pos_weight = params.get('scale_pos_weight', scale_pos_weight)
            random_state = params.get('random_state', random_state)
            early_stopping_rounds = params.get('early_stopping_rounds', early_stopping_rounds)
        
        # Store parameters
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'min_child_weight': min_child_weight,
            'gamma': gamma,
            'alpha': reg_alpha,
            'lambda': reg_lambda,
            'scale_pos_weight': scale_pos_weight,
            'seed': random_state
        }
        
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.model = None
        self.feature_importance = None
        self.feature_names = None
    
    def train(self, X_train, y_train, eval_set=None, verbose=True):
        """
        Train the XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target (binary)
            eval_set: Optional evaluation set for early stopping
            verbose: Whether to print training progress
        """
        # Store feature names if available
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Create evaluation set if provided
        watchlist = [(dtrain, 'train')]
        if eval_set is not None:
            X_val, y_val = eval_set
            deval = xgb.DMatrix(X_val, label=y_val)
            watchlist.append((deval, 'eval'))
        
        # Train the model
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=watchlist,
            early_stopping_rounds=self.early_stopping_rounds if len(watchlist) > 1 else None,
            verbose_eval=10 if verbose else 0
        )
        
        # Store feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')
        
        # Print training results
        if verbose:
            print(f"--- XGBoost model trained successfully")
            if len(watchlist) > 1 and hasattr(self.model, 'best_score'):
                print(f"--- Best validation log loss: {self.model.best_score}")
        
        return self
    
    def predict(self, X):
        """
        Make binary predictions
        
        Args:
            X: Features
            
        Returns:
            Binary predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X)
        
        # Get probabilities
        probabilities = self.model.predict(dtest)
        
        # Convert to binary predictions
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Predict probabilities
        
        Args:
            X: Features
            
        Returns:
            Probability of positive class (1)
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Convert to DMatrix
        dtest = xgb.DMatrix(X)
        
        # Get probabilities
        return self.model.predict(dtest)
    
    def get_feature_importance(self, feature_names=None, importance_type='gain'):
        """
        Get feature importance
        
        Args:
            feature_names: Names of features (optional)
            importance_type: Type of importance ('gain', 'weight', 'cover', 'total_gain', 'total_cover')
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Get feature importance
        importance = self.model.get_score(importance_type=importance_type)
        
        # Use provided feature names or stored feature names or default names
        if feature_names is None:
            if self.feature_names is not None:
                feature_names = self.feature_names
            else:
                feature_names = [f"f{i}" for i in range(len(importance))]
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': list(importance.keys()),
            'Importance': list(importance.values())
        })
        
        # Map feature indices to names if needed
        if all(f.startswith('f') for f in importance_df['Feature']):
            importance_df['Feature'] = importance_df['Feature'].apply(
                lambda x: feature_names[int(x[1:])] if int(x[1:]) < len(feature_names) else x
            )
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
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

#######################
# Evaluation Metrics  #
#######################

def calculate_returns(predictions, actual_returns):
    """
    Calculate returns from binary predictions
    - Long position (1): Buy the asset
    - Short position (0): Sell the asset
    
    Args:
        predictions: Binary predictions (0 or 1)
        actual_returns: Actual percentage returns
        
    Returns:
        strategy_returns: Returns from the trading strategy
    """
    # Convert 0/1 to -1/1 for short/long positions
    position = 2 * predictions - 1
    
    # Calculate strategy returns (position * actual return)
    strategy_returns = position * actual_returns
    
    return strategy_returns

def calculate_cumulative_returns(strategy_returns, initial_capital=10000):
    """Calculate cumulative returns from strategy returns"""
    cumulative_returns = (1 + strategy_returns).cumprod() * initial_capital
    return cumulative_returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Calculate the annualized Sharpe ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return np.min(drawdown)

def calculate_win_rate(returns):
    """Calculate win rate (percentage of positive returns)"""
    return np.mean(returns > 0)

def evaluate_trading_strategy(predictions, actual_returns, dates=None):
    """
    Evaluate a trading strategy based on predictions
    
    Args:
        predictions: Binary predictions (0 or 1)
        actual_returns: Actual percentage returns
        dates: Optional dates for the returns data
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    # Calculate strategy returns
    strategy_returns = calculate_returns(predictions, actual_returns)
    
    # Create a Series with dates if provided
    if dates is not None:
        strategy_returns = pd.Series(strategy_returns, index=dates)
    
    # Calculate metrics
    metrics = {
        'cumulative_return': (1 + strategy_returns).prod() - 1,
        'annualized_return': (1 + strategy_returns).prod() ** (252 / len(strategy_returns)) - 1,
        'sharpe_ratio': calculate_sharpe_ratio(strategy_returns),
        'max_drawdown': calculate_max_drawdown(strategy_returns),
        'win_rate': calculate_win_rate(strategy_returns),
        'total_trades': len(strategy_returns)
    }
    
    return metrics, strategy_returns

#######################
# Hyperparameter Tuning #
#######################

class SharpeRatioScorer:
    """
    Custom scorer for XGBoost hyperparameter tuning based on Sharpe ratio
    """
    def __init__(self, returns):
        """
        Initialize the scorer with actual returns
        
        Args:
            returns: Actual percentage returns for the validation set
        """
        self.returns = returns
    
    def __call__(self, estimator, X, y):
        """
        Calculate Sharpe ratio for the given estimator
        
        Args:
            estimator: XGBoost model
            X: Features
            y: Target
            
        Returns:
            Sharpe ratio
        """
        # Make predictions
        predictions = estimator.predict(X)
        
        # Calculate strategy returns
        strategy_returns = calculate_returns(predictions, self.returns)
        
        # Calculate Sharpe ratio
        sharpe = calculate_sharpe_ratio(strategy_returns)
        
        return sharpe

def tune_xgboost_hyperparameters(X_train, y_train, returns_train, param_grid=None, cv=5, verbose=1):
    """
    Tune XGBoost hyperparameters using grid search with Sharpe ratio as the scoring metric
    
    Args:
        X_train: Training features
        y_train: Training target (binary)
        returns_train: Actual percentage returns for the training set
        param_grid: Grid of hyperparameters to search
        cv: Number of cross-validation folds
        verbose: Verbosity level
        
    Returns:
        best_params: Best hyperparameters
        cv_results: Cross-validation results
    """
    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100, 200],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
    
    # Create XGBoost classifier
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=cv)
    
    # Create custom scorer based on Sharpe ratio
    # We need to handle the returns for each fold separately
    # This is a workaround since we can't directly use Sharpe ratio as a scorer
    
    def custom_sharpe_scorer(estimator, X, y):
        # Get indices of X in the original training set
        if isinstance(X, pd.DataFrame):
            indices = X.index
        else:
            # If X is a numpy array, we assume it's a subset of X_train
            # This is a simplification and might not work in all cases
            indices = np.arange(len(X))
        
        # Get corresponding returns
        fold_returns = returns_train[indices]
        
        # Make predictions
        predictions = estimator.predict(X)
        
        # Calculate strategy returns
        strategy_returns = calculate_returns(predictions, fold_returns)
        
        # Calculate Sharpe ratio
        sharpe = calculate_sharpe_ratio(strategy_returns)
        
        return sharpe
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=xgb_clf,
        param_grid=param_grid,
        scoring=custom_sharpe_scorer,
        cv=tscv,
        verbose=verbose,
        n_jobs=-1
    )
    
    # Fit grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters and CV results
    best_params = grid_search.best_params_
    cv_results = grid_search.cv_results_
    
    # Train a model with the best parameters
    best_model = xgb.XGBClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    
    # Make predictions
    predictions = best_model.predict(X_train)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_train, predictions)
    
    # Calculate Sharpe ratio
    strategy_returns = calculate_returns(predictions, returns_train)
    sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
    
    # Create result dictionary
    result = {
        'best_params': best_params,
        'accuracy': accuracy,
        'sharpe_ratio': sharpe_ratio,
        'cv_results': {
            'mean_test_score': cv_results['mean_test_score'].tolist(),
            'params': [str(p) for p in cv_results['params']]
        }
    }
    
    return result

#######################
# Visualization       #
#######################

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        
    plt.show()

def plot_cumulative_returns(model_predictions, actual_returns, dates=None, initial_capital=10000, save_path=None):
    """
    Plot cumulative returns for multiple models
    
    Args:
        model_predictions: Dictionary of model name to predictions
        actual_returns: Actual percentage returns
        dates: Optional dates for the returns data
        initial_capital: Initial capital amount
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(12, 8))
    
    # Create benchmark (buy and hold)
    benchmark_returns = pd.Series(actual_returns, index=dates) if dates is not None else pd.Series(actual_returns)
    benchmark_cumulative = (1 + benchmark_returns).cumprod() * initial_capital
    plt.plot(benchmark_cumulative, label="Buy & Hold", color="black", linestyle="--")
    
    # Plot each model's returns
    for model_name, predictions in model_predictions.items():
        # Calculate strategy returns
        position = 2 * predictions - 1
        strategy_returns = position * actual_returns
        
        # Convert to Series with dates if provided
        if dates is not None:
            strategy_returns = pd.Series(strategy_returns, index=dates)
        
        # Calculate cumulative returns
        cum_returns = (1 + strategy_returns).cumprod() * initial_capital
        
        # Plot
        plt.plot(cum_returns, label=model_name)
    
    plt.title("Cumulative Returns Comparison")
    plt.xlabel("Date" if dates is not None else "Trading Days")
    plt.ylabel(f"Portfolio Value (Initial: ${initial_capital})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()

def plot_feature_importance(importance_df, title="Feature Importance", save_path=None):
    """
    Plot feature importance
    
    Args:
        importance_df: DataFrame with feature importance
        title: Plot title
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        
    plt.show()

def plot_hyperparameter_tuning_results(cv_results, save_path=None):
    """
    Plot hyperparameter tuning results
    
    Args:
        cv_results: Cross-validation results from GridSearchCV
        save_path: Optional path to save the figure
    """
    # Convert to DataFrame
    results_df = pd.DataFrame({
        'params': cv_results['params'],
        'mean_test_score': cv_results['mean_test_score']
    })
    
    # Sort by mean test score
    results_df = results_df.sort_values('mean_test_score', ascending=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x='mean_test_score', y='params', data=results_df.head(10))
    plt.title('Top 10 Hyperparameter Combinations by Sharpe Ratio')
    plt.xlabel('Mean Sharpe Ratio')
    plt.ylabel('Hyperparameters')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        
    plt.show()

#######################
# Data Generation     #
#######################

def generate_sample_data(n_samples=500, n_features=10, random_state=42):
    """
    Generate sample data for testing the XGBoost model
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features to generate
        random_state: Random seed for reproducibility
        
    Returns:
        X_train, y_train, X_test, y_test, test_returns
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target (binary classification)
    # Use a simple rule: if sum of first 3 features > 0, then 1, else 0
    y = (X[:, :3].sum(axis=1) > 0).astype(int)
    
    # Split into train and test
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Generate returns for the training set
    train_returns = np.where(y_train == 1, np.random.normal(0.01, 0.02, len(y_train)), 
                           np.random.normal(-0.01, 0.02, len(y_train)))
    
    # Generate returns for the test set (correlated with target)
    test_returns = np.where(y_test == 1, np.random.normal(0.01, 0.02, len(y_test)), 
                           np.random.normal(-0.01, 0.02, len(y_test)))
    
    return X_train, y_train, train_returns, X_test, y_test, test_returns

#######################
# Main Test Function  #
#######################

def test_xgboost_model(save_plots=True, tune_hyperparameters=True):
    """
    Test the XGBoost model implementation with synthetic data
    
    Args:
        save_plots: Whether to save plots to files
        tune_hyperparameters: Whether to perform hyperparameter tuning
        
    Returns:
        model: Trained XGBoost model
        tuning_results: Hyperparameter tuning results (if performed)
    """
    print("\n--- Testing XGBoost Model Implementation ---\n")
    
    # Generate sample data
    X_train, y_train, train_returns, X_test, y_test, test_returns = generate_sample_data()
    
    # Create feature names
    feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
    
    # Convert to DataFrame with feature names
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Hyperparameter tuning
    tuning_results = None
    if tune_hyperparameters:
        print("Performing hyperparameter tuning...")
        
        # Define parameter grid
        param_grid = {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5],
            'n_estimators': [50, 100],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'gamma': [0, 0.1]
        }
        
        # Tune hyperparameters
        tuning_results = tune_xgboost_hyperparameters(
            X_train_df, y_train, train_returns, param_grid, cv=3, verbose=0
        )
        
        print(f"Best parameters: {tuning_results['best_params']}")
        print(f"Accuracy with best parameters: {tuning_results['accuracy']:.4f}")
        print(f"Sharpe ratio with best parameters: {tuning_results['sharpe_ratio']:.4f}")
        
        # Save tuning results to file
        with open('xgboost_tuning_results.json', 'w') as f:
            json.dump(tuning_results, f, indent=4)
        
        # Plot hyperparameter tuning results
        if save_plots:
            plot_hyperparameter_tuning_results(
                tuning_results['cv_results'], 
                save_path='xgboost_hyperparameter_tuning.png'
            )
        
        # Create and train XGBoost model with best parameters
        print("\nTraining XGBoost model with best parameters...")
        model = XGBoostModel(
            learning_rate=tuning_results['best_params']['learning_rate'],
            max_depth=tuning_results['best_params']['max_depth'],
            n_estimators=tuning_results['best_params']['n_estimators'],
            subsample=tuning_results['best_params']['subsample'],
            colsample_bytree=tuning_results['best_params']['colsample_bytree'],
            gamma=tuning_results['best_params']['gamma'],
            random_state=42,
            early_stopping_rounds=10
        )
    else:
        # Create and train XGBoost model with default parameters
        print("Training XGBoost model with default parameters...")
        model = XGBoostModel(
            learning_rate=0.1,
            max_depth=3,
            n_estimators=50,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1,
            scale_pos_weight=1,
            random_state=42,
            early_stopping_rounds=10
        )
    
    # Train the model with a small validation set
    val_size = int(0.2 * len(X_train_df))
    X_val = X_train_df.iloc[-val_size:]
    y_val = y_train[-val_size:]
    X_train_df = X_train_df.iloc[:-val_size]
    y_train = y_train[:-val_size]
    
    # Train the model
    model.train(X_train_df, y_train, eval_set=(X_val, y_val))
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_test_df)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - XGBoost Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_plots:
        plt.savefig('xgboost_confusion_matrix.png')
    plt.close()
    
    # Calculate trading strategy returns
    position = 2 * predictions - 1  # Convert 0/1 to -1/1
    strategy_returns = position * test_returns
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Plot cumulative returns
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns, label='XGBoost Strategy')
    plt.plot((1 + test_returns).cumprod(), label='Buy & Hold', linestyle='--')
    plt.title('Cumulative Returns')
    plt.xlabel('Trading Days')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_plots:
        plt.savefig('xgboost_returns.png')
    plt.close()
    
    # Calculate performance metrics
    sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
    cumulative_return = cumulative_returns[-1] - 1
    win_rate = calculate_win_rate(strategy_returns)
    
    print("\n--- XGBoost Trading Performance ---")
    print(f"Cumulative Return: {cumulative_return:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Win Rate: {win_rate:.4f}")
    
    # Get feature importance
    importance_df = model.get_feature_importance()
    print("\nTop Feature Importance:")
    print(importance_df.head())
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(10))
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    if save_plots:
        plt.savefig('xgboost_feature_importance.png')
    plt.close()
    
    # Test save and load functionality
    print("\nTesting save and load functionality...")
    model_path = "xgboost_model_test.pkl"
    model.save(model_path)
    
    # Load the model
    loaded_model = XGBoostModel.load(model_path)
    
    # Make predictions with loaded model
    loaded_predictions = loaded_model.predict(X_test_df)
    
    # Check if predictions match
    predictions_match = (predictions == loaded_predictions).all()
    print(f"Predictions match after save/load: {predictions_match}")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)
    
    print("\n--- XGBoost Model Test Complete ---")
    if save_plots:
        print("Evaluation plots saved as PNG files")
    
    return model, tuning_results

#######################
# Main Execution      #
#######################

if __name__ == "__main__":
    # Test the XGBoost model with hyperparameter tuning
    model, tuning_results = test_xgboost_model(save_plots=True, tune_hyperparameters=True)
    
    print("\nTo use this model in your own code:")
    print("1. Import the XGBoostModel class")
    print("2. Create an instance with your desired hyperparameters")
    print("3. Train the model with your data")
    print("4. Make predictions and evaluate performance")
    print("\nExample:")
    print("model = XGBoostModel(learning_rate=0.1, max_depth=3)")
    print("model.train(X_train, y_train)")
    print("predictions = model.predict(X_test)")
