"""
Advanced Hyperparameter Optimization for XGBoost Stock Prediction

This script implements comprehensive hyperparameter optimization techniques for XGBoost:
1. Bayesian optimization with hyperopt
2. Randomized search with cross-validation
3. Sequential model-based optimization
4. Custom optimization for financial time series
5. Learning curve analysis

These techniques can be integrated into the main XGBoost optimization script.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, make_scorer
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from hyperopt.pyll.base import scope
import optuna
import warnings
warnings.filterwarnings('ignore')

def optimize_xgboost_randomized_search(X_train, y_train, X_val, y_val, is_multiclass=False, n_iter=20):
    """
    Optimize XGBoost hyperparameters using randomized search with cross-validation.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        is_multiclass: Whether it's a multiclass problem
        n_iter: Number of parameter settings sampled
        
    Returns:
        Dictionary with best parameters
    """
    print("Optimizing XGBoost hyperparameters using randomized search...")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
        'n_estimators': [50, 100, 200, 300, 500],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.1, 0.5, 1, 10],
        'reg_lambda': [0.1, 1, 5, 10, 50]
    }
    
    # Create XGBoost classifier
    if is_multiclass:
        xgb_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            num_class=len(np.unique(y_train))
        )
        scoring = 'f1_macro'
    else:
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False
        )
        scoring = 'roc_auc'
    
    # Create time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create randomized search
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring,
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    
    # Fit randomized search
    random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    
    # Get best parameters
    best_params = random_search.best_params_
    print(f"Best parameters from randomized search: {best_params}")
    
    # Evaluate best model
    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    if is_multiclass:
        f1 = f1_score(y_val, y_pred, average='macro')
        print(f"Validation accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    else:
        roc_auc = roc_auc_score(y_val, best_model.predict_proba(X_val)[:, 1])
        print(f"Validation accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    return best_params

def optimize_xgboost_hyperopt(X_train, y_train, X_val, y_val, is_multiclass=False, max_evals=50):
    """
    Optimize XGBoost hyperparameters using Bayesian optimization with hyperopt.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        is_multiclass: Whether it's a multiclass problem
        max_evals: Maximum number of evaluations
        
    Returns:
        Dictionary with best parameters
    """
    print("Optimizing XGBoost hyperparameters using Bayesian optimization (hyperopt)...")
    
    # Define the search space
    space = {
        'max_depth': scope.int(hp.quniform('max_depth', 3, 10, 1)),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.3)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 500, 50)),
        'subsample': hp.uniform('subsample', 0.6, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
        'min_child_weight': scope.int(hp.quniform('min_child_weight', 1, 7, 1)),
        'gamma': hp.uniform('gamma', 0, 0.5),
        'reg_alpha': hp.loguniform('reg_alpha', np.log(0.01), np.log(10)),
        'reg_lambda': hp.loguniform('reg_lambda', np.log(0.1), np.log(50))
    }
    
    # Define the objective function
    def objective(params):
        # Ensure parameters are in the correct format
        params['max_depth'] = int(params['max_depth'])
        params['n_estimators'] = int(params['n_estimators'])
        params['min_child_weight'] = int(params['min_child_weight'])
        
        # Create XGBoost classifier
        if is_multiclass:
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                use_label_encoder=False,
                num_class=len(np.unique(y_train)),
                **params
            )
        else:
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                **params
            )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate model
        y_pred = model.predict(X_val)
        
        if is_multiclass:
            # For multiclass, use F1-score
            score = f1_score(y_val, y_pred, average='macro')
        else:
            # For binary, use ROC-AUC
            score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        
        # Return negative score for minimization
        return {'loss': -score, 'status': STATUS_OK, 'model': model}
    
    # Run optimization
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=1
    )
    
    # Get best parameters
    best_params = {
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'n_estimators': int(best['n_estimators']),
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'min_child_weight': int(best['min_child_weight']),
        'gamma': best['gamma'],
        'reg_alpha': best['reg_alpha'],
        'reg_lambda': best['reg_lambda']
    }
    
    print(f"Best parameters from hyperopt: {best_params}")
    
    # Get best score
    best_score = -min(trials.losses())
    if is_multiclass:
        print(f"Best validation F1-score: {best_score:.4f}")
    else:
        print(f"Best validation ROC-AUC: {best_score:.4f}")
    
    return best_params

def optimize_xgboost_optuna(X_train, y_train, X_val, y_val, is_multiclass=False, n_trials=50):
    """
    Optimize XGBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        is_multiclass: Whether it's a multiclass problem
        n_trials: Number of trials
        
    Returns:
        Dictionary with best parameters
    """
    print("Optimizing XGBoost hyperparameters using Optuna...")
    
    # Define the objective function
    def objective(trial):
        # Define hyperparameters to optimize
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50, log=True)
        }
        
        # Create XGBoost classifier
        if is_multiclass:
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                use_label_encoder=False,
                num_class=len(np.unique(y_train)),
                **params
            )
        else:
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                **params
            )
        
        # Train model
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate model
        y_pred = model.predict(X_val)
        
        if is_multiclass:
            # For multiclass, use F1-score
            score = f1_score(y_val, y_pred, average='macro')
        else:
            # For binary, use ROC-AUC
            score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        
        return score
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters from Optuna: {best_params}")
    
    # Get best score
    best_score = study.best_value
    if is_multiclass:
        print(f"Best validation F1-score: {best_score:.4f}")
    else:
        print(f"Best validation ROC-AUC: {best_score:.4f}")
    
    # Plot optimization history
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(f"optuna_optimization_history_{'multiclass' if is_multiclass else 'binary'}.png")
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(f"optuna_param_importances_{'multiclass' if is_multiclass else 'binary'}.png")
    except:
        print("Warning: Could not generate Optuna visualization plots.")
    
    return best_params

def optimize_xgboost_financial_ts(X_train, y_train, X_val, y_val, is_multiclass=False, n_trials=30):
    """
    Custom optimization for XGBoost specifically for financial time series.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        is_multiclass: Whether it's a multiclass problem
        n_trials: Number of trials
        
    Returns:
        Dictionary with best parameters
    """
    print("Optimizing XGBoost hyperparameters specifically for financial time series...")
    
    # Define the objective function
    def objective(trial):
        # Define hyperparameters to optimize with financial time series focus
        params = {
            # Tree structure parameters
            'max_depth': trial.suggest_int('max_depth', 3, 8),  # Typically shallower trees work better for financial data
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),  # Higher values prevent overfitting
            
            # Regularization parameters
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10, log=True),  # L1 regularization
            'reg_lambda': trial.suggest_float('reg_lambda', 1, 50, log=True),  # L2 regularization
            'gamma': trial.suggest_float('gamma', 0.1, 1.0),  # Higher gamma for financial data
            
            # Sampling parameters
            'subsample': trial.suggest_float('subsample', 0.7, 0.9),  # Subsampling rows
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.9),  # Subsampling columns
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 0.9),  # Subsampling at each level
            
            # Learning parameters
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),  # Lower learning rates
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),  # More trees for robustness
            
            # Financial-specific parameters
            'scale_pos_weight': 1.0,  # Balanced classes assumed
            'base_score': 0.5  # Default prediction
        }
        
        # Create XGBoost classifier
        if is_multiclass:
            model = xgb.XGBClassifier(
                objective='multi:softprob',
                eval_metric='mlogloss',
                use_label_encoder=False,
                num_class=len(np.unique(y_train)),
                **params
            )
        else:
            model = xgb.XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                use_label_encoder=False,
                **params
            )
        
        # Train model with time series validation
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_train):
            X_fold_train, X_fold_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
            y_fold_train, y_fold_test = y_train.iloc[train_idx], y_train.iloc[test_idx]
            
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_test, y_fold_test)],
                early_stopping_rounds=20,
                verbose=False
            )
            
            y_fold_pred = model.predict(X_fold_test)
            
            if is_multiclass:
                # For multiclass, use F1-score
                fold_score = f1_score(y_fold_test, y_fold_pred, average='macro')
            else:
                # For binary, use ROC-AUC
                fold_score = roc_auc_score(y_fold_test, model.predict_proba(X_fold_test)[:, 1])
            
            scores.append(fold_score)
        
        # Final model training on all training data
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        
        if is_multiclass:
            # For multiclass, use F1-score
            val_score = f1_score(y_val, y_pred, average='macro')
        else:
            # For binary, use ROC-AUC
            val_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        
        # Combine cross-validation score and validation score
        # Weight validation score more heavily
        final_score = 0.3 * np.mean(scores) + 0.7 * val_score
        
        return final_score
    
    # Create study
    study = optuna.create_study(direction='maximize')
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params
    print(f"Best parameters for financial time series: {best_params}")
    
    # Get best score
    best_score = study.best_value
    print(f"Best combined score: {best_score:.4f}")
    
    return best_params

def analyze_learning_curves(X_train, y_train, X_val, y_val, params, is_multiclass=False):
    """
    Analyze learning curves for XGBoost to find optimal n_estimators.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: XGBoost parameters
        is_multiclass: Whether it's a multiclass problem
        
    Returns:
        Optimal number of estimators
    """
    print("Analyzing learning curves to find optimal number of estimators...")
    
    # Create parameter dictionary
    xgb_params = params.copy()
    
    # Remove n_estimators if present
    if 'n_estimators' in xgb_params:
        del xgb_params['n_estimators']
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set objective and eval_metric
    if is_multiclass:
        xgb_params['objective'] = 'multi:softprob'
        xgb_params['eval_metric'] = 'mlogloss'
        xgb_params['num_class'] = len(np.unique(y_train))
    else:
        xgb_params['objective'] = 'binary:logistic'
        xgb_params['eval_metric'] = 'logloss'
    
    # Train model with evaluation
    results = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        evals_result=results,
        verbose_eval=False
    )
    
    # Get optimal number of estimators
    optimal_n_estimators = model.best_iteration + 1
    print(f"Optimal number of estimators: {optimal_n_estimators}")
    
    # Plot learning curves
    plt.figure(figsize=(12, 6))
    
    # Get metric name
    metric_name = 'mlogloss' if is_multiclass else 'logloss'
    
    # Plot training and validation curves
    plt.plot(results['train'][metric_name], label='Training')
    plt.plot(results['val'][metric_name], label='Validation')
    
    # Add vertical line at optimal number of estimators
    plt.axvline(x=optimal_n_estimators, color='r', linestyle='--', label=f'Optimal: {optimal_n_estimators}')
    
    plt.xlabel('Number of Boosting Rounds')
    plt.ylabel(metric_name)
    plt.title('XGBoost Learning Curves')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(f"learning_curves_{'multiclass' if is_multiclass else 'binary'}.png")
    plt.close()
    
    return optimal_n_estimators

def optimize_xgboost_comprehensive(X_train, y_train, X_val, y_val, is_multiclass=False):
    """
    Comprehensive XGBoost hyperparameter optimization combining multiple approaches.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        is_multiclass: Whether it's a multiclass problem
        
    Returns:
        Dictionary with best parameters
    """
    print("Starting comprehensive XGBoost hyperparameter optimization...")
    
    # Step 1: Initial optimization with randomized search (faster)
    initial_params = optimize_xgboost_randomized_search(
        X_train, y_train, X_val, y_val, 
        is_multiclass=is_multiclass,
        n_iter=20
    )
    
    # Step 2: Refined optimization with financial time series focus
    refined_params = optimize_xgboost_financial_ts(
        X_train, y_train, X_val, y_val,
        is_multiclass=is_multiclass,
        n_trials=20
    )
    
    # Step 3: Final optimization with Bayesian approach
    # Use the best parameters from previous steps as a starting point
    # by adjusting the search space in hyperopt
    
    # Combine parameters from previous steps
    combined_params = {}
    for param in set(list(initial_params.keys()) + list(refined_params.keys())):
        if param in initial_params and param in refined_params:
            # If both have the parameter, take the average for numerical values
            if isinstance(initial_params[param], (int, float)) and isinstance(refined_params[param], (int, float)):
                if isinstance(initial_params[param], int):
                    combined_params[param] = int((initial_params[param] + refined_params[param]) / 2)
                else:
                    combined_params[param] = (initial_params[param] + refined_params[param]) / 2
            else:
                # For non-numerical, prefer the refined params
                combined_params[param] = refined_params[param]
        elif param in refined_params:
            combined_params[param] = refined_params[param]
        else:
            combined_params[param] = initial_params[param]
    
    # Step 4: Analyze learning curves to find optimal n_estimators
    optimal_n_estimators = analyze_learning_curves(
        X_train, y_train, X_val, y_val,
        combined_params,
        is_multiclass=is_multiclass
    )
    
    # Update n_estimators in combined parameters
    combined_params['n_estimators'] = optimal_n_estimators
    
    print(f"Final optimized parameters: {combined_params}")
    
    # Create and train final model with optimized parameters
    if is_multiclass:
        final_model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            use_label_encoder=False,
            num_class=len(np.unique(y_train)),
            **combined_params
        )
    else:
        final_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            **combined_params
        )
    
    # Train final model
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=50,
        verbose=False
    )
    
    # Evaluate final model
    y_pred = final_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    
    if is_multiclass:
        f1 = f1_score(y_val, y_pred, average='macro')
        print(f"Final model validation accuracy: {accuracy:.4f}, F1-score: {f1:.4f}")
    else:
        roc_auc = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])
        print(f"Final model validation accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    
    return combined_params, final_model

# Example usage
if __name__ == "__main__":
    # This is just an example of how to use these functions
    # In practice, you would integrate them into the main XGBoost optimization script
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate synthetic data
    X = np.random.randn(n_samples, n_features)
    
    # Binary classification
    y_binary = (np.random.randn(n_samples) > 0).astype(int)
    
    # Multiclass classification
    y_multi = np.random.randint(0, 4, size=n_samples)
    
    # Split data
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train_binary = y_binary[:train_size]
    y_train_multi = y_multi[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val_binary = y_binary[train_size:train_size+val_size]
    y_val_multi = y_multi[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test_binary = y_binary[train_size+val_size:]
    y_test_multi = y_multi[train_size+val_size:]
    
    # Convert to pandas DataFrame
    X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(n_features)])
    X_val_df = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(n_features)])
    X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(n_features)])
    
    y_train_binary_df = pd.Series(y_train_binary)
    y_val_binary_df = pd.Series(y_val_binary)
    y_test_binary_df = pd.Series(y_test_binary)
    
    y_train_multi_df = pd.Series(y_train_multi)
    y_val_multi_df = pd.Series(y_val_multi)
    y_test_multi_df = pd.Series(y_test_multi)
    
    # Example of using the optimization functions
    print("\nBinary Classification Example:")
    best_params_binary, best_model_binary = optimize_xgboost_comprehensive(
        X_train_df, y_train_binary_df, 
        X_val_df, y_val_binary_df,
        is_multiclass=False
    )
    
    print("\nMulticlass Classification Example:")
    best_params_multi, best_model_multi = optimize_xgboost_comprehensive(
        X_train_df, y_train_multi_df, 
        X_val_df, y_val_multi_df,
        is_multiclass=True
    )
