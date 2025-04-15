from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, make_scorer  
import numpy as np
from evaluation.metrics import calculate_sharpe_ratio, calculate_returns

def tune_random_forest(X_train, y_train, X_test, y_test, test_returns):
    """
    Tune hyperparameters for random forest
    
    Args:
        X_train: Training features
        y_train: Binary training labels (0: loss, 1: gain)
        X_test: Testing features
        y_test: Binary testing labels
        
    Returns:
        Dictionary with best parameters and results
    """
    print("\n--- Tuning Random Forest Hyperparameters ---")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'class_weight': [None, 'balanced'],
        'random_state': [42]
    }
    
    # Define the model
    rf = RandomForestClassifier()
    
    # Use F1 score for classification tasks
    scorer = make_scorer(f1_score)

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_grid,
        n_iter=20,
        scoring=scorer,
        cv=5,
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
        
    # Fit randomized search
    random_search.fit(X_train, y_train)
    
    # Best params
    best_params = random_search.best_params_
    
    # Train model with best params
    best_model = RandomForestClassifier(**best_params)
    best_model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = best_model.predict(X_test)
    
    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    signals = np.where(y_pred == 0, -1, 1)
    strategy_returns = calculate_returns(signals, test_returns)

    sharpe = calculate_sharpe_ratio(strategy_returns)
    
    print(f"--- Best Parameters: {best_params}")
    print(f"--- Accuracy on Test Set: {accuracy:.4f}")
    print(f"--- F1 Score on Test Set: {f1:.4f}")
    
    print("--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    results = {
        'best_params': best_params,
        'accuracy': accuracy,
        'f1_score': f1,
        'sharpe_ratio': sharpe,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'cv_results': {
            'mean_test_score': random_search.cv_results_['mean_test_score'].tolist(),
            'params': [str(p) for p in random_search.cv_results_['params']]
        }
    }
    
    return results, best_model
