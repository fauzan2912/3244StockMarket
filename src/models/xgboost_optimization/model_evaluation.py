"""
XGBoost Model Performance Evaluation

This script evaluates XGBoost model performance across different optimization approaches:
1. Comparing rolling vs expanding window strategies
2. Evaluating binary vs multiclass prediction approaches
3. Assessing the impact of enhanced feature engineering
4. Analyzing the effect of correlation features
5. Measuring the improvement from hyperparameter optimization

This script integrates all the optimization components and provides comprehensive evaluation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from enhanced_features import (
    add_technical_indicators, 
    add_market_regime_features,
    add_volatility_features,
    add_seasonality_features,
    add_statistical_features
)

from correlation_analysis import (
    load_stock_data,
    create_correlation_features,
    analyze_feature_importance,
    analyze_correlation_feature_impact
)

from multiclass_prediction import (
    create_multiclass_target,
    handle_class_imbalance,
    evaluate_multiclass_prediction,
    plot_multiclass_confusion_matrix,
    analyze_financial_impact
)

from hyperparameter_optimization import (
    optimize_xgboost_comprehensive
)

# Constants
STOCKS = ['AAPL', 'MSFT', 'GOOGL']
HORIZONS = [1, 3, 7, 30, 252]
RESULTS_DIR = "./results/"
DATA_DIR = "./data/"

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def load_and_preprocess_data(stocks, data_dir):
    """
    Load and preprocess stock data.
    
    Args:
        stocks: List of stock symbols
        data_dir: Directory containing stock data
        
    Returns:
        Dictionary of preprocessed DataFrames
    """
    print("Loading and preprocessing stock data...")
    
    # Load stock data
    stock_dfs = load_stock_data(stocks, data_dir)
    
    # Preprocess each stock
    preprocessed_dfs = {}
    
    for stock, df in stock_dfs.items():
        print(f"\nPreprocessing {stock}...")
        
        # Add technical indicators
        df = add_technical_indicators(df)
        print(f"✅ Added technical indicators for {stock}")
        
        # Add market regime features
        df = add_market_regime_features(df)
        print(f"✅ Added market regime features for {stock}")
        
        # Add volatility features
        df = add_volatility_features(df)
        print(f"✅ Added volatility features for {stock}")
        
        # Add seasonality features
        df = add_seasonality_features(df)
        print(f"✅ Added seasonality features for {stock}")
        
        # Add statistical features
        df = add_statistical_features(df)
        print(f"✅ Added statistical features for {stock}")
        
        # Store preprocessed dataframe
        preprocessed_dfs[stock] = df
    
    # Add correlation features
    for stock in stocks:
        print(f"\nAdding correlation features for {stock}...")
        preprocessed_dfs[stock] = create_correlation_features(stock_dfs, stock)
        print(f"✅ Added correlation features for {stock}")
    
    return preprocessed_dfs

def prepare_target_variables(preprocessed_dfs, horizons, multiclass=False):
    """
    Prepare target variables for different prediction horizons.
    
    Args:
        preprocessed_dfs: Dictionary of preprocessed DataFrames
        horizons: List of prediction horizons
        multiclass: Whether to create multiclass targets
        
    Returns:
        Dictionary of DataFrames with target variables
    """
    print("\nPreparing target variables...")
    
    result_dfs = {}
    
    for stock, df in preprocessed_dfs.items():
        stock_result = {}
        
        for horizon in horizons:
            print(f"Creating {'multiclass' if multiclass else 'binary'} target for {stock} at horizon {horizon}...")
            
            if multiclass:
                # Create multiclass target
                df_with_target = create_multiclass_target(df, horizon)
                target_col = f'Target_MC_{horizon}d'
            else:
                # Create binary target
                df_with_target = df.copy()
                df_with_target[f'Future_Return_{horizon}d'] = df_with_target['Close'].pct_change(periods=horizon).shift(-horizon)
                df_with_target[f'Target_{horizon}d'] = (df_with_target[f'Future_Return_{horizon}d'] > 0).astype(int)
                target_col = f'Target_{horizon}d'
            
            # Drop rows with NaN in target
            df_with_target = df_with_target.dropna(subset=[target_col])
            
            # Store result
            stock_result[horizon] = df_with_target
        
        result_dfs[stock] = stock_result
    
    return result_dfs

def train_and_evaluate_models(data_dfs, window_strategy='rolling', multiclass=False, use_correlation=True, optimize_hyperparams=True):
    """
    Train and evaluate XGBoost models with different configurations.
    
    Args:
        data_dfs: Dictionary of DataFrames with target variables
        window_strategy: 'rolling' or 'expanding'
        multiclass: Whether to use multiclass prediction
        use_correlation: Whether to use correlation features
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Dictionary with evaluation results
    """
    print(f"\nTraining and evaluating models with configuration:")
    print(f"- Window strategy: {window_strategy}")
    print(f"- Prediction type: {'Multiclass' if multiclass else 'Binary'}")
    print(f"- Using correlation features: {'Yes' if use_correlation else 'No'}")
    print(f"- Hyperparameter optimization: {'Yes' if optimize_hyperparams else 'No'}")
    
    results = {}
    
    for stock in data_dfs:
        stock_results = {}
        
        for horizon in data_dfs[stock]:
            print(f"\nProcessing {stock} for horizon {horizon}...")
            
            # Get data
            df = data_dfs[stock][horizon]
            
            # Determine target column
            if multiclass:
                target_col = f'Target_MC_{horizon}d'
            else:
                target_col = f'Target_{horizon}d'
            
            # Prepare features
            if use_correlation:
                # Use all features including correlation features
                feature_cols = [col for col in df.columns if col not in [target_col, f'Future_Return_{horizon}d']]
            else:
                # Exclude correlation features
                feature_cols = [col for col in df.columns if col not in [target_col, f'Future_Return_{horizon}d'] 
                               and not ('Corr_' in col or 'Beta_' in col or 'Lead_Lag_' in col or 'Regime_' in col)]
            
            X = df[feature_cols]
            y = df[target_col]
            
            # Train and evaluate model
            if window_strategy == 'rolling':
                results_dict = train_model_rolling_window(X, y, df[f'Future_Return_{horizon}d'], 
                                                         multiclass=multiclass, optimize_hyperparams=optimize_hyperparams)
            else:  # expanding
                results_dict = train_model_expanding_window(X, y, df[f'Future_Return_{horizon}d'], 
                                                           multiclass=multiclass, optimize_hyperparams=optimize_hyperparams)
            
            # Store results
            stock_results[horizon] = results_dict
        
        results[stock] = stock_results
    
    return results

def train_model_rolling_window(X, y, future_returns, window_size=90, step_size=30, multiclass=False, optimize_hyperparams=True):
    """
    Train XGBoost model using rolling window approach.
    
    Args:
        X: Feature matrix
        y: Target vector
        future_returns: Future returns for financial impact analysis
        window_size: Size of the rolling window
        step_size: Step size for the rolling window
        multiclass: Whether it's a multiclass problem
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Dictionary with evaluation results
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples) if not multiclass else np.zeros((n_samples, len(np.unique(y))))
    actual_returns = np.zeros(n_samples)
    valid_indices = np.zeros(n_samples, dtype=bool)
    models = []
    
    # Determine the number of windows
    n_windows = (n_samples - window_size) // step_size + 1
    
    for i in range(n_windows):
        # Define window indices
        start_idx = i * step_size
        end_idx = start_idx + window_size
        
        # Skip if we've reached the end of the data
        if end_idx >= n_samples:
            break
        
        # Split data into train and test
        X_train = X.iloc[start_idx:end_idx]
        y_train = y.iloc[start_idx:end_idx]
        
        # Define test indices
        test_start_idx = end_idx
        test_end_idx = min(test_start_idx + step_size, n_samples)
        
        # Skip if test set is empty
        if test_start_idx >= n_samples or test_end_idx <= test_start_idx:
            break
        
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]
        returns_test = future_returns.iloc[test_start_idx:test_end_idx]
        
        # Skip if training data has only one class
        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {start_idx} due to single class in y_train: {np.unique(y_train)}")
            continue
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            # Split training data for validation
            train_size = int(0.8 * len(X_train))
            X_train_subset, X_val_subset = X_train.iloc[:train_size], X_train.iloc[train_size:]
            y_train_subset, y_val_subset = y_train.iloc[:train_size], y_train.iloc[train_size:]
            
            # Optimize hyperparameters
            best_params, _ = optimize_xgboost_comprehensive(
                X_train_subset, y_train_subset, X_val_subset, y_val_subset, is_multiclass=multiclass
            )
        else:
            # Use default parameters
            if multiclass:
                best_params = {
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'eta': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'alpha': 0,
                    'lambda': 1,
                    'num_class': len(np.unique(y))
                }
            else:
                best_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'eta': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'alpha': 0,
                    'lambda': 1
                }
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(best_params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], 
                          early_stopping_rounds=20, verbose_eval=False)
        
        # Make predictions
        if multiclass:
            # Multiclass classification
            pred_probs = model.predict(dtest)
            preds = np.argmax(pred_probs, axis=1)
            predictions[test_start_idx:test_end_idx] = preds
        else:
            # Binary classification
            pred_probs = model.predict(dtest)
            preds = (pred_probs > 0.5).astype(int)
            predictions[test_start_idx:test_end_idx] = preds
        
        # Store actual returns and mark valid indices
        actual_returns[test_start_idx:test_end_idx] = returns_test.values
        valid_indices[test_start_idx:test_end_idx] = True
        
        # Store model
        models.append(model)
        
        print(f"Window {i+1}/{n_windows}: Trained model with {len(X_train)} samples, tested on {len(X_test)} samples")
    
    # Filter valid predictions and returns
    valid_predictions = predictions[valid_indices]
    valid_returns = actual_returns[valid_indices]
    valid_y = y.iloc[valid_indices].values
    
    # Evaluate predictions
    if multiclass:
        eval_metrics = evaluate_multiclass_prediction(valid_y, valid_predictions)
        financial_impact = analyze_financial_impact(valid_y, valid_predictions, valid_returns)
    else:
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_score': f1_score(valid_y, valid_predictions),
            'roc_auc': roc_auc_score(valid_y, valid_predictions) if len(np.unique(valid_y)) > 1 else None,
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        
        # Simple financial impact for binary
        long_returns = valid_returns[valid_predictions == 1].mean()
        short_returns = -valid_returns[valid_predictions == 0].mean()
        strategy_returns = np.where(valid_predictions == 1, valid_returns, -valid_returns)
        
        financial_impact = {
            'long_returns': long_returns,
            'short_returns': short_returns,
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
    
    # Print evaluation results
    print("\nEvaluation results:")
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"F1-Score: {eval_metrics['f1_score'] if not multiclass else eval_metrics['f1_macro']:.4f}")
    
    if not multiclass and eval_metrics['roc_auc'] is not None:
        print(f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    
    print("\nFinancial impact:")
    if multiclass:
        print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
        print(f"Strategy Sharpe: {financial_impact['strategy_sharpe']:.4f}")
    else:
        print(f"Long Returns: {financial_impact['long_returns']:.4f}")
        print(f"Short Returns: {financial_impact['short_returns']:.4f}")
        print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
        print(f"Strategy Sharpe: {financial_impact['strategy_sharpe']:.4f}")
    
    # Return results
    return {
        'predictions': valid_predictions,
        'actual_returns': valid_returns,
        'true_labels': valid_y,
        'eval_metrics': eval_metrics,
        'financial_impact': financial_impact,
        'models': models,
        'params': best_params if optimize_hyperparams else None
    }

def train_model_expanding_window(X, y, future_returns, initial_window_size=90, step_size=30, multiclass=False, optimize_hyperparams=True):
    """
    Train XGBoost model using expanding window approach.
    
    Args:
        X: Feature matrix
        y: Target vector
        future_returns: Future returns for financial impact analysis
        initial_window_size: Initial size of the window
        step_size: Step size for expanding the window
        multiclass: Whether it's a multiclass problem
        optimize_hyperparams: Whether to optimize hyperparameters
        
    Returns:
        Dictionary with evaluation results
    """
    n_samples = len(X)
    predictions = np.zeros(n_samples) if not multiclass else np.zeros((n_samples, len(np.unique(y))))
    actual_returns = np.zeros(n_samples)
    valid_indices = np.zeros(n_samples, dtype=bool)
    models = []
    
    # Determine the number of windows
    n_windows = (n_samples - initial_window_size) // step_size + 1
    
    for i in range(n_windows):
        # Define window indices
        end_idx = initial_window_size + i * step_size
        
        # Skip if we've reached the end of the data
        if end_idx >= n_samples:
            break
        
        # Split data into train and test
        X_train = X.iloc[:end_idx]
        y_train = y.iloc[:end_idx]
        
        # Define test indices
        test_start_idx = end_idx
        test_end_idx = min(test_start_idx + step_size, n_samples)
        
        # Skip if test set is empty
        if test_start_idx >= n_samples or test_end_idx <= test_start_idx:
            break
        
        X_test = X.iloc[test_start_idx:test_end_idx]
        y_test = y.iloc[test_start_idx:test_end_idx]
        returns_test = future_returns.iloc[test_start_idx:test_end_idx]
        
        # Skip if training data has only one class
        if len(np.unique(y_train)) < 2:
            print(f"Skipping window at index {end_idx} due to single class in y_train: {np.unique(y_train)}")
            continue
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            # Split training data for validation
            train_size = int(0.8 * len(X_train))
            X_train_subset, X_val_subset = X_train.iloc[:train_size], X_train.iloc[train_size:]
            y_train_subset, y_val_subset = y_train.iloc[:train_size], y_train.iloc[train_size:]
            
            # Optimize hyperparameters
            best_params, _ = optimize_xgboost_comprehensive(
                X_train_subset, y_train_subset, X_val_subset, y_val_subset, is_multiclass=multiclass
            )
        else:
            # Use default parameters
            if multiclass:
                best_params = {
                    'objective': 'multi:softprob',
                    'eval_metric': 'mlogloss',
                    'eta': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'alpha': 0,
                    'lambda': 1,
                    'num_class': len(np.unique(y))
                }
            else:
                best_params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'eta': 0.1,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'alpha': 0,
                    'lambda': 1
                }
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        model = xgb.train(best_params, dtrain, num_boost_round=100, evals=[(dtest, 'eval')], 
                          early_stopping_rounds=20, verbose_eval=False)
        
        # Make predictions
        if multiclass:
            # Multiclass classification
            pred_probs = model.predict(dtest)
            preds = np.argmax(pred_probs, axis=1)
            predictions[test_start_idx:test_end_idx] = preds
        else:
            # Binary classification
            pred_probs = model.predict(dtest)
            preds = (pred_probs > 0.5).astype(int)
            predictions[test_start_idx:test_end_idx] = preds
        
        # Store actual returns and mark valid indices
        actual_returns[test_start_idx:test_end_idx] = returns_test.values
        valid_indices[test_start_idx:test_end_idx] = True
        
        # Store model
        models.append(model)
        
        print(f"Expanding Window {i+1}/{n_windows}: Trained model with {len(X_train)} samples, tested on {len(X_test)} samples")
    
    # Filter valid predictions and returns
    valid_predictions = predictions[valid_indices]
    valid_returns = actual_returns[valid_indices]
    valid_y = y.iloc[valid_indices].values
    
    # Evaluate predictions
    if multiclass:
        eval_metrics = evaluate_multiclass_prediction(valid_y, valid_predictions)
        financial_impact = analyze_financial_impact(valid_y, valid_predictions, valid_returns)
    else:
        eval_metrics = {
            'accuracy': accuracy_score(valid_y, valid_predictions),
            'f1_score': f1_score(valid_y, valid_predictions),
            'roc_auc': roc_auc_score(valid_y, valid_predictions) if len(np.unique(valid_y)) > 1 else None,
            'confusion_matrix': confusion_matrix(valid_y, valid_predictions),
            'classification_report': classification_report(valid_y, valid_predictions, output_dict=True)
        }
        
        # Simple financial impact for binary
        long_returns = valid_returns[valid_predictions == 1].mean()
        short_returns = -valid_returns[valid_predictions == 0].mean()
        strategy_returns = np.where(valid_predictions == 1, valid_returns, -valid_returns)
        
        financial_impact = {
            'long_returns': long_returns,
            'short_returns': short_returns,
            'strategy_return': strategy_returns.mean(),
            'strategy_sharpe': strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
        }
    
    # Print evaluation results
    print("\nEvaluation results:")
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"F1-Score: {eval_metrics['f1_score'] if not multiclass else eval_metrics['f1_macro']:.4f}")
    
    if not multiclass and eval_metrics['roc_auc'] is not None:
        print(f"ROC-AUC: {eval_metrics['roc_auc']:.4f}")
    
    print("\nFinancial impact:")
    if multiclass:
        print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
        print(f"Strategy Sharpe: {financial_impact['strategy_sharpe']:.4f}")
    else:
        print(f"Long Returns: {financial_impact['long_returns']:.4f}")
        print(f"Short Returns: {financial_impact['short_returns']:.4f}")
        print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
        print(f"Strategy Sharpe: {financial_impact['strategy_sharpe']:.4f}")
    
    # Return results
    return {
        'predictions': valid_predictions,
        'actual_returns': valid_returns,
        'true_labels': valid_y,
        'eval_metrics': eval_metrics,
        'financial_impact': financial_impact,
        'models': models,
        'params': best_params if optimize_hyperparams else None
    }

def compare_model_configurations(all_results):
    """
    Compare model performance across different configurations.
    
    Args:
        all_results: Dictionary with results from different configurations
        
    Returns:
        DataFrame with comparison results
    """
    print("\nComparing model configurations...")
    
    # Prepare data for comparison
    comparison_data = []
    
    for config, results in all_results.items():
        window_strategy, prediction_type, use_correlation, optimize_hyperparams = config
        
        for stock in results:
            for horizon in results[stock]:
                result = results[stock][horizon]
                
                # Extract metrics
                if prediction_type == 'multiclass':
                    accuracy = result['eval_metrics']['accuracy']
                    f1_score = result['eval_metrics']['f1_macro']
                    roc_auc = None
                else:
                    accuracy = result['eval_metrics']['accuracy']
                    f1_score = result['eval_metrics']['f1_score']
                    roc_auc = result['eval_metrics']['roc_auc']
                
                # Extract financial metrics
                strategy_return = result['financial_impact']['strategy_return']
                strategy_sharpe = result['financial_impact']['strategy_sharpe']
                
                # Add to comparison data
                comparison_data.append({
                    'Window Strategy': window_strategy,
                    'Prediction Type': prediction_type,
                    'Use Correlation': 'Yes' if use_correlation else 'No',
                    'Optimize Hyperparams': 'Yes' if optimize_hyperparams else 'No',
                    'Stock': stock,
                    'Horizon': horizon,
                    'Accuracy': accuracy,
                    'F1-Score': f1_score,
                    'ROC-AUC': roc_auc,
                    'Strategy Return': strategy_return,
                    'Strategy Sharpe': strategy_sharpe
                })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Print summary
    print("\nSummary of model configurations:")
    
    # Group by configuration and calculate mean metrics
    summary = comparison_df.groupby(['Window Strategy', 'Prediction Type', 'Use Correlation', 'Optimize Hyperparams']).agg({
        'Accuracy': 'mean',
        'F1-Score': 'mean',
        'Strategy Return': 'mean',
        'Strategy Sharpe': 'mean'
    }).reset_index()
    
    print(summary)
    
    return comparison_df

def plot_comparison_results(comparison_df, output_dir):
    """
    Plot comparison results.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Output directory for plots
    """
    print("\nPlotting comparison results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot accuracy by horizon for different configurations
    plt.figure(figsize=(15, 10))
    
    # Group by configuration and horizon
    horizon_accuracy = comparison_df.groupby(['Window Strategy', 'Prediction Type', 'Horizon']).agg({
        'Accuracy': 'mean'
    }).reset_index()
    
    # Plot
    sns.lineplot(x='Horizon', y='Accuracy', hue='Window Strategy', style='Prediction Type', 
                data=horizon_accuracy, markers=True, dashes=False)
    
    plt.title('Accuracy by Horizon for Different Configurations')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(f"{output_dir}/accuracy_by_horizon.png")
    plt.close()
    
    # Plot strategy return by horizon for different configurations
    plt.figure(figsize=(15, 10))
    
    # Group by configuration and horizon
    horizon_return = comparison_df.groupby(['Window Strategy', 'Prediction Type', 'Horizon']).agg({
        'Strategy Return': 'mean'
    }).reset_index()
    
    # Plot
    sns.lineplot(x='Horizon', y='Strategy Return', hue='Window Strategy', style='Prediction Type', 
                data=horizon_return, markers=True, dashes=False)
    
    plt.title('Strategy Return by Horizon for Different Configurations')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Strategy Return')
    plt.grid(True)
    plt.savefig(f"{output_dir}/return_by_horizon.png")
    plt.close()
    
    # Plot impact of correlation features
    plt.figure(figsize=(15, 10))
    
    # Group by correlation usage
    corr_impact = comparison_df.groupby(['Use Correlation', 'Horizon']).agg({
        'Accuracy': 'mean',
        'Strategy Return': 'mean'
    }).reset_index()
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    sns.barplot(x='Horizon', y='Accuracy', hue='Use Correlation', data=corr_impact)
    plt.title('Impact of Correlation Features on Accuracy')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot return
    plt.subplot(1, 2, 2)
    sns.barplot(x='Horizon', y='Strategy Return', hue='Use Correlation', data=corr_impact)
    plt.title('Impact of Correlation Features on Strategy Return')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Strategy Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_impact.png")
    plt.close()
    
    # Plot impact of hyperparameter optimization
    plt.figure(figsize=(15, 10))
    
    # Group by hyperparameter optimization
    hyperparam_impact = comparison_df.groupby(['Optimize Hyperparams', 'Horizon']).agg({
        'Accuracy': 'mean',
        'Strategy Return': 'mean'
    }).reset_index()
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    sns.barplot(x='Horizon', y='Accuracy', hue='Optimize Hyperparams', data=hyperparam_impact)
    plt.title('Impact of Hyperparameter Optimization on Accuracy')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot return
    plt.subplot(1, 2, 2)
    sns.barplot(x='Horizon', y='Strategy Return', hue='Optimize Hyperparams', data=hyperparam_impact)
    plt.title('Impact of Hyperparameter Optimization on Strategy Return')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Strategy Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/hyperparameter_impact.png")
    plt.close()
    
    # Plot comparison of binary vs multiclass
    plt.figure(figsize=(15, 10))
    
    # Group by prediction type
    prediction_impact = comparison_df.groupby(['Prediction Type', 'Horizon']).agg({
        'Accuracy': 'mean',
        'Strategy Return': 'mean'
    }).reset_index()
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    sns.barplot(x='Horizon', y='Accuracy', hue='Prediction Type', data=prediction_impact)
    plt.title('Binary vs Multiclass: Impact on Accuracy')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    
    # Plot return
    plt.subplot(1, 2, 2)
    sns.barplot(x='Horizon', y='Strategy Return', hue='Prediction Type', data=prediction_impact)
    plt.title('Binary vs Multiclass: Impact on Strategy Return')
    plt.xlabel('Prediction Horizon (Days)')
    plt.ylabel('Strategy Return')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/binary_vs_multiclass.png")
    plt.close()

def main():
    """
    Main function to run the evaluation.
    """
    start_time = time.time()
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Load and preprocess data
    preprocessed_dfs = load_and_preprocess_data(STOCKS, DATA_DIR)
    
    # Prepare target variables for binary prediction
    binary_dfs = prepare_target_variables(preprocessed_dfs, HORIZONS, multiclass=False)
    
    # Prepare target variables for multiclass prediction
    multiclass_dfs = prepare_target_variables(preprocessed_dfs, HORIZONS, multiclass=True)
    
    # Define configurations to evaluate
    configurations = [
        # (window_strategy, prediction_type, use_correlation, optimize_hyperparams)
        ('rolling', 'binary', True, True),      # Rolling window, binary, with correlation, optimized
        ('expanding', 'binary', True, True),    # Expanding window, binary, with correlation, optimized
        ('rolling', 'multiclass', True, True),  # Rolling window, multiclass, with correlation, optimized
        ('expanding', 'multiclass', True, True),# Expanding window, multiclass, with correlation, optimized
        ('rolling', 'binary', False, True),     # Rolling window, binary, without correlation, optimized
        ('rolling', 'multiclass', False, True), # Rolling window, multiclass, without correlation, optimized
        ('rolling', 'binary', True, False),     # Rolling window, binary, with correlation, not optimized
        ('rolling', 'multiclass', True, False)  # Rolling window, multiclass, with correlation, not optimized
    ]
    
    # Train and evaluate models for each configuration
    all_results = {}
    
    for config in configurations:
        window_strategy, prediction_type, use_correlation, optimize_hyperparams = config
        
        print(f"\n{'='*80}")
        print(f"Evaluating configuration: {config}")
        print(f"{'='*80}")
        
        # Select appropriate data
        if prediction_type == 'binary':
            data_dfs = binary_dfs
        else:  # multiclass
            data_dfs = multiclass_dfs
        
        # Train and evaluate models
        results = train_and_evaluate_models(
            data_dfs, 
            window_strategy=window_strategy,
            multiclass=(prediction_type == 'multiclass'),
            use_correlation=use_correlation,
            optimize_hyperparams=optimize_hyperparams
        )
        
        # Store results
        all_results[config] = results
    
    # Compare model configurations
    comparison_df = compare_model_configurations(all_results)
    
    # Plot comparison results
    plot_comparison_results(comparison_df, RESULTS_DIR)
    
    # Save comparison results
    comparison_df.to_csv(f"{RESULTS_DIR}/model_comparison.csv", index=False)
    print(f"Saved model comparison to {RESULTS_DIR}/model_comparison.csv")
    
    # End time
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nTotal runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

if __name__ == "__main__":
    main()
