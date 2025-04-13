"""
Multiclass Stock Price Prediction Implementation

This script implements a multiclass approach for stock price prediction:
1. Defines granular price movement classes based on percentage ranges
2. Implements specialized evaluation metrics for multiclass prediction
3. Provides visualization tools for multiclass prediction results
4. Includes confusion matrix analysis with financial implications
5. Implements class weight balancing for imbalanced datasets

This can be integrated with the main XGBoost optimization script.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from collections import Counter

# Define class labels for better readability
CLASS_LABELS = {
    0: "Down >5%",
    1: "Down 3-5%",
    2: "Down 1-3%",
    3: "Down <1%",
    4: "Up <1%",
    5: "Up 1-3%",
    6: "Up 3-5%",
    7: "Up >5%"
}

def create_multiclass_target(df, horizon, thresholds=None):
    """
    Create multiclass target variable based on future returns.
    
    Args:
        df: DataFrame with stock price data
        horizon: Prediction horizon in days
        thresholds: Custom thresholds for class boundaries (optional)
        
    Returns:
        DataFrame with added multiclass target variable
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Calculate future return
    df[f'Future_Return_{horizon}d'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
    
    # Use default thresholds if not provided
    if thresholds is None:
        thresholds = [-0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05]
    
    # Create conditions for each class
    conditions = []
    for i in range(len(thresholds)):
        if i == 0:
            # First class: below first threshold
            conditions.append(df[f'Future_Return_{horizon}d'] < thresholds[i])
        elif i == len(thresholds):
            # Last class: above last threshold
            conditions.append(df[f'Future_Return_{horizon}d'] >= thresholds[i-1])
        else:
            # Middle classes: between thresholds
            conditions.append(
                (df[f'Future_Return_{horizon}d'] >= thresholds[i-1]) & 
                (df[f'Future_Return_{horizon}d'] < thresholds[i])
            )
    
    # Add the last condition for values above the last threshold
    conditions.append(df[f'Future_Return_{horizon}d'] >= thresholds[-1])
    
    # Create class values (0 to n)
    values = list(range(len(conditions)))
    
    # Create target variable
    df[f'Target_MC_{horizon}d'] = np.select(conditions, values)
    
    # Print class distribution
    class_counts = df[f'Target_MC_{horizon}d'].value_counts().sort_index()
    total_samples = len(df)
    
    print(f"\nClass distribution for horizon {horizon}:")
    for class_idx, count in class_counts.items():
        percentage = (count / total_samples) * 100
        label = CLASS_LABELS.get(class_idx, f"Class {class_idx}")
        print(f"{label}: {count} samples ({percentage:.2f}%)")
    
    # Drop rows with NaN in target
    df = df.dropna(subset=[f'Target_MC_{horizon}d'])
    
    return df

def handle_class_imbalance(X, y, sampling_strategy=None):
    """
    Handle class imbalance using SMOTE.
    
    Args:
        X: Feature matrix
        y: Target vector
        sampling_strategy: Sampling strategy for SMOTE (optional)
        
    Returns:
        Resampled X and y
    """
    print("\nOriginal class distribution:")
    original_counts = Counter(y)
    for class_idx, count in sorted(original_counts.items()):
        label = CLASS_LABELS.get(class_idx, f"Class {class_idx}")
        percentage = (count / len(y)) * 100
        print(f"{label}: {count} samples ({percentage:.2f}%)")
    
    # Apply SMOTE
    if sampling_strategy is None:
        # Auto-determine sampling strategy
        # For classes with very few samples, set a minimum
        min_samples = 50
        sampling_strategy = {}
        
        for class_idx, count in sorted(original_counts.items()):
            # For minority classes, increase to at least min_samples
            # For majority classes, keep as is
            if count < min_samples:
                sampling_strategy[class_idx] = min_samples
        
        # If no adjustments needed, use 'auto'
        if not sampling_strategy:
            sampling_strategy = 'auto'
    
    try:
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print("\nResampled class distribution:")
        resampled_counts = Counter(y_resampled)
        for class_idx, count in sorted(resampled_counts.items()):
            label = CLASS_LABELS.get(class_idx, f"Class {class_idx}")
            percentage = (count / len(y_resampled)) * 100
            print(f"{label}: {count} samples ({percentage:.2f}%)")
        
        return X_resampled, y_resampled
    except ValueError as e:
        print(f"Warning: SMOTE failed with error: {e}")
        print("Proceeding with original data.")
        return X, y

def calculate_class_weights(y):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y: Target vector
        
    Returns:
        Dictionary of class weights
    """
    # Count samples in each class
    class_counts = Counter(y)
    total_samples = len(y)
    n_classes = len(class_counts)
    
    # Calculate weights
    weights = {}
    for class_idx, count in class_counts.items():
        # Inverse frequency weighting
        weights[class_idx] = total_samples / (n_classes * count)
    
    print("\nClass weights:")
    for class_idx, weight in sorted(weights.items()):
        label = CLASS_LABELS.get(class_idx, f"Class {class_idx}")
        print(f"{label}: {weight:.4f}")
    
    return weights

def evaluate_multiclass_prediction(y_true, y_pred, y_proba=None):
    """
    Evaluate multiclass prediction with specialized metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Calculate directional accuracy (up vs down)
    # Classes 0-3 are down, 4-7 are up
    y_true_binary = (y_true >= 4).astype(int)
    y_pred_binary = (y_pred >= 4).astype(int)
    directional_accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Calculate magnitude error
    # This measures how far off the predictions are in terms of class distance
    magnitude_error = np.mean(np.abs(y_true - y_pred))
    
    # Calculate weighted error based on financial impact
    # Errors in extreme classes (0 and 7) are more costly
    class_distances = np.abs(y_true - y_pred)
    financial_weights = np.ones_like(class_distances)
    
    # Assign higher weights to errors in extreme classes
    for i in range(len(y_true)):
        true_class = y_true[i]
        pred_class = y_pred[i]
        
        # If true class is extreme (0 or 7) and prediction is far off
        if (true_class == 0 or true_class == 7) and class_distances[i] > 2:
            financial_weights[i] = 2.0
        # If prediction is extreme but true is not
        elif (pred_class == 0 or pred_class == 7) and (true_class != pred_class):
            financial_weights[i] = 1.5
    
    weighted_error = np.mean(class_distances * financial_weights)
    
    # Calculate adjacent accuracy (prediction within ±1 class)
    adjacent_correct = np.sum(class_distances <= 1)
    adjacent_accuracy = adjacent_correct / len(y_true)
    
    # Return all metrics
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'directional_accuracy': directional_accuracy,
        'magnitude_error': magnitude_error,
        'weighted_error': weighted_error,
        'adjacent_accuracy': adjacent_accuracy,
        'confusion_matrix': cm,
        'classification_report': report
    }

def plot_multiclass_confusion_matrix(y_true, y_pred, title="Multiclass Confusion Matrix", output_file=None):
    """
    Plot confusion matrix for multiclass prediction with class labels.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        title: Plot title
        output_file: Output file path (optional)
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create class labels for the plot
    class_labels = [CLASS_LABELS.get(i, f"Class {i}") for i in range(len(np.unique(np.concatenate([y_true, y_pred]))))]
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        print(f"Saved confusion matrix plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_class_distribution(y, title="Class Distribution", output_file=None):
    """
    Plot class distribution.
    
    Args:
        y: Target vector
        title: Plot title
        output_file: Output file path (optional)
    """
    # Count samples in each class
    class_counts = Counter(y)
    
    # Create class labels
    class_labels = [CLASS_LABELS.get(i, f"Class {i}") for i in sorted(class_counts.keys())]
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(sorted(class_counts.keys())), y=list(class_counts.values()))
    plt.xticks(range(len(class_labels)), class_labels, rotation=45, ha='right')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title(title)
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        print(f"Saved class distribution plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def plot_multiclass_roc_curve(y_true, y_proba, title="Multiclass ROC Curve", output_file=None):
    """
    Plot ROC curve for multiclass prediction (one-vs-rest).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        title: Plot title
        output_file: Output file path (optional)
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    # Binarize the labels
    n_classes = len(np.unique(y_true))
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(12, 10))
    
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'{CLASS_LABELS.get(i, f"Class {i}")} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        print(f"Saved ROC curve plot to {output_file}")
    else:
        plt.show()
    
    plt.close()

def analyze_financial_impact(y_true, y_pred, future_returns):
    """
    Analyze financial impact of predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        future_returns: Actual future returns
        
    Returns:
        Dictionary with financial impact metrics
    """
    # Create a DataFrame for analysis
    df = pd.DataFrame({
        'true_class': y_true,
        'pred_class': y_pred,
        'future_return': future_returns
    })
    
    # Calculate metrics
    
    # 1. Average return when prediction is correct
    correct_predictions = df[df['true_class'] == df['pred_class']]
    avg_return_correct = correct_predictions['future_return'].mean() if len(correct_predictions) > 0 else 0
    
    # 2. Average return when prediction is wrong
    wrong_predictions = df[df['true_class'] != df['pred_class']]
    avg_return_wrong = wrong_predictions['future_return'].mean() if len(wrong_predictions) > 0 else 0
    
    # 3. Average return by predicted class
    avg_return_by_class = df.groupby('pred_class')['future_return'].mean().to_dict()
    
    # 4. Confusion matrix with average returns
    cm_returns = np.zeros((8, 8))
    for true_class in range(8):
        for pred_class in range(8):
            subset = df[(df['true_class'] == true_class) & (df['pred_class'] == pred_class)]
            if len(subset) > 0:
                cm_returns[true_class, pred_class] = subset['future_return'].mean()
    
    # 5. Simulated trading strategy
    # Assume we go long when predicted class >= 4 (up) and short when < 4 (down)
    df['position'] = np.where(df['pred_class'] >= 4, 1, -1)
    df['strategy_return'] = df['position'] * df['future_return']
    
    strategy_return = df['strategy_return'].mean()
    strategy_sharpe = df['strategy_return'].mean() / df['strategy_return'].std() if df['strategy_return'].std() > 0 else 0
    
    # 6. Accuracy by return magnitude
    # Group future returns into bins
    df['return_bin'] = pd.cut(df['future_return'], bins=[-np.inf, -0.05, -0.03, -0.01, 0, 0.01, 0.03, 0.05, np.inf], 
                             labels=range(8))
    
    accuracy_by_return = df.groupby('return_bin').apply(
        lambda x: accuracy_score(x['true_class'], x['pred_class']) if len(x) > 0 else 0
    ).to_dict()
    
    return {
        'avg_return_correct': avg_return_correct,
        'avg_return_wrong': avg_return_wrong,
        'avg_return_by_class': avg_return_by_class,
        'confusion_matrix_returns': cm_returns,
        'strategy_return': strategy_return,
        'strategy_sharpe': strategy_sharpe,
        'accuracy_by_return': accuracy_by_return
    }

def plot_financial_impact_heatmap(cm_returns, title="Average Returns by Prediction", output_file=None):
    """
    Plot heatmap of average returns for each true/predicted class combination.
    
    Args:
        cm_returns: Confusion matrix with average returns
        title: Plot title
        output_file: Output file path (optional)
    """
    # Create class labels
    class_labels = [CLASS_LABELS.get(i, f"Class {i}") for i in range(cm_returns.shape[0])]
    
    # Create plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_returns, annot=True, fmt='.4f', cmap='RdYlGn', center=0,
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file)
        print(f"Saved financial impact heatmap to {output_file}")
    else:
        plt.show()
    
    plt.close()

def train_multiclass_xgboost(X_train, y_train, X_val, y_val, params=None):
    """
    Train XGBoost model for multiclass prediction.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        params: XGBoost parameters (optional)
        
    Returns:
        Trained XGBoost model
    """
    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    
    # Set default parameters if not provided
    if params is None:
        params = {
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
            'num_class': len(np.unique(y_train))
        }
    else:
        # Ensure num_class is set
        params['num_class'] = len(np.unique(y_train))
        params['objective'] = 'multi:softprob'
        params['eval_metric'] = 'mlogloss'
    
    # Create DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=[class_weights[y] for y in y_train])
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Evaluate model
    y_pred_proba = model.predict(dval)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate evaluation metrics
    eval_metrics = evaluate_multiclass_prediction(y_val, y_pred, y_pred_proba)
    
    # Print evaluation results
    print("\nMulticlass model evaluation:")
    print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"F1 Score (Macro): {eval_metrics['f1_macro']:.4f}")
    print(f"F1 Score (Weighted): {eval_metrics['f1_weighted']:.4f}")
    print(f"Directional Accuracy: {eval_metrics['directional_accuracy']:.4f}")
    print(f"Magnitude Error: {eval_metrics['magnitude_error']:.4f}")
    print(f"Weighted Error: {eval_metrics['weighted_error']:.4f}")
    print(f"Adjacent Accuracy: {eval_metrics['adjacent_accuracy']:.4f}")
    
    return model, eval_metrics, y_pred, y_pred_proba

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
    
    # Generate synthetic returns
    returns = np.random.normal(0, 0.02, n_samples)
    
    # Create multiclass labels based on returns
    conditions = [
        returns < -0.05,
        (returns >= -0.05) & (returns < -0.03),
        (returns >= -0.03) & (returns < -0.01),
        (returns >= -0.01) & (returns < 0),
        (returns >= 0) & (returns < 0.01),
        (returns >= 0.01) & (returns < 0.03),
        (returns >= 0.03) & (returns < 0.05),
        returns >= 0.05
    ]
    
    values = list(range(8))
    y = np.select(conditions, values)
    
    # Split data
    train_size = int(0.6 * n_samples)
    val_size = int(0.2 * n_samples)
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    returns_train = returns[:train_size]
    
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    returns_val = returns[train_size:train_size+val_size]
    
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    returns_test = returns[train_size+val_size:]
    
    # Convert to pandas DataFrame
    X_train_df = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(n_features)])
    X_val_df = pd.DataFrame(X_val, columns=[f'feature_{i}' for i in range(n_features)])
    X_test_df = pd.DataFrame(X_test, columns=[f'feature_{i}' for i in range(n_features)])
    
    y_train_df = pd.Series(y_train)
    y_val_df = pd.Series(y_val)
    y_test_df = pd.Series(y_test)
    
    # Plot class distribution
    plot_class_distribution(y_train_df, title="Training Set Class Distribution", output_file="train_class_dist.png")
    
    # Handle class imbalance
    X_train_resampled, y_train_resampled = handle_class_imbalance(X_train_df, y_train_df)
    
    # Train multiclass XGBoost model
    model, eval_metrics, y_val_pred, y_val_pred_proba = train_multiclass_xgboost(
        X_train_resampled, y_train_resampled, X_val_df, y_val_df
    )
    
    # Plot confusion matrix
    plot_multiclass_confusion_matrix(y_val_df, y_val_pred, 
                                    title="Validation Set Confusion Matrix",
                                    output_file="val_confusion_matrix.png")
    
    # Plot ROC curve
    plot_multiclass_roc_curve(y_val_df, y_val_pred_proba,
                             title="Validation Set ROC Curve",
                             output_file="val_roc_curve.png")
    
    # Analyze financial impact
    financial_impact = analyze_financial_impact(y_val_df, y_val_pred, returns_val)
    
    # Plot financial impact heatmap
    plot_financial_impact_heatmap(financial_impact['confusion_matrix_returns'],
                                 title="Average Returns by Prediction (Validation Set)",
                                 output_file="val_financial_impact.png")
    
    print("\nFinancial Impact Analysis:")
    print(f"Average Return (Correct Predictions): {financial_impact['avg_return_correct']:.4f}")
    print(f"Average Return (Wrong Predictions): {financial_impact['avg_return_wrong']:.4f}")
    print(f"Strategy Return: {financial_impact['strategy_return']:.4f}")
    print(f"Strategy Sharpe Ratio: {financial_impact['strategy_sharpe']:.4f}")
    
    print("\nAverage Return by Predicted Class:")
    for class_idx, avg_return in sorted(financial_impact['avg_return_by_class'].items()):
        label = CLASS_LABELS.get(class_idx, f"Class {class_idx}")
        print(f"{label}: {avg_return:.4f}")
