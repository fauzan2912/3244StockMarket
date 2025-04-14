import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from evaluation.metrics import calculate_cumulative_returns, calculate_returns

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
        strategy_returns = calculate_returns(predictions, actual_returns)
        
        # Convert to Series with dates if provided
        if dates is not None:
            strategy_returns = pd.Series(strategy_returns, index=dates)
        
        # Calculate cumulative returns
        cum_returns = calculate_cumulative_returns(strategy_returns, initial_capital)
        
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

def plot_model_comparison(metrics_df, save_path=None):
    """
    Plot bar charts comparing model performance metrics
    
    Args:
        metrics_df: DataFrame with model metrics
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot Sharpe Ratio
    sns.barplot(x=metrics_df.index, y="sharpe_ratio", data=metrics_df, ax=axes[0])
    axes[0].set_title("Sharpe Ratio")
    axes[0].grid(True, alpha=0.3)
    
    # Plot Cumulative Return
    sns.barplot(x=metrics_df.index, y="cumulative_return", data=metrics_df, ax=axes[1])
    axes[1].set_title("Cumulative Return")
    axes[1].grid(True, alpha=0.3)
    
    # Plot Max Drawdown
    sns.barplot(x=metrics_df.index, y="max_drawdown", data=metrics_df, ax=axes[2])
    axes[2].set_title("Maximum Drawdown")
    axes[2].grid(True, alpha=0.3)
    
    # Plot Win Rate
    sns.barplot(x=metrics_df.index, y="win_rate", data=metrics_df, ax=axes[3])
    axes[3].set_title("Win Rate")
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, model_name, save_path=None):
    """
    Plot confusion matrix for binary classification
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Optional path to save the figure
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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