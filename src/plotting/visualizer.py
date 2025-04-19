# src/plotting/visualizer.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from evaluation.metrics import calculate_returns, calculate_cumulative_returns

def plot_cumulative_returns(model_predictions, actual_returns, dates=None, initial_capital=10000, save_path=None):
    plt.figure(figsize=(12, 6))

    benchmark = pd.Series(actual_returns, index=dates if dates is not None else range(len(actual_returns)))
    benchmark_cum = (1 + benchmark).cumprod() * initial_capital
    plt.plot(benchmark_cum, label="Buy & Hold", color="black", linestyle="--")

    for label, preds in model_predictions.items():
        strat_returns = calculate_returns(preds, actual_returns)
        strat_series = pd.Series(strat_returns, index=dates if dates is not None else range(len(preds)))
        cum = calculate_cumulative_returns(strat_series, initial_capital)
        plt.plot(cum, label=label)

    plt.title("Cumulative Returns")
    plt.xlabel("Date" if dates is not None else "Step")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Model", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sell (0)', 'Buy (1)'],
                yticklabels=['Sell (0)', 'Buy (1)'])
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()

import numpy as np

def plot_comparison_cumulative_returns(rolling_preds, expanding_preds, actual_returns, dates, strategy_labels, title, save_name):
    # Ensure inputs are arrays
    rolling_preds = np.array(rolling_preds)
    expanding_preds = np.array(expanding_preds)
    actual_returns = np.array(actual_returns)

    import matplotlib.pyplot as plt
    import pandas as pd
    from evaluation.metrics import calculate_returns, calculate_cumulative_returns

    # Calculate returns
    rolling_returns = calculate_returns(rolling_preds, actual_returns)
    expanding_returns = calculate_returns(expanding_preds, actual_returns)
    buy_hold_returns = actual_returns

    # Cumulative values
    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        strategy_labels[0]: calculate_cumulative_returns(rolling_returns),
        strategy_labels[1]: calculate_cumulative_returns(expanding_returns),
        'Buy & Hold': calculate_cumulative_returns(buy_hold_returns)
    })

    df.set_index('Date').plot(
        figsize=(12, 6),
        title=title,
        lw=2,
        linestyle='-'
    )

    plt.ylabel("Portfolio Value")
    plt.xlabel("Date")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Save inside results/{stock}/{model}/
    stock, model = save_name.split("_")[:2]  # e.g., "AAPL_svm_full_comparison.png"
    save_dir = os.path.join("results", stock, model)
    os.makedirs(save_dir, exist_ok=True)

    result_path = os.path.join(save_dir, save_name)

    plt.savefig(result_path, dpi=300)
    plt.close()
    print(f"[✓] Saved comparison plot: {result_path}")

