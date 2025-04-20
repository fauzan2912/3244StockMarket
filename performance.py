import os
import json
import pandas as pd

RESULTS_DIR = "results"

def load_metrics(results_dir="results"):
    records = []

    for stock in os.listdir(results_dir):
        stock_path = os.path.join(results_dir, stock)
        if not os.path.isdir(stock_path):
            continue

        for model in os.listdir(stock_path):
            model_path = os.path.join(stock_path, model)
            if not os.path.isdir(model_path):
                continue

            for file in os.listdir(model_path):
                if file.startswith("metrics_") and file.endswith(".json"):
                    path = os.path.join(model_path, file)
                    parts = file.replace("metrics_", "").replace(".json", "").split("_")
                    if len(parts) == 2:
                        date_str, window_type = parts
                        with open(path) as f:
                            try:
                                metrics = json.load(f)
                            except:
                                continue

                        record = {
                            "stock": stock,
                            "model": model,
                            "window_type": window_type,
                            "date": date_str
                        }
                        record.update(metrics)
                        records.append(record)

    return pd.DataFrame(records)

def summarize_metrics(df):
    grouped = df.groupby(['model']).agg({
        'sharpe_ratio': 'mean',
        'f1_score': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'accuracy': 'mean',
        'cumulative_return': 'mean',
        'annualized_return': 'mean',
        'max_drawdown': 'mean',
        'win_rate': 'mean'
    }).sort_values(by='sharpe_ratio', ascending=False)

    return grouped

def summarize_by_strategy(df):
    return df.groupby(['model', 'window_type']).agg({
        'sharpe_ratio': 'mean',
        'f1_score': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'accuracy': 'mean',
        'cumulative_return': 'mean',
        'annualized_return': 'mean',
        'max_drawdown': 'mean',
        'win_rate': 'mean'
    }).sort_values(by='sharpe_ratio', ascending=False)

if __name__ == "__main__":
    df = load_metrics()
    if df.empty:
        print("No metrics found.")
    else:
        summary = summarize_metrics(df)
        print("\n=== Overall Model Performance Summary ===")
        print(summary.round(4))
        summary.to_csv("summary.csv")
        print("[✓] Saved overall summary to summary.csv")

        print("\n=== Overall Best Model per Metric ===")
        for metric in summary.columns:
            best_model = summary[metric].idxmax()
            best_value = summary[metric].max()
            print(f"{metric}: {best_model} (score: {best_value:.4f})")

        # Strategy-specific summary
        strat_summary = summarize_by_strategy(df)
        print("\n=== Overall Model Performance Summary by Strategy (Rolling vs Expanding) ===")
        print(strat_summary.round(4))
        strat_summary.to_csv("summary_by_strategy.csv")
        print("[✓] Saved strategy-specific summary to summary_by_strategy.csv")

        print("\n=== Overall Best Model + Strategy per Metric ===")
        for metric in strat_summary.columns:
            best_combo = strat_summary[metric].idxmax()
            best_value = strat_summary[metric].max()
            model, strategy = best_combo
            print(f"{metric}: {model} ({strategy}) (score: {best_value:.4f})")