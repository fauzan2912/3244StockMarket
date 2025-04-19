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

if __name__ == "__main__":
    df = load_metrics()
    if df.empty:
        print("No metrics found.")
    else:
        summary = summarize_metrics(df)
        print("\n=== Model Performance Summary ===")
        print(summary.round(4))
        summary.to_csv("summary.csv")
        print("\n[✓] Saved summary to model_comparison_summary.csv")

        # Identify best model for each metric
        print("\n=== Best Model per Metric ===")
        for metric in summary.columns:
            best_model = summary[metric].idxmax()
            best_value = summary[metric].max()
            print(f"{metric}: {best_model} (score: {best_value:.4f})")
