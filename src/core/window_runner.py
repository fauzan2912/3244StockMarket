# src/core/window_runner.py

import os
import sys
import csv
import pandas as pd
from dateutil.relativedelta import relativedelta

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(ROOT_DIR)

from src.data.windows import prepare_time_window_data
from src.core.trainer import train_model
from src.core.evaluator import evaluate_model
from src.plotting.visualizer import plot_comparison_cumulative_returns

def generate_yearly_ranges(start_date, end_date, window_years=1, expanding=False):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    ranges = []
    test_start = start + relativedelta(years=window_years)

    while True:
        train_start = start if expanding else test_start - relativedelta(years=window_years)
        train_end = test_start - relativedelta(days=1)
        test_end = test_start + relativedelta(years=1) - relativedelta(days=1)

        if test_end > end:
            break

        ranges.append((train_start, train_end, test_start, test_end))
        test_start += relativedelta(years=1)

    return ranges

def save_combined_metrics_csv(stock, model, all_metrics):
    path = os.path.join("results", stock)
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{stock}_{model}_metrics_summary.csv")

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"[✓] Saved metrics summary: {filepath}")

def run_rolling_strategy(args):
    for stock in args.stocks:
        print(f"\n[START] Running combined strategy for {stock}")

        rolling_ranges = generate_yearly_ranges(args.start_date, args.end_date, args.window_size_years, expanding=False)
        expanding_ranges = generate_yearly_ranges(args.start_date, args.end_date, args.window_size_years, expanding=True)

        full_preds = {'rolling': [], 'expanding': [], 'returns': [], 'dates': []}
        all_window_metrics = []

        for i, ((roll_train_start, roll_train_end, test_start, test_end),
                (exp_train_start, exp_train_end, _, _)) in enumerate(zip(rolling_ranges, expanding_ranges)):

            print(f"\n[Window {i+1}] Test: {test_start.date()} → {test_end.date()}")
            print(f"  Rolling Train: {roll_train_start.date()} to {roll_train_end.date()}")
            print(f"  Expand  Train: {exp_train_start.date()} to {exp_train_end.date()}")

            roll_train_df, test_df = prepare_time_window_data(stock, roll_train_start, roll_train_end, test_start, test_end)
            exp_train_df, _ = prepare_time_window_data(stock, exp_train_start, exp_train_end, test_start, test_end)

            if test_df is None or test_df.empty or roll_train_df is None or exp_train_df is None:
                print("[SKIP] Incomplete or empty data")
                continue

            model_roll, features = train_model(args.model, stock, roll_train_df, meta=(test_start, 'rolling'))
            model_exp, _ = train_model(args.model, stock, exp_train_df, meta=(test_start, 'expanding'))

            print(f"[✓] Rolling Params ({test_start.strftime('%Y-%m')}): {model_roll.params}")
            print(f"[✓] Expanding Params ({test_start.strftime('%Y-%m')}): {model_exp.params}")

            metrics_r, _, preds_r = evaluate_model(args.model, model_roll, features, stock, test_df, return_preds=True, window_id=f"{test_start.strftime('%Y-%m')}_rolling")
            metrics_e, _, preds_e = evaluate_model(args.model, model_exp, features, stock, test_df, return_preds=True, window_id=f"{test_start.strftime('%Y-%m')}_expanding")


            full_preds['rolling'].extend(preds_r)
            full_preds['expanding'].extend(preds_e)
            full_preds['returns'].extend(test_df['Returns'].tolist())
            full_preds['dates'].extend(test_df['Date'].tolist())

            metrics_r.update({'strategy': 'rolling', 'date': test_start.strftime('%Y-%m')})
            metrics_e.update({'strategy': 'expanding', 'date': test_start.strftime('%Y-%m')})
            all_window_metrics.extend([metrics_r, metrics_e])

        save_combined_metrics_csv(stock, args.model, all_window_metrics)

        if (args.model == 'lstm'):
            continue;
        plot_comparison_cumulative_returns(
            rolling_preds=full_preds['rolling'],
            expanding_preds=full_preds['expanding'],
            actual_returns=full_preds['returns'],
            dates=full_preds['dates'],
            strategy_labels=['Rolling', 'Expanding'],
            title=f"{stock} {args.model.upper()} - Full Strategy Comparison",
            save_name=f"{stock}_{args.model}_full_comparison.png"
        )

        # Optional: clean up any "latest" files
        clean_latest_files(stock, args.model)

def clean_latest_files(stock, model):
    folder = os.path.join("results", stock)
    for f in os.listdir(folder):
        if f.endswith("latest.pkl") or f.endswith("latest.json"):
            os.remove(os.path.join(folder, f))
            print(f"[✗] Removed stale: {f}")
