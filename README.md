# 🧠 Stock Market Forecasting Pipeline

This project implements a **rolling vs expanding window forecasting pipeline** for stock price prediction using machine learning models (e.g., SVM, Logistic Regression, Random Forest).

## 📌 Features

- ✅ Yearly rolling & expanding window strategy
- ✅ Per-window hyperparameter tuning using validation splits
- ✅ Buy & Hold benchmark included
- ✅ Final cumulative performance plot (Rolling vs Expanding vs Buy & Hold)
- ✅ Per-window evaluation metrics (Sharpe, return, drawdown)
- ✅ Full CSV metrics summary per stock

---

## 🛠 Project Setup

### 1. Clone the project
```bash
git clone <your-repo-url>
cd 3244StockMarket
```

### 2. Install requirements
```bash
pip install -r requirements.txt
```

### 3. First-time run will auto-download Kaggle dataset:
- [`borismarjanovic/price-volume-data-for-all-us-stocks-etfs`](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)
- Data is cached and cleaned to form `stocks.pkl`

---

## 🚀 Usage

### Run the pipeline:
```bash
python main.py --stocks AAPL --model svm --start_date 2010-01-01 --end_date 2017-12-31
```

### Required Arguments:
| Argument       | Description                                |
|----------------|--------------------------------------------|
| `--stocks`     | One or more stock tickers (e.g. `AAPL`)    |
| `--model`      | Model to use: `svm`, `logistic`, `rf`      |
| `--start_date` | Start date of your dataset window          |
| `--end_date`   | End date of your dataset window            |

---

## 📁 Output Directory: `results/{stock}/`

Each stock’s results are saved in a dedicated folder:
```
results/
└── AAPL/
    ├── model_2012-01_rolling.pkl
    ├── features_2012-01_rolling.pkl
    ├── metrics_2012-01_rolling.json
    ├── metrics_2012-01_expanding.json
    ├── AAPL_svm_full_comparison.png      ← Final performance chart
    └── AAPL_svm_metrics_summary.csv      ← All metrics across windows
```

---

## 🔍 Strategy Details

### Rolling Window
- Train on a **fixed 1-year window**
- Predict the following **1-year**
- Window slides forward 1 year at a time

### Expanding Window
- Train from `start_date` up to the current test period
- Predict on 1-year ahead

### Evaluation Metrics
- ✅ Cumulative Return
- ✅ Sharpe Ratio
- ✅ Max Drawdown
- ✅ Win Rate
- ✅ Total Trades

---

## ⚙️ Models

| Model Type | Description |
|------------|-------------|
| `svm`      | Scikit-learn SVC with tuning over C, kernel, gamma |
| `logistic` | Logistic Regression with L2 penalty tuning |
| `rf`       | Random Forest with n_estimators, depth, and sampling tuning |

---

## 📊 Plot

Final result is saved as:
```
results/{stock}/{stock}_{model}_full_comparison.png
```

It compares:
- 🔵 Rolling strategy
- 🟢 Expanding strategy
- ⚫ Buy & Hold benchmark

---

## 📌 Notes

- Models are tuned per-window using a time-respecting 80/20 split
- Metrics are stored per window and summarized in a CSV
- No "latest" suffixes are used — all files are clearly versioned

---

## 🧼 Optional Enhancements

- Add new models to `models/` and register in `model_factory.py`
- Customize technical indicators in `processor.py`
- Add multi-stock cross-analysis in `results/summary/`

---

## 👨‍💻 Authors

This pipeline was built as part of a CS3244 Machine Learning project focused on real-world financial forecasting.

```

---
