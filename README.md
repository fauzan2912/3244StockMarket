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
git clone https://github.com/fauzan2912/3244StockMarket
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
python main.py --mode all --stocks AAPL MSFT BAC CSCO F GE HPQ INTC PFE SIRI --model all --start_date 2010-01-01 --end_date 2017-12-31
```

### Run with specific models:
```bash
python main.py --mode all --stocks AAPL MSFT ... --model logistic rf lstm --start_date ...
```

### Required Arguments:
| Argument       | Description                                |
|----------------|--------------------------------------------|
| `--stocks`     | One or more stock tickers (e.g. `AAPL`)    |
| `--model`      | One or more models: `logistic`, `rf`, `svm`, `xgb`, `lstm`, `attention`, `deep_rnn`, or `all` |
| `--start_date` | Start date of your dataset window          |
| `--end_date`   | End date of your dataset window            |

---

## 📁 Output Directory: `results/{stock}/`

Each stock’s results are saved in a dedicated folder:
```
results/
└── AAPL/
    ├── logistic
         ├── model_2012-01_rolling.pkl
         ├── features_2012-01_rolling.pkl
         ├── metrics_2012-01_rolling.json
         ├── metrics_2012-01_expanding.json
         ├── AAPL_svm_full_comparison.png      ← Final performance chart
         └── AAPL_svm_metrics_summary.csv      ← All metrics across windows
```

### Fetch the aggregated outputs:
```bash
python performance.py
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
- Window expands every 1 year

## 📈 Evaluation Metrics
Each strategy is evaluated using:

### 🧮 Financial Metrics
- **Cumulative Return**
- **Annualized Return**
- **Sharpe Ratio**
- **Maximum Drawdown**
- **Win Rate**
- **Total Trades**

### 📊 Machine Learning Metrics
- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

All metrics are saved in `metrics.json` for each stock/model/year combination, and summarized via `performance.py`.

These include both finance-focused (e.g. Sharpe, drawdown) and ML-focused (e.g. F1, precision) metrics, allowing multi-dimensional comparison across strategies.

---

## ⚙️ Models

| Model Name     | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `logistic`     | Logistic Regression — simple and interpretable baseline                     |
| `rf`           | Random Forest Classifier — non-linear ensemble with strong baseline accuracy|
| `svm`          | Support Vector Machine — margin-based classifier with kernel support        |
| `xgb`          | XGBoost Classifier — powerful gradient-boosted tree model                   |
| `lstm`         | LSTM (Long Short-Term Memory) — recurrent neural net for temporal patterns  |
| `attention`    | Attention-based LSTM — enhanced LSTM with attention for time-step weighting |
| `deep_rnn`     | Multi-layer RNN — a deep sequential model using stacked recurrent layers    |

---

## 📁 Project Structure Highlights

- `main.py` – CLI runner for batch training
- `src/core/` – training and window strategy logic
- `src/models/` – individual model implementations (e.g., `model_lstm.py`, `model_svm.py`)
- `src/tuning/` – hyperparameter tuning logic per model
- `evaluation/` – metric calculators and visualizers
- `results/` – saved models, predictions, and metric outputs

---

## 📊 Plot

Final result is saved as:
```
results/{stock}/{stock}_{model}_full_comparison.png
```

It compares:
- 🔵 Rolling strategy
- 🟠 Expanding strategy
- 🟢 Buy & Hold benchmark

---

## 📌 Interpretation
- A higher Sharpe Ratio indicates better risk-adjusted returns.
- The model with the highest average F1 Score may be most reliable in directional classification.
- Use `performance.py` to aggregate and rank models by strategy.

## 📌 Notes

- Models are tuned per-window using a time-respecting 80/20 split
- Metrics are stored per window and summarized in a CSV
- No "latest" suffixes are used — all files are clearly versioned

---

## 📊 Dashboard Visualization (Streamlit)

Users can explore model predictions, strategy comparisons, and cumulative returns through an interactive dashboard.

### 1. Install Streamlit
```bash
pip install streamlit
```

### 2. Run the dashboard
```bash
streamlit run app.py
```

This will launch a local server where users can visually compare:
- Rolling vs Expanding vs Buy & Hold strategies
- Model-wise performance metrics

--- 

## 🧼 Optional Enhancements

- Add new models to `models/` and register in `model_factory.py`
- Customize technical indicators in `processor.py`
- Add multi-stock cross-analysis in `results/summary/`

---

## 👨‍💻 Authors

This pipeline was built as part of a CS3244 Machine Learning project focused on real-world financial forecasting.
