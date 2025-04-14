# Stock Market Prediction Pipeline

This pipeline allows you to train and evaluate machine learning models to predict stock price movements and implement trading strategies (long/short positions) based on those predictions.

## Features

- Train individual models for each stock
- Support for multiple model types (Logistic Regression, Random Forest, XGBoost, LSTM)
- Date range filtering for specific time periods
- Automatic calculation of technical indicators
- Trading strategy evaluation with performance metrics
- Visualization of results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/3244StockMarket.git
cd 3244StockMarket
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

The pipeline uses a hierarchical structure to organize models and results by both stock and model type:

```
3244STOCKMARKET/
├── models/
│   ├── AAPL/                  # Stock-specific folder
│   │   ├── feature_cols.pkl   # Features for this stock
│   │   ├── logistic/          # Model-specific subfolder
│   │   │   └── model.pkl      # The logistic model for AAPL
│   │   ├── rf/                # Another model type
│   │   │   └── model.pkl
│   │   └── xgb/               # Another model type
│   │       └── model.pkl
│   ├── MSFT/                  # Another stock
│   │   └── ...
│   └── training_info.pkl      # Overall training info
├── results/
│   ├── AAPL/                  # Stock-specific results
│   │   ├── logistic/          # Model-specific results
│   │   │   ├── confusion_matrix.png
│   │   │   ├── cumulative_returns.png
│   │   │   └── metrics.json
│   │   ├── rf/
│   │   └── xgb/
│   ├── MSFT/
│   │   └── ...
│   └── logistic_metrics.csv   # Summary metrics across stocks
└── config/
    ├── AAPL/                  # Stock-specific configs
    │   ├── logistic/
    │   │   └── params.json
    │   └── ...
    └── MSFT/
```

## Usage

The main entry point is `src/run.py`, which orchestrates the entire pipeline.

### Basic Command Structure

```bash
python src/run.py --mode [mode] --model [model_type] --stocks [stock_symbols] [options]
```

### Mode Options

- `train`: Only train models
- `tune`: Only tune hyperparameters
- `evaluate`: Only evaluate models
- `all`: Run the full pipeline (tune, train, evaluate)

### Model Options

- `logistic`: Logistic Regression
- `rf`: Random Forest
- `xgb`: XGBoost
- `lstm`: LSTM Neural Network
- `attention`: Attention-based LSTM
- `all`: Use all model types

### Stock Selection

You can specify one or more stock symbols:
```bash
python src/run.py --stocks AAPL MSFT GOOGL
```

### Date Range Options

You can filter data by date range in three ways:

1. Specific year:
```bash
python src/run.py --year 2016
```

2. Custom date range:
```bash
python src/run.py --start_date 2016-01-01 --end_date 2016-12-31
```

3. Default: Uses all available data

### Test Size Option

You can specify the proportion of data to use for testing:
```bash
python src/run.py --test_size 0.3
```

## Examples

### Train and Evaluate a Logistic Regression Model for Apple in 2016

```bash
python src/run.py --mode all --model logistic --stocks AAPL --year 2016
```

### Tune Hyperparameters for XGBoost Models for Multiple Stocks

```bash
python src/run.py --mode tune --model xgb --stocks AAPL MSFT GOOGL AMZN
```

### Train Models for All Stock Types

```bash
python src/run.py --mode train --model all --stocks AAPL MSFT GOOGL
```

### Train Models for Multiple Stocks in a Custom Date Range

```bash
python src/run.py --mode train --model logistic --stocks AAPL MSFT --start_date 2018-01-01 --end_date 2019-12-31
```

### Evaluate Previously Trained Models

```bash
python src/run.py --mode evaluate --model all
```

### Train and Evaluate One Model Type for All Stocks in Dataset

```bash
python src/run.py --mode all --model logistic
```

## Trading Strategy

For each model, the pipeline implements a simple trading strategy:
- If predicted return is positive → Take LONG position
- If predicted return is negative → Take SHORT position

Performance is measured based on cumulative returns and Sharpe ratio.

## Performance Metrics

The pipeline evaluates trading strategy performance using several metrics:

- **Cumulative Return**: Total return over the testing period
- **Annualized Return**: Annualized version of the cumulative return
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Maximum Drawdown**: Largest peak-to-trough decline (smaller is better)
- **Win Rate**: Percentage of trades that are profitable

## Extending the Pipeline

You can extend the pipeline by:
1. Adding new model types in the `src/models/` directory
2. Implementing new technical indicators in `src/data_loader.py`
3. Adding new evaluation metrics in `evaluation/metrics.py`

## Tips for Best Results

1. Train models on specific stocks rather than combining data from multiple stocks
2. Use sufficient historical data to capture market cycles
3. Ensure your test set represents a realistic trading period
4. Compare multiple model types to find the best performer for each stock
5. Pay attention to the Sharpe ratio, not just raw returns
6. For hyperparameter tuning, use multiple years of data when possible
7. Consider different time periods for different market conditions (bull/bear markets)
