# XGBoost Stock Market Prediction

This repository contains optimized XGBoost models for stock market prediction using various advanced techniques.

## Overview

This project implements and evaluates several optimization strategies for XGBoost-based stock market prediction:

1. **Rolling vs Expanding Window Comparison**
   - Rolling window: Trains on the most recent N days and slides forward
   - Expanding window: Initially trains on the first N days and expands the training set

2. **Multiclass Prediction Approach**
   - Enhanced prediction of price movement magnitude instead of just binary up/down
   - Classes: Down >5%, Down 3-5%, Down 1-3%, Down <1%, Up <1%, Up 1-3%, Up 3-5%, Up >5%

3. **Enhanced Feature Engineering**
   - Technical indicators
   - Market regime features
   - Volatility-based features
   - Seasonality features
   - Statistical features

4. **Stock Correlation Features**
   - Cross-stock correlation features
   - Lead-lag relationship detection
   - Correlation regime detection

5. **Hyperparameter Optimization**
   - Comprehensive hyperparameter tuning
   - Financial time series-specific optimization

## Dataset

The code is designed to work with the Huge Stock Market Dataset from Kaggle:
https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs

This dataset provides data for about 8.4K companies spanning from 2009 to 2017, for all US-based stocks and ETFs trading on the NYSE, NASDAQ, and NYSE MKT.

## Files

- `xgboost_optimization_all_in_one.py`: Complete implementation in a single file
- `fix_paths.py`: Utility to fix path issues for cross-platform compatibility
- `enhanced_features_fixed.py`: Advanced feature engineering module
- `correlation_analysis_fixed.py`: Stock correlation analysis
- `model_evaluation_fixed.py`: Comprehensive model evaluation
- `optimization_results.md`: Detailed results and findings

## Usage

### Quick Start

For the simplest approach, use the all-in-one script:

```bash
python xgboost_optimization_all_in_one.py
```

This script contains all necessary components and will run a complete analysis with default settings.

### Step-by-Step Approach

Alternatively, you can run the individual components:

1. First, fix any path issues:
```bash
python fix_paths.py
```

2. Run the enhanced feature engineering:
```bash
python enhanced_features_fixed.py
```

3. Analyze stock correlations:
```bash
python correlation_analysis_fixed.py
```

4. Run the comprehensive model evaluation:
```bash
python model_evaluation_fixed.py
```

## Customization

You can customize the analysis by modifying these variables in the scripts:

- `STOCKS`: List of stock symbols to analyze
- `HORIZONS`: List of prediction horizons in days
- `DATA_DIR`: Directory containing stock data files
- `RESULTS_DIR`: Directory to save results

## Key Findings

- **Optimal Window Strategy by Horizon**
  - For short-term predictions (1-7 days): Use rolling window approach
  - For long-term predictions (30-252 days): Use expanding window approach

- **Prediction Type Selection**
  - For maximum accuracy: Use binary prediction
  - For maximum financial performance: Use multiclass prediction
  - For user-facing applications: Multiclass provides more actionable insights

- **Feature Engineering Impact**
  - Technical indicators and correlation features provide the most significant improvements
  - Market regime features are particularly valuable for longer horizons

- **Hyperparameter Optimization**
  - Different horizons benefit from different hyperparameter settings
  - Short-term predictions: Lower max_depth, higher learning rate
  - Long-term predictions: Higher max_depth, lower learning rate

## Requirements

- Python 3.6+
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn
- networkx (for correlation network visualization)

## License

MIT License
