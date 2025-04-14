# evaluation/metrics.py

import numpy as np
import pandas as pd

def calculate_returns(predictions, actual_returns):
    positions = 2 * predictions - 1  # 0 → -1 (short), 1 → +1 (long)
    return positions * actual_returns

def calculate_cumulative_returns(strategy_returns, initial_capital=10000):
    return (1 + pd.Series(strategy_returns)).cumprod() * initial_capital

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess_returns = returns - risk_free_rate / periods_per_year
    std = np.std(excess_returns)
    if std == 0:
        return 0
    return np.sqrt(periods_per_year) * np.mean(excess_returns) / std

def calculate_max_drawdown(returns):
    cumulative = (1 + pd.Series(returns)).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_win_rate(returns):
    return np.mean(np.array(returns) > 0)

def evaluate_trading_strategy(predictions, actual_returns, dates=None):
    strategy_returns = calculate_returns(predictions, actual_returns)
    strategy_returns = pd.Series(strategy_returns, index=dates) if dates is not None else pd.Series(strategy_returns)

    metrics = {
        'cumulative_return': (1 + strategy_returns).prod() - 1,
        'annualized_return': (1 + strategy_returns).prod() ** (252 / len(strategy_returns)) - 1,
        'sharpe_ratio': calculate_sharpe_ratio(strategy_returns),
        'max_drawdown': calculate_max_drawdown(strategy_returns),
        'win_rate': calculate_win_rate(strategy_returns),
        'total_trades': len(strategy_returns)
    }

    return metrics, strategy_returns
