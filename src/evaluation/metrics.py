import numpy as np
import pandas as pd

def calculate_returns(predictions, actual_returns):
    """
    Calculate returns from binary predictions
    - Long position (1): Buy the asset
    - Short position (0): Sell the asset
    
    Args:
        predictions: Binary predictions (0 or 1)
        actual_returns: Actual percentage returns
        
    Returns:
        strategy_returns: Returns from the trading strategy
    """
    # Convert 0/1 to -1/1 for short/long positions
    position = 2 * predictions - 1
    
    # Calculate strategy returns (position * actual return)
    strategy_returns = position * actual_returns
    
    return strategy_returns

def calculate_cumulative_returns(strategy_returns, initial_capital=10000):
    """Calculate cumulative returns from strategy returns"""
    cumulative_returns = (1 + strategy_returns).cumprod() * initial_capital
    return cumulative_returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """Calculate the annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()

def calculate_max_drawdown(returns):
    """Calculate maximum drawdown"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

def calculate_win_rate(returns):
    """Calculate win rate (percentage of positive returns)"""
    return (returns > 0).mean()

def evaluate_trading_strategy(predictions, actual_returns, dates=None):
    """
    Evaluate a trading strategy based on predictions
    
    Args:
        predictions: Binary predictions (0 or 1)
        actual_returns: Actual percentage returns
        dates: Optional dates for the returns data
        
    Returns:
        metrics: Dictionary of performance metrics
    """
    # Calculate strategy returns
    strategy_returns = calculate_returns(predictions, actual_returns)
    
    # Create a Series with dates if provided
    if dates is not None:
        strategy_returns = pd.Series(strategy_returns, index=dates)
    
    # Calculate metrics
    metrics = {
        'cumulative_return': (1 + strategy_returns).prod() - 1,
        'annualized_return': (1 + strategy_returns).prod() ** (252 / len(strategy_returns)) - 1,
        'sharpe_ratio': calculate_sharpe_ratio(strategy_returns),
        'max_drawdown': calculate_max_drawdown(strategy_returns),
        'win_rate': calculate_win_rate(strategy_returns),
        'total_trades': len(strategy_returns)
    }
    
    return metrics, strategy_returns