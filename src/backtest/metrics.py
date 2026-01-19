import numpy as np
import pandas as pd

def sharpe_ratio(returns, risk_free=0.0):
    """Annualized Sharpe Ratio"""
    excess_returns = returns - risk_free
    if returns.std() == 0:
        return 0
    return (excess_returns.mean() / returns.std()) * np.sqrt(252)

def max_drawdown(cumulative_returns):
    """Calculates Maximum Drawdown from peak."""
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()

def win_rate(returns):
    """Percent of days with positive returns"""
    return len(returns[returns > 0]) / len(returns)

def information_coefficient(predictions, targets):
    """Spearman Rank Correlation (IC) - Key for Factor Models"""
    return pd.Series(predictions).corr(pd.Series(targets), method='spearman')