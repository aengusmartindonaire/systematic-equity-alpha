import pandas as pd
import numpy as np

def calculate_rsi(series, window=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_volatility(series, window=20):
    """Rolling annualized volatility"""
    return series.pct_change().rolling(window=window).std() * np.sqrt(252)

def calculate_momentum(series, window=12):
    """Rate of Change / Momentum"""
    return series.pct_change(window)