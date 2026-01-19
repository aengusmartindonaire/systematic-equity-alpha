import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger()

class BacktestEngine:
    def __init__(self, config, strategy_class, data):
        """
        :param config: Dict containing configuration
        :param strategy_class: Instance of the strategy (e.g., XGBStrategy)
        :param data: DataFrame with features and targets
        """
        self.config = config
        self.strategy = strategy_class
        self.data = data
        self.results = []

    def run(self):
        """Runs the backtest simulation."""
        logger.info("Starting Backtest Simulation...")
        
        test_start = self.config['data']['test_start']
        test_end = self.config['data']['test_end']
        
        # Filter for test period
        test_data = self.data.loc[test_start:test_end].copy()
        
        if test_data.empty:
            logger.error("Test data is empty. Check dates in config.")
            return

        # Feature columns (exclude target and non-features)
        target_col = self.config['data']['target_col']
        ticker_col = self.config['data']['ticker_col']
        exclude_cols = [target_col, ticker_col]
        feature_cols = [c for c in test_data.columns if c not in exclude_cols]

        # Generate Signals (Predictions)
        logger.info(f"Predicting on {len(test_data)} rows...")
        signals = self.strategy.predict(test_data[feature_cols])
        
        test_data['signal'] = signals
        
        # Simple Vectorized Backtest (Signal * Return)
        # This assumes we trade everything immediately (simplified)
        test_data['strategy_return'] = test_data['signal'] * test_data[target_col]
        
        self.results = test_data
        logger.info("Backtest complete.")
        
        return test_data

    def get_performance_stats(self):
        """Calculates basic metrics."""
        if len(self.results) == 0:
            return None
        
        # Group by date to get portfolio daily return
        daily_returns = self.results.groupby(level=0)['strategy_return'].mean()
        
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        total_return = (1 + daily_returns).prod() - 1
        
        stats = {
            "Sharpe Ratio": round(sharpe, 2),
            "Total Return": round(total_return, 4)
        }
        return stats