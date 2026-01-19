import pandas as pd
import os
from src.utils.logger import setup_logger

logger = setup_logger()

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['paths']['raw_data']

    def load_raw_data(self):
        """Loads the raw Parquet file."""
        if not os.path.exists(self.raw_path):
            logger.error(f"File not found: {self.raw_path}")
            raise FileNotFoundError(f"File not found: {self.raw_path}")

        logger.info(f"Loading data from {self.raw_path}...")
        try:
            # Read parquet
            df = pd.read_parquet(self.raw_path)
            
            # Ensure date column is datetime with specific YYYYMM format
            date_col = self.config['data']['date_col']
            
            # Fallback check if case sensitivity is an issue
            if date_col not in df.columns:
                for c in df.columns:
                    if 'date' in c.lower():
                        date_col = c
                        break
            
            if date_col in df.columns:
                # FIX: Explicitly handle 'YYYYMM' format (e.g. 199904)
                df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m', errors='coerce')
                
                # Check for bad dates
                if df[date_col].isna().any():
                    logger.warning(f"Found {df[date_col].isna().sum()} invalid dates. Dropping them.")
                    df = df.dropna(subset=[date_col])

                # Handle MultiIndex (Date, Ticker) if ticker exists
                ticker_col = self.config['data'].get('ticker_col', 'ticker')
                
                # Check for ticker column variations
                if ticker_col not in df.columns and 'Ticker' in df.columns:
                    ticker_col = 'Ticker'

                if ticker_col in df.columns:
                    df = df.set_index([date_col, ticker_col]).sort_index()
                else:
                    df = df.set_index(date_col).sort_index()
            
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def save_processed(self, df, filename="processed_data.parquet"):
        """Saves processed dataframe to data/processed using absolute paths."""
        processed_dir = os.path.dirname(self.config['paths']['processed_data'])
        os.makedirs(processed_dir, exist_ok=True)
        
        save_path = os.path.join(processed_dir, filename)
        df.to_parquet(save_path)
        logger.info(f"Saved processed data to {save_path}")