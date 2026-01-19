import xgboost as xgb
import pandas as pd
import numpy as np
import joblib
import os
from src.utils.logger import setup_logger

logger = setup_logger()

class XGBStrategy:
    def __init__(self, config):
        self.config = config
        self.params = config['model']['params']
        self.model = None
        self.model_path = os.path.join(config['paths']['models'], "xgb_model.json")

    def train(self, X_train, y_train):
        """Trains the XGBoost model."""
        logger.info("Starting XGBoost training...")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Train
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.config['model']['params']['n_estimators']
        )
        logger.info("Training complete.")
        self.save_model()

    def predict(self, X_new):
        """Generates predictions (alpha signals)."""
        if self.model is None:
            self.load_model()
            
        dtest = xgb.DMatrix(X_new)
        predictions = self.model.predict(dtest)
        return predictions

    def save_model(self):
        """Saves model to JSON."""
        self.model.save_model(self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads model from JSON."""
        if os.path.exists(self.model_path):
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.error("No model found to load.")
            raise FileNotFoundError("Train the model before predicting.")