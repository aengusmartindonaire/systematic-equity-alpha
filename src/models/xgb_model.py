import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor
from src.utils.logger import setup_logger

logger = setup_logger()


class XGBStrategy:
    """
    Wrapper around XGBRegressor (sklearn API) for training, predicting,
    and persisting the tuned XGBoost champion model.

    Uses the sklearn-compatible API throughout — matching the consolidated
    notebook §2.3, §3.1, and §3.2 where the model is always instantiated
    via XGBRegressor and called with .fit() / .predict().
    """

    def __init__(self, config):
        self.config = config
        self.model_dir = config["paths"]["models"]
        self.model_path = os.path.join(self.model_dir, "xgb_champion.joblib")

        # Build XGBRegressor from config params
        # These should match the Optuna best params from §2.2
        params = config["model"]["params"]
        self.model = XGBRegressor(
            n_estimators=params.get("n_estimators", 801),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.0168),
            subsample=params.get("subsample", 0.83),
            colsample_bytree=params.get("colsample_bytree", 0.74),
            reg_lambda=params.get("reg_lambda", 10.17),
            objective="reg:squarederror",
            tree_method="hist",
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
        )

    def train(self, X_train, y_train):
        """Fits the XGBRegressor on training data."""
        logger.info(f"Training XGBRegressor on {X_train.shape[0]} rows, {X_train.shape[1]} features...")
        self.model.fit(X_train, y_train)
        logger.info("Training complete.")
        self.save_model()

    def predict(self, X_new):
        """Generates predictions (alpha signals)."""
        predictions = self.model.predict(X_new)
        return predictions

    def save_model(self):
        """Saves the fitted model to disk via joblib."""
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Loads a previously saved model from disk."""
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded from {self.model_path}")
        else:
            logger.error(f"No model found at {self.model_path}")
            raise FileNotFoundError("Train the model before predicting.")