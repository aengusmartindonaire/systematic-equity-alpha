import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor
from src.utils.logger import setup_logger

logger = setup_logger()

def optimize_xgboost(X, y, n_iter=20, cv_splits=3, random_state=42):
    """
    Performs RandomizedSearchCV to optimize XGBoost hyperparameters.
    Uses TimeSeriesSplit to respect temporal order (optional but recommended for finance).
    """
    logger.info(f"Starting XGBoost optimization with {n_iter} iterations...")

    # 1. Define the Parameter Grid
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1, 10],  # L1 Regularization
        'reg_lambda': [1, 10, 50]      # L2 Regularization
    }

    # 2. Setup Cross-Validation
    # TimeSeriesSplit is safer for financial data to prevent look-ahead bias
    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # 3. Setup the Model
    xgb = XGBRegressor(random_state=random_state, n_jobs=-1)

    # 4. Run Randomized Search
    search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring='r2',
        cv=tscv,
        verbose=1,
        random_state=random_state,
        n_jobs=-1
    )

    search.fit(X, y)

    logger.info(f"Optimization Complete. Best Score: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.cv_results_