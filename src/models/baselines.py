import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import BayesianRidge, LassoCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.utils.logger import setup_logger

logger = setup_logger()

def get_baseline_models(random_state=42):
    """
    Returns a dictionary of model archetypes with fixed baseline parameters.
    """
    models = {
        'BayesianRidge': BayesianRidge(),
        'LassoCV': LassoCV(alphas=[0.001, 0.01, 0.1], cv=3, random_state=random_state, max_iter=2000),
        'LinearSVR': LinearSVR(max_iter=2000, random_state=random_state, dual=True),
        'RandomForest': RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=random_state,
            n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.01,
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models

def run_model_bakeoff(X, y, models=None, n_splits=5, random_state=42):
    """
    Runs K-Fold Cross Validation on all provided models and returns a scoreboard DataFrame.
    """
    if models is None:
        models = get_baseline_models(random_state)
        
    logger.info(f"Starting Model Bake-off with {len(models)} models and {n_splits} CV splits...")
    
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = []

    for name, model in models.items():
        logger.info(f"Training {name}...")
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
            mean_score = scores.mean()
            std_score = scores.std()
            
            results.append({
                'Model': name,
                'R2 Mean': mean_score,
                'R2 Std': std_score
            })
            logger.info(f"   -> {name} R2: {mean_score:.4f}")
            
        except Exception as e:
            logger.error(f"Failed to train {name}: {e}")

    # Return sorted results
    results_df = pd.DataFrame(results).sort_values(by='R2 Mean', ascending=False)
    return results_df