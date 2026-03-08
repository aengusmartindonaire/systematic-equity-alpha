import pandas as pd
import numpy as np
from scipy.stats import randint, uniform
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import optuna
from src.utils.logger import setup_logger

logger = setup_logger()


# ---------------------------------------------------------------------------
# 1.  Parameter distributions  (consolidated notebook §2.2, cell 94)
# ---------------------------------------------------------------------------

RF_PARAM_DISTRIBUTIONS = {
    "model__n_estimators": randint(200, 600),       # 200–599 trees
    "model__max_depth":    randint(3, 15),           # depth 3–14
    "model__max_features": uniform(0.3, 0.7),       # 0.3–1.0 of features
}

XGB_PARAM_DISTRIBUTIONS = {
    "model__n_estimators":      randint(200, 1000),  # 200–999 trees
    "model__max_depth":         randint(2, 9),       # depth 2–8
    "model__learning_rate":     uniform(0.01, 0.09), # 0.01–0.10
    "model__subsample":         uniform(0.6, 0.4),   # 0.6–1.0
    "model__colsample_bytree":  uniform(0.6, 0.4),   # 0.6–1.0
    "model__reg_lambda":        uniform(1.0, 99.0),  # 1–100
}


# ---------------------------------------------------------------------------
# 2.  Base model definitions  (consolidated notebook §2.2, cell 92)
# ---------------------------------------------------------------------------

def get_tree_finalists(random_state=42):
    """
    Returns the two tree-based finalists that advanced from the §2.1 bake-off.
    """
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


# ---------------------------------------------------------------------------
# 3.  RandomizedSearchCV tuning  (consolidated notebook §2.2, cell 96)
# ---------------------------------------------------------------------------

def run_randomized_search(X, y, n_iter=10, n_splits=5, random_state=42):
    """
    Runs RandomizedSearchCV for both tree finalists (RandomForest and XGBoost),
    matching the consolidated notebook §2.2 exactly:
      - Pipeline-wrapped models  (params prefixed with "model__")
      - KFold(n_splits=5, shuffle=True, random_state=42)
      - scipy.stats distributions for continuous sampling
      - scoring='r2'

    Returns
    -------
    tuned_results : dict
        {model_name: {"best_score", "best_estimator", "best_params"}}
    """
    logger.info(f"Starting RandomizedSearchCV with n_iter={n_iter}, {n_splits}-fold CV...")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    tree_models = get_tree_finalists(random_state)

    param_spaces = {
        "RandomForest": RF_PARAM_DISTRIBUTIONS,
        "XGBoost":      XGB_PARAM_DISTRIBUTIONS,
    }

    tuned_results = {}

    for name, base_model in tree_models.items():
        logger.info(f"Tuning {name} with RandomizedSearchCV...")

        # 1. Wrap in Pipeline (no scaler needed for tree models)
        pipe = Pipeline([("model", base_model)])

        # 2. Select the matching search space
        param_distributions = param_spaces[name]

        # 3. Run the search
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring="r2",
            cv=kfold,
            random_state=random_state,
            n_jobs=-1,
            verbose=1,
        )

        search.fit(X, y)

        # 4. Store results
        tuned_results[name] = {
            "best_score":     search.best_score_,
            "best_estimator": search.best_estimator_,
            "best_params":    search.best_params_,
        }

        logger.info(f"Best mean CV R² for {name}: {search.best_score_:.6f}")
        for k, v in search.best_params_.items():
            logger.info(f"  {k}: {v}")

    return tuned_results


# ---------------------------------------------------------------------------
# 4.  Optuna Bayesian Optimization  (consolidated notebook §2.2, cell 100)
# ---------------------------------------------------------------------------

def optimize_xgboost_optuna(X, y, n_trials=50, n_splits=5, random_state=42):
    """
    Runs Optuna Bayesian optimization for XGBoost, matching consolidated §2.2
    exactly:
      - Same search ranges as cell 100
      - KFold(n_splits=5, shuffle=True, random_state=42)
      - Maximizes mean CV R²

    Returns
    -------
    study : optuna.Study
        The completed study (access study.best_params, study.best_value)
    best_model : XGBRegressor
        A fresh XGBRegressor instantiated with the winning params
    """
    logger.info(f"Starting Optuna XGBoost optimization with {n_trials} trials...")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    def xgb_objective(trial):
        n_estimators  = trial.suggest_int("n_estimators", 200, 1000)
        max_depth     = trial.suggest_int("max_depth", 2, 9)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.10, log=False)
        subsample     = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bt  = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_lambda    = trial.suggest_float("reg_lambda", 1.0, 100.0, log=False)

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bt,
            reg_lambda=reg_lambda,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

        scores = cross_val_score(
            model, X, y,
            cv=kfold,
            scoring="r2",
            n_jobs=-1,
        )

        return scores.mean()

    # Run the study
    study = optuna.create_study(direction="maximize")
    study.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Optuna complete. Best mean CV R²: {study.best_value:.6f}")
    for k, v in study.best_params.items():
        logger.info(f"  {k}: {v}")

    # Build a fresh model with the winning parameters
    best_model = XGBRegressor(
        **study.best_params,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_jobs=-1,
    )

    return study, best_model