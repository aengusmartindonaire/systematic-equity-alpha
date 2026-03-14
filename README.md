# Building the Black Box: A Production ML System for Systematic Trading

A full-cycle quantitative research pipeline that ingests 25 years of Bloomberg risk factor data (1999–2025), engineers features, selects and tunes models, performs deep SHAP-driven explainability analysis, and runs walk-forward backtests with long-short portfolio construction.

---

## Project Context

Built for a quantitative hedge fund managing $15B in assets. The objective: develop a machine learning system to predict forward equity returns and identify investment themes using Bloomberg's proprietary factor database.

The pipeline covers every stage of a production quant workflow. From raw data validation through to a walk-forward backtested trading strategy with performance attribution.

---

## Key Results

| Stage | Metric | Value |
|---|---|---|
| Model Selection | Best CV R² | XGBoost: 0.104 (Optuna-tuned) |
| Explainability | Top SHAP driver | `Beta:Y-1` by wide margin, then `Dvd Yld`, `Earn Var` |
| PM Dashboard | In-sample IC | 0.59 (Spearman rank correlation) |
| Walk-Forward | **Best strategy** | **Adaptive Blended: +439% cumulative alpha, Sharpe 0.38** |

**Central Finding:** The R²-tuned XGBoost that dominated in-sample (R² ~ 0.10) produced *negative* alpha (−20%) in walk-forward testing. The "boring" BayesianRidge achieved better out-of-sample rank quality (IC = 0.030 vs 0.022). The Adaptive Blended Model (dynamically weighting both) delivered the best overall performance: +439% cumulative long-short alpha over 25 years.

---

## Repository Structure

```
systematic-equity-alpha/
├── configs/
│   └── config.yaml                          # Central config: paths, columns, model params
├── data/
│   ├── README.md                            # Schema documentation (data not included)
│   ├── raw/                                 # Bloomberg factor file (.par) — excluded
│   └── processed/                           # Intermediate datasets — excluded
├── notebooks/
│   ├── 01_data_ingestion.ipynb              # Load, validate, inspect raw data
│   ├── 02_feature_engineering.ipynb         # Multicollinearity, outlier treatment, GICS
│   ├── 03_model_baselines_and_selection.ipynb   # 5-model bake-off
│   ├── 04_model_tuning_and_optimization.ipynb   # RandomizedSearchCV + Optuna
│   ├── 05_model_explainability_and_analysis.ipynb  # Full SHAP analysis
│   ├── 06_pm_dashboard_and_alpha.ipynb      # PM dashboard, surprise detection
│   └── 07_walk_forward_strategy.ipynb       # Walk-forward backtest, blended model
├── src/
│   ├── loaders/
│   │   ├── data_loader.py                   # Parquet loading, date parsing, MultiIndex
│   │   └── preprocessing.py                 # GICS cleaning, one-hot encoding, bake-off
│   ├── factors/
│   │   ├── factor_analysis.py               # Core vs. extra feature redundancy analysis
│   │   ├── transformations.py               # Winsorization, normalization, clipping bake-off
│   │   └── technical_indicators.py          # RSI, rolling volatility, momentum
│   ├── models/
│   │   ├── baselines.py                     # 5-model archetype bake-off (CV R²)
│   │   ├── optimization.py                  # RandomizedSearchCV + Optuna Bayesian optimization
│   │   └── xgb_model.py                     # XGBStrategy wrapper (train/predict/save/load)
│   ├── backtest/
│   │   ├── engine.py                        # Signal-based backtest engine
│   │   ├── metrics.py                       # Sharpe, max drawdown, win rate, IC
│   │   └── walk_forward.py                  # Walk-forward engine, master strategy, blended model
│   └── utils/
│       ├── config_loader.py                 # YAML config with automatic path resolution
│       ├── logger.py                        # Dual console + file logging
│       └── visualizations.py                # Reusable plots (heatmaps, distributions, comparisons)
├── models/                                  # Saved model artifacts (.joblib)
├── results/                                 # Exported metrics, SHAP caches
├── logs/                                    # Runtime logs
├── tests/
│   └── conftest.py                          # Shared pytest fixtures
├── requirements.txt
├── setup.py
```

---

## Pipeline Overview

### Phase 1 — Data & Features (Notebooks 01–02)

**Data:** 25 years of Bloomberg risk factors — ~93,000 rows × 35 columns as a panel dataset indexed by `(Date, Ticker)`.

| Step | What Happens |
|---|---|
| **Data Ingestion** | Load parquet, parse YYYYMM dates, construct MultiIndex, validate schema and sparsity. |
| **Multicollinearity Audit** | Cross-correlate 10 core risk factors against 11 extra features. Drop redundant features using correlation thresholds + manual domain review. Final keep: `BEst P/S BF12M`, `BEst P/B BF12M`, `BEst P/E BF12M`, `Beta:Y-1`. |
| **Total Return Investigation** | Solve the −0.96 Pearson mystery between `Beta:Y-1` and `Total Return:Y-1` (caused by extreme outliers; Spearman ≈ 0.00). Drop `Total Return:Y-1` as redundant with `Mom` (Spearman ≈ 0.79). |
| **Target Treatment** | Run a clipping bake-off on `FwdRet` using GFC-period Leverage correlation as selection criterion. Winner: q = 0.008 symmetric clip. Preserve `FwdRetOrig` for walk-forward P&L. |
| **GICS Encoding** | Hierarchical NaN fill, rare category handling (< 500 obs → NaN), one-hot encode. BayesianRidge CV bake-off across Sector/Industry/SubIndustry granularity. |

### Phase 2 — Modeling (Notebooks 03–05)

| Step | What Happens |
|---|---|
| **Model Bake-Off** | 5 archetypes (BayesianRidge, LassoCV, LinearSVR, RandomForest, XGBoost) via 5-fold CV R². LassoCV and LinearSVR eliminated (R² ≈ 0). Tree models advance. |
| **Hyperparameter Tuning** | RandomizedSearchCV (10 iterations) for both tree finalists with `scipy.stats` distributions and Pipeline wrapping. Optuna Bayesian optimization (50 trials) for XGBoost. Best CV R² ≈ 0.104. |
| **Explainability** | Full SHAP analysis on 1,000-stock sample: global bar/dot plots, partial dependence, interaction plots (top 5 features), 4-metric XGBoost importance comparison, sector scorecard with signed themes, 2023 Banking Crisis case study with survivorship bias check, Technology and Consumer Staples segment deep-dives, single-stock force plots (NVDA, ORCL, PACW). |

**SHAP Key Finding:** `Beta:Y-1` dominates by wide margin. All 14 numerical factors rank above any GICS subindustry. Model directions are economically sensible: high beta → higher returns, low dividend yield → higher returns (growth regime), cheap valuations rewarded.

### Phase 3 — Strategy & Backtesting (Notebooks 06–07)

| Step | What Happens |
|---|---|
| **PM Dashboard** | Generate in-sample predictions (IC = 0.59). Build "Unexpected Gain" residual table. SHAP waterfall post-mortems: PACW (−54.6% unexpected loss, banking crisis) and VST (+68.5% unexpected gain, AI power demand). |
| **Walk-Forward Part 1** | Basic TV-style XGB with R²-tuned params, 10-period window, 14 features. **Failed:** IC ≈ 0, alpha = −20%. |
| **Walk-Forward Part 2** | Optuna IC-tuned XGBRanker with alpha target, monotonic constraints, 20-period window. **Marginal improvement:** IC ≈ 0.01, still negative alpha. $1 → $0.35. |
| **Walk-Forward Part 3** | Three-model head-to-head (XGBoost, BayesianRidge, Adaptive Blended) with 1-period window, 92 features (numericals + GICS dummies). **Winner: Adaptive Blended** — +439% alpha, Sharpe 0.38, Win Rate 60.4%. |

---

## Data

**The raw Bloomberg data file is not included** in this repository due to licensing restrictions.

**File:** `data/raw/20251109_Blg_Rsk_Factors.par` (Parquet format)

| Column Group | Columns |
|---|---|
| **Identifiers** | `Date` (YYYYMM), `Ticker` |
| **Core Risk Factors (10)** | `Sz`, `Prof`, `Vol`, `Trd Act`, `Lev`, `Mom`, `Val`, `Gr`, `Dvd Yld`, `Earn Var` |
| **Extra Features (11)** | `P/S`, `BEst P/S BF12M`, `P/B`, `BEst P/B BF12M`, `P/E`, `BEst P/E BF12M`, `ROE LF`, `Beta:Y-1`, `Total Return:Y-1`, `Number of Employees:Y`, `Market Cap` |
| **Classification** | `GICS_Sector_Name`, `GICS_Industry_Name`, `GICS_SubInd_Name` |
| **Target** | `FwdRet` (forward 1-month return) |

**Coverage:** ~93,000 stock-months, April 1999 – September 2025, US equities.

See `data/README.md` for the full schema.

The notebooks retain their cell outputs (plots, tables, print statements) so the full analysis is readable without the data. If you have access to the dataset, place it in `data/raw/` and the pipeline will pick it up via `configs/config.yaml`.

---

## Quick Start

```bash
# Clone
git clone https://github.com/aengusmartindonaire/systematic-equity-alpha.git
cd systematic-equity-alpha

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode (enables `from src.xxx import yyy`)
pip install -e .

# Run notebooks in order
cd notebooks/
jupyter notebook
```

### Requirements

- Python ≥ 3.9
- Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `optuna`, `shap`
- Visualization: `matplotlib`, `seaborn`, `plotly`
- Optional: `yfinance` (for S&P 500 market beta in Notebook 07; falls back to universe-mean proxy if unavailable)

### SHAP + XGBoost Compatibility

If you encounter a `ValueError: could not convert string to float` when initializing SHAP's `TreeExplainer`, add this before creating the explainer:

```python
xgb_champion.set_params(base_score=float(y_clean.mean()))
xgb_champion.fit(X_clean, y_clean)
```

This forces XGBoost to store `base_score` as a plain float instead of the array-like string that newer versions produce.

---

## Architecture

The `src/` package is organized by function:

```
src/
├── loaders/      → Data in
├── factors/      → Feature engineering
├── models/       → Model training & tuning
├── backtest/     → Walk-forward engine & metrics
└── utils/        → Config, logging, visualization
```

Every `.py` module is imported by at least one notebook. The notebooks are the primary interface — they orchestrate the `src/` modules and contain all narrative analysis, plots, and commentary. The `src/` modules contain only reusable, parameterized functions with no hardcoded values (everything flows from `configs/config.yaml`).

**Key `src/` modules:**

| Module | Key Functions | Used By |
|---|---|---|
| `loaders/data_loader.py` | `DataLoader.load_raw_data()` | NB 01, 02 |
| `factors/factor_analysis.py` | `get_feature_decisions()` | NB 02 |
| `models/baselines.py` | `run_model_bakeoff()` | NB 03 |
| `models/optimization.py` | `run_randomized_search()`, `optimize_xgboost_optuna()` | NB 04 |
| `backtest/walk_forward.py` | `run_walk_forward_tv_style()`, `run_master_strategy()`, `AdaptiveBlendedModel` | NB 07 |
| `backtest/metrics.py` | `information_coefficient()`, `sharpe_ratio()` | NB 06, 07 |

---

## Notebook Guide

| # | Notebook | Sections | Runtime | Key Outputs |
|---|---|---|---|---|
| 01 | Data Ingestion | §1.1 | < 1 min | Schema validation, sparsity analysis |
| 02 | Feature Engineering | §1.2–1.5 | ~2 min | Correlation heatmaps, clipping bake-off, GICS bake-off, `final_model_data.parquet` |
| 03 | Model Baselines | §2.1 | ~3 min | 5-model CV scoreboard, `03_model_bakeoff_metrics.csv` |
| 04 | Model Tuning | §2.2 | ~15 min | RandomizedSearchCV + Optuna (50 trials), `xgb_champion.joblib` |
| 05 | Explainability | §2.3 | ~30 min | SHAP bar/dot/interaction/PDP plots, sector scorecard, banking crisis case study, force plots |
| 06 | PM Dashboard | §3.1 | ~10 min | Unexpected gain/loss table, SHAP waterfall post-mortems (PACW, VST) |
| 07 | Walk-Forward | §3.2 | ~60 min | 3-strategy comparison, cumulative alpha charts, model comparison summary |

**Note on runtime:** Notebook 05's SHAP interaction values (~20 min) are cached to `results/shap_interactions.pkl` after first computation. Notebook 07's Optuna IC study (30 trials × walk-forward) is the longest-running cell.

---

## Configuration

All runtime parameters are centralized in `configs/config.yaml`:

```yaml
# Model: Optuna XGBoost Champion (from §2.2)
model:
  params:
    n_estimators: 593
    max_depth: 5
    learning_rate: 0.0233
    subsample: 0.754
    colsample_bytree: 0.762
    reg_lambda: 5.013

# Walk-forward settings (from §3.2)
  walk_forward:
    train_window: 12
    top_bottom_frac: 0.10
    min_names: 30
    window_type: "rolling"
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data | pandas, NumPy, SciPy |
| ML | scikit-learn, XGBoost, Optuna |
| Explainability | SHAP |
| Visualization | Matplotlib, Seaborn, Plotly |
| Infrastructure | YAML config, structured logging, modular `src/` package |

---

## Acknowledgements

Thanks to Prof. Low for the guidance.

