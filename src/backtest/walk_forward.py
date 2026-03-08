"""
Walk-forward backtesting functions for systematic equity strategies.

This file contains:
- run_walk_forward_tv_style: Basic walk-forward with rolling/expanding window
- run_master_strategy: Advanced walk-forward with alpha target + XGBRanker
- AdaptiveBlendedModel: Blends BayesianRidge and XGBoost with adaptive weights
- print_performance_metrics: Detailed strategy performance reporting
- calculate_performance_metrics: Returns metrics dict for comparisons
- Helper functions for date parsing, feature ranking, market beta estimation
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import BayesianRidge
from xgboost import XGBRegressor

from src.utils.logger import setup_logger

logger = setup_logger()


# ============================================================================
# 1. BASIC WALK-FORWARD
# ============================================================================

def run_walk_forward_tv_style(
    df,
    feature_cols,
    model,
    window=12,
    top_bottom_frac=0.10,
    min_names=30,
    window_type="rolling",
    verbose=True
):
    """
    Walk-forward backtest for a TV-style XGB (or any sklearn-compatible) strategy.

    Expects df to have columns: Date, FwdRet, FwdRetOrig, plus feature_cols.
    Date should be a regular column (not index).

    Returns
    -------
    results_df : DataFrame
        Per-period metrics indexed by date.
    summary_stats : dict
        Aggregate stats: mean_ic, ic_hit_rate, cumulative_alpha, sharpe_ratio, max_drawdown.
    """
    dts = sorted(df["Date"].unique())
    score_results = []

    if window_type == "rolling":
        start_idx = window
    elif window_type == "expanding":
        start_idx = 1
    else:
        raise ValueError("window_type must be 'rolling' or 'expanding'.")

    for idx in range(start_idx, len(dts)):
        dt = dts[idx]

        if window_type == "rolling":
            train_dates = dts[idx - window: idx]
        else:
            train_dates = dts[:idx]

        if verbose:
            print(f"\nTrain: {train_dates[0]} ... {train_dates[-1]}  |  Test: {dt}")

        # Build train / test sets
        X_train = df.loc[df["Date"].isin(train_dates), feature_cols]
        y_train = df.loc[df["Date"].isin(train_dates), "FwdRet"]

        X_test = df.loc[df["Date"] == dt, feature_cols]
        y_test = df.loc[df["Date"] == dt, "FwdRet"]
        y_test_actual = df.loc[df["Date"] == dt, "FwdRetOrig"]

        # Clean data (no leakage)
        train_mask = ~y_train.isna()
        X_train_clean = X_train.loc[train_mask].copy()
        y_train_clean = y_train.loc[train_mask].copy()

        train_median = X_train_clean.median()
        X_train_clean = X_train_clean.fillna(train_median)

        test_mask = ~y_test.isna() & ~y_test_actual.isna()
        X_test_clean = X_test.loc[test_mask].copy()
        y_test_clean = y_test.loc[test_mask].copy()
        y_test_actual_clean = y_test_actual.loc[test_mask].copy()

        X_test_clean = X_test_clean.fillna(train_median)

        # Ensure enough stocks
        universe_size = len(y_test_clean)
        if universe_size == 0 or X_train_clean.empty:
            continue

        names_per_side = max(min_names, int(np.floor(top_bottom_frac * universe_size)))
        if universe_size < names_per_side * 2:
            continue

        # Fit model and predict
        period_model = clone(model)
        period_model.fit(X_train_clean, y_train_clean)
        preds = period_model.predict(X_test_clean)

        # Rank IC
        ic, _ = spearmanr(preds, y_test_clean)

        # Portfolio construction
        temp_df = pd.DataFrame(
            {"predicted": preds, "actual": y_test_actual_clean.values},
            index=y_test_actual_clean.index,
        )
        temp_df = temp_df.sort_values("predicted", ascending=False)

        long_return = temp_df.head(names_per_side)["actual"].mean()
        short_return = temp_df.tail(names_per_side)["actual"].mean()
        benchmark_return = temp_df["actual"].mean()
        long_short_alpha = long_return - short_return

        score_results.append({
            "period": dt,
            "rank_ic": ic,
            "long_return": long_return,
            "short_return": short_return,
            "benchmark_return": benchmark_return,
            "long_short_alpha": long_short_alpha,
        })

    results_df = pd.DataFrame(score_results).set_index("period").sort_index()

    if results_df.empty:
        summary_stats = {
            "mean_ic": np.nan, "ic_hit_rate": np.nan,
            "cumulative_alpha": np.nan, "sharpe_ratio": np.nan, "max_drawdown": np.nan,
        }
        return results_df, summary_stats

    alpha_series = results_df["long_short_alpha"]
    mean_ic = results_df["rank_ic"].mean()
    ic_hit_rate = (results_df["rank_ic"] > 0).mean()
    cumulative_alpha = alpha_series.sum()
    alpha_mean = alpha_series.mean()
    alpha_vol = alpha_series.std()
    sharpe_ratio = (alpha_mean / alpha_vol) * np.sqrt(4) if alpha_vol > 0 else np.nan

    cum_alpha = alpha_series.cumsum()
    running_max = cum_alpha.cummax()
    max_drawdown = (cum_alpha - running_max).min()

    summary_stats = {
        "mean_ic": mean_ic,
        "ic_hit_rate": ic_hit_rate,
        "cumulative_alpha": cumulative_alpha,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }

    if verbose:
        print(f"\nMean IC: {mean_ic:.4f} | IC Hit Rate: {ic_hit_rate:.2%}")
        print(f"Cumulative Alpha: {cumulative_alpha:+.4f} | Sharpe: {sharpe_ratio:.4f}")

    return results_df, summary_stats


# ============================================================================
# 2. PERFORMANCE REPORTING
# ============================================================================

def print_performance_metrics(results_df, label="Strategy"):
    """Prints detailed strategy performance breakdown."""
    print("" + "=" * 60)
    print(f"STRATEGY PERFORMANCE METRICS: {label}")
    print("=" * 60)

    mean_ic = results_df['rank_ic'].mean()
    print(f"Mean Rank IC: {mean_ic:.4f}")

    ic_hit_rate = (results_df['rank_ic'] > 0).sum() / len(results_df)
    print(f"IC Hit Rate: {ic_hit_rate:.2%}")

    # Long portfolio
    long_cumulative = results_df['long_return'].sum()
    long_mean = results_df['long_return'].mean()
    long_vol = results_df['long_return'].std()
    long_ir = (long_mean / long_vol) * np.sqrt(4) if long_vol > 0 else np.nan

    print(f"\nLong Portfolio:")
    print(f"  Cumulative Return: {long_cumulative:+.4f} ({long_cumulative*100:+.2f}%)")
    print(f"  Mean Quarterly Return: {long_mean:+.4f}")
    print(f"  Volatility (Quarterly): {long_vol:.4f}")
    print(f"  Information Ratio (Annualized): {long_ir:.4f}")

    # Short portfolio
    short_cumulative = results_df['short_return'].sum()
    short_mean = results_df['short_return'].mean()
    short_vol = results_df['short_return'].std()
    short_ir = (short_mean / short_vol) * np.sqrt(4) if short_vol > 0 else np.nan

    print(f"\nShort Portfolio:")
    print(f"  Cumulative Return: {short_cumulative:+.4f} ({short_cumulative*100:+.2f}%)")
    print(f"  Mean Quarterly Return: {short_mean:+.4f}")
    print(f"  Volatility (Quarterly): {short_vol:.4f}")
    print(f"  Information Ratio (Annualized): {short_ir:.4f}")

    # Benchmark
    benchmark_cumulative = results_df['benchmark_return'].sum()
    benchmark_mean = results_df['benchmark_return'].mean()
    print(f"\nBenchmark (Market Average):")
    print(f"  Cumulative Return: {benchmark_cumulative:+.4f} ({benchmark_cumulative*100:+.2f}%)")
    print(f"  Mean Quarterly Return: {benchmark_mean:+.4f}")

    # Excess returns
    long_excess = long_cumulative - benchmark_cumulative
    short_excess = benchmark_cumulative - short_cumulative
    print(f"\nExcess Returns vs Benchmark:")
    print(f"  Long Portfolio Excess: {long_excess:+.4f} ({long_excess*100:+.2f}%)")
    print(f"  Short Portfolio Excess: {short_excess:+.4f} ({short_excess*100:+.2f}%)")

    print("=" * 60)


def calculate_performance_metrics(results_df, model_name):
    """Calculates and prints performance metrics, returns a dict for comparisons."""
    print(f"\n{model_name} Performance Metrics")
    print("-" * 40)

    mean_ic = results_df['rank_ic'].mean()
    ic_hit_rate = (results_df['rank_ic'] > 0).sum() / len(results_df)

    ls_cumulative = results_df['long_short_alpha'].sum()
    ls_mean = results_df['long_short_alpha'].mean()
    ls_vol = results_df['long_short_alpha'].std()
    ls_ir = (ls_mean / ls_vol) * np.sqrt(4) if ls_vol > 0 else 0
    ls_sharpe = ls_ir
    win_rate = (results_df['long_short_alpha'] > 0).sum() / len(results_df)

    print(f"Mean Rank IC: {mean_ic:.4f}")
    print(f"IC Hit Rate: {ic_hit_rate:.2%}")
    print(f"Cumulative LS Alpha: {ls_cumulative:+.4f} ({ls_cumulative*100:+.2f}%)")
    print(f"Annualized Sharpe: {ls_sharpe:.4f}")
    print(f"Win Rate: {win_rate:.2%}")

    return {
        'cumulative_ls_alpha': ls_cumulative,
        'mean_ic': mean_ic,
        'ic_hit_rate': ic_hit_rate,
        'ir': ls_ir,
        'sharpe': ls_sharpe,
        'win_rate': win_rate,
    }


# ============================================================================
# 3. ADVANCED HELPERS
# ============================================================================

def coerce_yyyymm_to_month_end(date_series):
    """Parse YYYYMM integers to month-end Timestamps."""
    s = pd.to_datetime(date_series.astype(str), format="%Y%m", errors="coerce")
    if s.notna().mean() > 0.80:
        return s + pd.offsets.MonthEnd(0)
    return pd.to_datetime(date_series, errors="coerce")


def quant_preprocess_features(df, feature_cols):
    """Cross-sectionally rank each feature within Date to [0, 1]."""
    out = df.copy()
    for col in feature_cols:
        out[col] = out.groupby("Date")[col].transform(
            lambda s: s.rank(pct=True, method="average")
        )
    return out


def build_market_fwdret_series(
    unique_dates,
    start_date,
    end_date,
    use_yfinance=True,
    fallback_universe_mean=None,
):
    """
    Returns market forward return series indexed by unique_dates.
    Uses yfinance (optional) with fallback to universe-mean proxy.
    """
    if use_yfinance:
        try:
            import yfinance as yf
            px = yf.Ticker("^GSPC").history(
                start=start_date, end=end_date, auto_adjust=True
            )["Close"]
            if px.empty:
                raise ValueError("Empty price history for ^GSPC")
            px.index = px.index.tz_localize(None)

            px_aligned = px.reindex(unique_dates, method="ffill")
            mkt_fwd = px_aligned.pct_change().shift(-1)
            return pd.Series(
                mkt_fwd.values,
                index=pd.Index(unique_dates, name="Date"),
                name="Mkt_FwdRet",
            )
        except Exception as e:
            logger.warning(f"yfinance market data failed ({e}). Using fallback.")

    if fallback_universe_mean is None or fallback_universe_mean.empty:
        raise ValueError(
            "No market series available: yfinance failed and fallback not provided."
        )
    return fallback_universe_mean.reindex(unique_dates)


def add_rolling_market_beta(
    df,
    mkt_fwdret_by_date,
    beta_window=12,
    min_obs=8,
    beta_clip=(-1.0, 3.0),
):
    """Creates MarketBeta using only backward-looking info (no lookahead)."""
    out = df.copy()
    out["Mkt_FwdRet"] = out["Date"].map(mkt_fwdret_by_date)

    out = out.sort_values(["Ticker", "Date"])

    out["_cov"] = out.groupby("Ticker", group_keys=False).apply(
        lambda g: g["FwdRet"].rolling(beta_window, min_periods=min_obs).cov(g["Mkt_FwdRet"])
    ).values

    mkt_var = (
        out.groupby("Date")["Mkt_FwdRet"]
        .first()
        .rolling(beta_window, min_periods=min_obs)
        .var()
    )
    out["_mkt_var"] = out["Date"].map(mkt_var)

    out["MarketBeta"] = (out["_cov"] / out["_mkt_var"]).fillna(1.0).clip(*beta_clip)
    out.drop(columns=["_cov", "_mkt_var"], inplace=True)

    return out


def default_monotonic_constraints(feature_cols):
    """
    Returns monotonic constraint tuple for XGBRanker.
    1 = increasing helps, -1 = increasing hurts, 0 = unconstrained.
    """
    sign = {
        "Sz": 0, "Prof": 1, "Vol": -1, "Trd Act": 0,
        "Lev": -1, "Mom": 1, "Val": 1, "Gr": 0,
        "Dvd Yld": 1, "Earn Var": -1,
        "BEst P/S BF12M": -1, "BEst P/B BF12M": -1, "BEst P/E BF12M": -1,
        "Beta:Y-1": 0,
    }
    missing = [c for c in feature_cols if c not in sign]
    if missing:
        raise KeyError(f"Missing monotonic sign spec for: {missing}")
    return tuple(sign[c] for c in feature_cols)


# ============================================================================
# 4. ADVANCED MASTER STRATEGY 
# ============================================================================

def run_master_strategy(
    df,
    feature_cols,
    xgb_params,
    window=12,
    top_bottom_frac=0.10,
    min_test_names=50,
):
    """
    Walk-forward backtest using XGBRanker with:
    - Cross-sectional feature ranking
    - Beta-neutral alpha target
    - Monotonic constraints
    - Pairwise ranking objective

    Returns DataFrame with per-period Rank_IC, Long_Ret, Short_Ret, Spread.
    """
    df_clean = df.copy()

    # 1) Rank features cross-sectionally
    df_clean = quant_preprocess_features(df_clean, feature_cols)

    # 2) Build alpha label
    if "MarketBeta" in df_clean.columns and "Mkt_FwdRet" in df_clean.columns:
        df_clean["Alpha_Target"] = (
            df_clean["FwdRet"].astype(float)
            - df_clean["MarketBeta"].astype(float) * df_clean["Mkt_FwdRet"].astype(float)
        )
    else:
        logger.warning("Missing MarketBeta or Mkt_FwdRet. Using raw FwdRet as Alpha_Target.")
        df_clean["Alpha_Target"] = df_clean["FwdRet"].astype(float)

    df_clean["Alpha_Target"] = df_clean["Alpha_Target"].replace([np.inf, -np.inf], np.nan)

    # 3) Configure ranker params
    params = dict(xgb_params)
    params["objective"] = "rank:pairwise"
    params["eval_metric"] = "ndcg"
    params["monotone_constraints"] = default_monotonic_constraints(feature_cols)

    dts = sorted(df_clean["Date"].dropna().unique())
    results = []

    for idx in range(window, len(dts)):
        test_dt = dts[idx]
        train_dts = dts[idx - window: idx]

        train_df = df_clean.loc[df_clean["Date"].isin(train_dts)].sort_values("Date")
        test_df = df_clean.loc[df_clean["Date"] == test_dt].copy()

        train_df = train_df.dropna(subset=["Alpha_Target"])
        if train_df.empty or len(test_df) < min_test_names:
            continue

        train_medians = train_df[feature_cols].median(numeric_only=True)
        X_train = train_df[feature_cols].fillna(train_medians)
        y_train = train_df["Alpha_Target"].astype(float)

        group_sizes = train_df.groupby("Date").size().tolist()

        X_test = test_df[feature_cols].fillna(train_medians)
        y_test_alpha = test_df["Alpha_Target"].astype(float)
        y_test_raw = test_df["FwdRetOrig"].astype(float)

        model = xgb.XGBRanker(**params)
        model.fit(X_train, y_train, group=group_sizes, verbose=False)
        preds = model.predict(X_test)

        ic = spearmanr(preds, y_test_alpha, nan_policy="omit").correlation
        ic = 0.0 if ic is None or np.isnan(ic) else float(ic)

        port = pd.DataFrame({"pred": preds, "actual": y_test_raw.values}).sort_values("pred")
        n = max(1, int(len(port) * top_bottom_frac))
        long_ret = float(port.tail(n)["actual"].mean())
        short_ret = float(port.head(n)["actual"].mean())

        results.append({
            "Date": test_dt,
            "Rank_IC": ic,
            "Long_Ret": long_ret,
            "Short_Ret": short_ret,
            "Spread": long_ret - short_ret,
            "N": len(port),
        })

    return pd.DataFrame(results)


# ============================================================================
# 5. ADAPTIVE BLENDED MODEL
# ============================================================================

class AdaptiveBlendedModel:
    """
    Blends predictions from BayesianRidge and XGBoost with adaptive weights.
    Weights update based on rolling IC performance.
    """

    def __init__(self, br_model, xgb_model, initial_br_weight=0.7):
        self.br_model = br_model
        self.xgb_model = xgb_model
        self.initial_br_weight = initial_br_weight
        self.br_weight = initial_br_weight
        self.xgb_weight = 1 - initial_br_weight
        self.is_fitted = False
        self.performance_history = []

    def update_weights_based_on_performance(self, br_ic, xgb_ic, window=4):
        """Adapt weights based on recent IC performance."""
        self.performance_history.append({'br_ic': br_ic, 'xgb_ic': xgb_ic})

        if len(self.performance_history) > window:
            self.performance_history = self.performance_history[-window:]

        if len(self.performance_history) >= 2:
            avg_br_ic = np.mean([p['br_ic'] for p in self.performance_history])
            avg_xgb_ic = np.mean([p['xgb_ic'] for p in self.performance_history])

            br_perf = max(avg_br_ic, 0.01)
            xgb_perf = max(avg_xgb_ic, 0.01)

            total_perf = br_perf + xgb_perf
            if total_perf > 0:
                self.br_weight = br_perf / total_perf
                self.xgb_weight = 1 - self.br_weight

    def fit(self, X, y):
        self.br_model.fit(X, y)
        self.xgb_model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        br_pred = self.br_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        return self.br_weight * br_pred + self.xgb_weight * xgb_pred

    def get_current_weights(self):
        return {'br_weight': self.br_weight, 'xgb_weight': self.xgb_weight}