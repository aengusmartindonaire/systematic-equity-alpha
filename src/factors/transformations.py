import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger()

def winsorize(series, limits=(0.01, 0.01)):
    """Clips values at the given quantiles."""
    return series.clip(lower=series.quantile(limits[0]), upper=series.quantile(1 - limits[1]))

def normalize_cross_section(df, factor_cols):
    """Applies Z-score per date group."""
    # Ensure index is unique to avoid errors
    df = df[~df.index.duplicated(keep='first')]
    return df.groupby(level=0)[factor_cols].apply(lambda x: (x - x.mean()) / x.std())

def run_clipping_bakeoff(df, target_col, quantiles, gfc_start='200710', gfc_end='200901'):
    """
    Tests different clipping thresholds to maximize correlation between Lev and Target during GFC.
    """
    logger.info(f"Running Clipping Bake-off on {target_col}...")
    
    gfc_mask = (df['Date'] >= gfc_start) & (df['Date'] <= gfc_end)
    results = []

    for q in quantiles:
        # Calculate clip limits
        lower = df[target_col].quantile(q)
        upper = df[target_col].quantile(1 - q)
        
        # Apply temporary clip
        clipped_target = df[target_col].clip(lower=lower, upper=upper)
        
        # Check correlation with Leverage during GFC (proxy for signal quality)
        # Note: We assume 'Lev' exists as it is a Core factor
        if 'Lev' in df.columns:
            gfc_corr = clipped_target[gfc_mask].corr(df.loc[gfc_mask, 'Lev'])
        else:
            gfc_corr = 0.0
            
        results.append({
            "q": q,
            "lower_clip": lower,
            "upper_clip": upper,
            "gfc_corr": gfc_corr,
            "skew": clipped_target.skew(),
            "kurtosis": clipped_target.kurtosis()
        })
        
    return pd.DataFrame(results)