import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger()

import pandas as pd
from src.utils.logger import setup_logger

logger = setup_logger()

def classify_extra_feature(feature, max_abs_corr):
    """
    Apply correlation-based rules only.
    We will override manually for conceptual redundancy.
    """
    if max_abs_corr > 0.7:
        decision = "drop: very high redundancy (|r| > 0.7)"
    elif max_abs_corr > 0.5:
        decision = "drop: high redundancy (|r| > 0.5)"
    elif max_abs_corr > 0.4:
        decision = "review: moderate correlation (0.4 < |r| <= 0.5)"
    else:
        decision = "keep candidate: low correlation (|r| <= 0.4)"
    return decision

def get_feature_decisions(df, core_cols, extra_cols):
    """
    Calculates max correlations between extra features and core factors,
    then applies the classification rules to generate a decision table.
    """
    logger.info("Calculating feature redundancy decisions...")
    
    # 1. Validate Columns
    valid_core = [c for c in core_cols if c in df.columns]
    valid_extra = [c for c in extra_cols if c in df.columns]
    
    # 2. Calculate Cross-Correlation Matrix
    # We select all cols first to handle overlaps correctly
    cross_corr = df[valid_core + valid_extra].corr(method='pearson')
    # Slice: Rows = Core, Cols = Extra
    cross_corr_matrix = cross_corr.loc[valid_core, valid_extra]
    
    # 3. Find Max Absolute Correlation for each Extra Feature
    extra_vs_core_abs = cross_corr_matrix.abs()
    max_corr_with_core = extra_vs_core_abs.max(axis=0).sort_values(ascending=False)
    
    # 4. Apply Logic Loop
    extra_decisions = []
    for feature, max_abs_corr in max_corr_with_core.items():
        extra_decisions.append({
            "feature": feature,
            "max_abs_corr_with_core": max_abs_corr,
            "rule_based_decision": classify_extra_feature(feature, max_abs_corr)
        })

    extra_decisions_df = pd.DataFrame(extra_decisions)
    return extra_decisions_df