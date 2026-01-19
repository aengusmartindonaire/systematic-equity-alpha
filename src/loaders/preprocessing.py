import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score, KFold
from src.utils.logger import setup_logger

logger = setup_logger()

class Preprocessor:
    def __init__(self):
        pass

    def clean_gics(self, df):
        """Hierarchical fill for GICS columns."""
        logger.info("Cleaning GICS columns...")
        
        # Fill Sector NaNs with 'Other'
        df['GICS_Sector_Name'] = df['GICS_Sector_Name'].fillna('Other')
        
        # Fill Industry with Sector
        ind_na = df['GICS_Industry_Name'].isna()
        df.loc[ind_na, 'GICS_Industry_Name'] = df.loc[ind_na, 'GICS_Sector_Name']
        
        # Fill SubInd with Industry
        sub_na = df['GICS_SubInd_Name'].isna()
        df.loc[sub_na, 'GICS_SubInd_Name'] = df.loc[sub_na, 'GICS_Sector_Name']
        
        return df

    def encode_gics(self, df, rare_threshold=500):
        """Handles rare categories and One-Hot Encodes."""
        gics_cols = ['GICS_Sector_Name', 'GICS_Industry_Name', 'GICS_SubInd_Name']
        
        # Handle Rare Categories
        for col in gics_cols:
            vc = df[col].value_counts()
            rare_cats = vc[vc < rare_threshold].index
            if len(rare_cats) > 0:
                logger.info(f"Setting {len(rare_cats)} rare categories in {col} to NaN")
                df.loc[df[col].isin(rare_cats), col] = np.nan
        
        # One Hot Encode
        df_encoded = pd.get_dummies(df, columns=gics_cols, prefix=gics_cols, drop_first=False)
        return df_encoded

    def run_gics_bakeoff(self, df, numeric_features, target_col='FwdRet', start_date='202301'):
        """
        Runs Bayesian Ridge CV to find best GICS level.
        """
        logger.info("Running GICS Bayesian Ridge Bake-off...")
        
        # Filter for recent data
        gdf = df.loc[df['Date'] >= start_date].copy()
        
        levels = ['GICS_Sector_Name', 'GICS_Industry_Name', 'GICS_SubInd_Name']
        results = []
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for level in levels:
            # Find all dummy columns for this level
            dummies = [c for c in df.columns if c.startswith(level + '_')]
            
            if not dummies:
                continue
                
            # Prepare X and y
            X = gdf[numeric_features + dummies].copy()
            y = gdf[target_col]
            
            # Drop NaNs
            mask = X.notna().all(axis=1) & y.notna()
            X = X.loc[mask]
            y = y.loc[mask]
            
            # Run CV
            model = BayesianRidge()
            scores = cross_val_score(model, X, y, cv=kfold, scoring='r2', n_jobs=-1)
            
            results.append({
                "GICS_Level": level,
                "Mean_R2": scores.mean(),
                "Std_R2": scores.std()
            })
            
        return pd.DataFrame(results).sort_values("Mean_R2", ascending=False)