import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from scipy import stats  

import matplotlib.pyplot as plt
import seaborn as sns

def plot_model_performance(results_df, metric='R2 Mean', std_col='R2 Std'):
    """
    Plots a bar chart of the model bake-off results using Matplotlib/Seaborn.
    Includes error bars to show stability (Standard Deviation).
    """
    plt.figure(figsize=(10, 6))
    
    # Create the bar chart
    ax = sns.barplot(
        data=results_df,
        x='Model',
        y=metric,
        palette='viridis',
        capsize=0.1,  # Adds caps to error bars
        errorbar=None  # We manually handle error bars below if pre-calculated
    )
    
    # Add Error Bars manually since we calculated 'R2 Std' in the baseline logic
    if std_col in results_df.columns:
        plt.errorbar(
            x=range(len(results_df)),
            y=results_df[metric],
            yerr=results_df[std_col],
            fmt='none',
            c='black',
            capsize=5
        )

    # Annotate bars with values
    for i, v in enumerate(results_df[metric]):
        ax.text(
            i, 
            v + (0.01 if v > 0 else -0.02),  # Position text slightly above/below bar
            f"{v:.4f}", 
            ha='center', 
            va='bottom' if v > 0 else 'top',
            fontweight='bold'
        )

    plt.title(f"Model Archetype Bake-Off ({metric})", fontsize=14)
    plt.xlabel("Model Archetype")
    plt.ylabel(metric)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--') # Zero line
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_distribution_diagnostics(series, title_prefix="Data", figsize=(15, 5)):
    """
    Plots a 3-panel diagnostic for a distribution:
    1. Histogram with Median Line
    2. Q-Q Plot (Normality check)
    3. Boxplot (Outlier check)
    """
    # 1. Prepare Data (Drop NaNs to avoid plotting errors)
    clean_data = series.dropna()
    
    # 2. Setup Figure
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # --- Panel 1: Histogram ---
    ax0 = axes[0]
    ax0.hist(
        clean_data,
        bins=50,
        edgecolor='black',
        alpha=0.7
    )
    # Add Median Line
    median_val = clean_data.median()
    ax0.axvline(
        median_val,
        color='red',
        linestyle='--',
        label=f'Median: {median_val:.4f}'
    )
    ax0.set_title(f'{title_prefix} Distribution')
    ax0.set_xlabel('Value')
    ax0.set_ylabel('Frequency')
    ax0.legend()

    # --- Panel 2: Q-Q Plot ---
    ax1 = axes[1]
    stats.probplot(clean_data, dist="norm", plot=ax1)
    ax1.set_title('Q-Q Plot (Normality Check)')
    
    # --- Panel 3: Boxplot ---
    ax2 = axes[2]
    ax2.boxplot(clean_data, vert=True)
    ax2.set_title(f'{title_prefix} Box Plot')
    ax2.set_ylabel('Value')
    
    # Final Layout
    plt.tight_layout()
    plt.show()

def plot_equity_curve(returns, benchmark=None, save_path=None):
    """
    Plots cumulative returns over time.
    """
    plt.figure(figsize=(12, 6))
    
    cumulative = (1 + returns).cumprod()
    plt.plot(cumulative, label='Strategy')
    
    if benchmark is not None:
        bench_cum = (1 + benchmark).cumprod()
        plt.plot(bench_cum, label='Benchmark', alpha=0.7)
        
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """
    Plots XGBoost feature importance.
    """
    plt.figure(figsize=(10, 8))
    # Basic xgb importance
    xgb.plot_importance(model, max_num_features=20)
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def plot_log_comparison(df, col1, col2, n_samples=500):
    """
    Samples data, applies log1p transform, and plots comparison using Matplotlib.
    Handles MultiIndex by resetting index to find the Date column.
    """
    # 1. Sample, Sort, and Reset Index (The critical fix)
    # We select only the needed columns first to avoid issues with other NaN columns
    plot_data = df[[col1, col2]].dropna().sample(n=n_samples, random_state=42).sort_index().reset_index()

    # 2. Find the Date column for the X-axis
    # reset_index() might name it 'Date', 'level_0', or something else.
    # We look for the first datetime column.
    date_col = None
    for c in plot_data.columns:
        if pd.api.types.is_datetime64_any_dtype(plot_data[c]):
            date_col = c
            break
            
    x_axis = plot_data[date_col] if date_col else plot_data.index

    # 3. Apply Log Transform
    val1 = np.log1p(plot_data[col1])
    val2 = np.log1p(plot_data[col2])

    # 4. Plot
    plt.figure(figsize=(12, 6))
    
    plt.plot(x_axis, val1, label=f'log1p({col1})', alpha=0.8)
    plt.plot(x_axis, val2, label=f'log1p({col2})', alpha=0.8)

    plt.title("After log1p transform: scales closer, still not standardized")
    plt.xlabel("Date" if date_col else "Index")
    plt.ylabel("log1p(value)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df, cols, title="Correlation Heatmap", figsize=(12, 4)):
    """
    Calculates and plots a Pearson correlation heatmap for the specified columns.
    """
    # Calculate correlation
    corr_matrix = df[cols].corr(method="pearson")

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.5,
        linecolor="white"
    )
    
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_cross_correlation_heatmap(df, row_cols, col_cols, title="Cross-Correlation Heatmap", figsize=(14, 6)):
    """
    Calculates cross-correlation between two sets of features (rows vs columns) 
    and plots the heatmap.
    """
    # 1. Identify valid columns present in the DataFrame
    valid_rows = [c for c in row_cols if c in df.columns]
    valid_cols = [c for c in col_cols if c in df.columns]
    
    # 2. Calculate Correlation
    # We combine them to calculate the full matrix first to handle any overlaps correctly
    combined_cols = list(set(valid_rows + valid_cols))
    full_corr = df[combined_cols].corr(method="pearson")
    
    # 3. Slice the specific Cross-Correlation (Rows x Cols)
    cross_corr_matrix = full_corr.loc[valid_rows, valid_cols]

    # 4. Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cross_corr_matrix,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        linecolor="white"
    )

    plt.title(title)
    plt.xlabel("Extra Features")
    plt.ylabel("Core Factors")
    plt.tight_layout()
    plt.show()

def plot_grouped_heatmap(df, group_col, value_cols, title="Grouped Heatmap", figsize=(12, 5)):
    """
    Groups the DataFrame by a categorical column (e.g., Sector), calculates the mean 
    of the value columns, and plots a heatmap.
    """
    # 1. Group and Aggregate
    grouped_data = df.groupby(group_col)[value_cols].mean().sort_index()

    # 2. Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        grouped_data,
        annot=True,
        fmt=".4f",
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        linecolor="white"
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_box_distribution(df, cols, title="Distribution of Features", figsize=(10, 8)):
    """
    Plots a boxplot for the specified columns to visualize distributions.
    Automatically drops NaNs and melts the dataframe for Seaborn.
    """
    # 1. Prepare Data
    df_box = df[cols].dropna()
    print(f"Boxplot sample size: {len(df_box):,} rows")
    
    # Melt into long format
    plot_data = df_box.melt(var_name='Feature', value_name='Value')

    # 2. Plot
    plt.figure(figsize=figsize)
    sns.boxplot(
        data=plot_data,
        x='Value',
        y='Feature',
        showfliers=False
    )

    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Feature")
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()