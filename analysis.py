# Stats project on Nigeria energy data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns


def plot_relational_plot(df):
    """
    Create a line plot showing CO2 intensity trend over time.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate daily median CO2 intensity
    df_sorted = df.sort_values('timestamp')
    df_daily = df_sorted.set_index('timestamp').resample('D')[
        'co2_g_per_kwh'
    ].median().dropna().reset_index()
    
    # Create line plot
    sns.lineplot(
        data=df_daily,
        x='timestamp',
        y='co2_g_per_kwh',
        color='#e15759',
        linewidth=2,
        ax=ax
    )
    
    ax.set_title('Daily Median CO2 Intensity Over Time',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('CO2 Intensity (g/kWh)', fontsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('relational_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


def plot_categorical_plot(df):
    """
    Create a pie chart showing the overall average energy generation mix
    across all regions.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Define energy mix columns
    mix_cols = ['gas_share_pct', 'hydro_share_pct',
                'solar_share_pct', 'wind_share_pct']
    
    # Calculate overall average for each energy source
    overall_mix = df[mix_cols].mean().clip(lower=0)
    
    # Create labels with better formatting
    labels = ['Gas', 'Hydro', 'Solar', 'Wind']
    
    # Define colors for better visualization
    colors = ['#ff9999', '#66b3ff', '#ffcc99', '#99ff99']
    
    # Create pie chart
    ax.pie(
        overall_mix.values,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=(0.05, 0, 0, 0),
        textprops={'fontsize': 11}
    )
    
    ax.set_title('Overall Average Energy Generation Mix',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('categorical_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


def plot_statistical_plot(df):
    """
    Create a correlation heatmap showing relationships between
    CO2 intensity and energy generation shares.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataframe with energy data
        
    Returns
    -------
    None
        Saves plot to 'statistical_plot.png'
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Select columns for correlation matrix
    corr_cols = ['co2_g_per_kwh', 'gas_share_pct', 'hydro_share_pct',
                 'solar_share_pct', 'wind_share_pct']
    
    # Calculate correlation matrix
    corr_matrix = df[corr_cols].corr()
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'},
        ax=ax
    )
    
    ax.set_title('Correlation Heatmap: CO2 and Energy Generation Shares',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('statistical_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


def statistical_analysis(df, col: str):
    """
    Calculate statistical moments for a specified column.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned dataframe with energy data
    col : str
        Column name to analyze
        
    Returns
    -------
    tuple
        (mean, standard deviation, skewness, excess kurtosis)
    """
    data = df[col].dropna()
    
    mean = np.mean(data)
    stddev = np.std(data, ddof=1)
    skew = stats.skew(data)
    excess_kurtosis = stats.kurtosis(data)
    
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    print("="*60)
    print("PREPROCESSING")
    print("="*60)
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    print(f"\nInitial dataset shape: {df_clean.shape}")
    print("\n--- Initial Data Sample ---")
    print(df_clean.head())
    
    # Convert timestamp to datetime
    print("\nConverting timestamp to datetime...")
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'],
                                           errors='coerce')
    
    # Remove rows with missing critical fields
    critical_cols = ['timestamp', 'region', 'co2_g_per_kwh']
    before_drop = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_cols)
    after_drop = len(df_clean)
    print(f"Removed {before_drop - after_drop} rows with missing "
          f"critical fields")
    
    # Clip share percentages to valid range
    share_cols = ['gas_share_pct', 'hydro_share_pct',
                  'solar_share_pct', 'wind_share_pct']
    print("\nClipping share percentages to [0, 100] range...")
    for col in share_cols:
        df_clean[col] = pd.to_numeric(df_clean[col],
                                      errors='coerce').clip(lower=0, upper=100)
    
    # Remove duplicates
    before_dups = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after_dups = len(df_clean)
    print(f"Removed {before_dups - after_dups} duplicate rows")
    
    # Feature Engineering
    print("\nPerforming feature engineering...")
    df_clean['date'] = df_clean['timestamp'].dt.date
    df_clean['year'] = df_clean['timestamp'].dt.year
    df_clean['month'] = df_clean['timestamp'].dt.to_period('M').astype(str)
    
    # Calculate renewable and fossil shares
    df_clean['renewable_share_pct'] = df_clean[
        ['hydro_share_pct', 'solar_share_pct', 'wind_share_pct']
    ].sum(axis=1)
    df_clean['fossil_share_pct'] = df_clean['gas_share_pct']
    
    # Sanity check: total shares should be between 0 and 120
    total_share = (df_clean['renewable_share_pct'] +
                   df_clean['fossil_share_pct'])
    before_sanity = len(df_clean)
    df_clean = df_clean[total_share.between(0, 120)]
    after_sanity = len(df_clean)
    print(f"Removed {before_sanity - after_sanity} rows failing "
          f"sanity check")
    
    print(f"\nFinal dataset shape: {df_clean.shape}")
    
    # Display summary statistics
    print("\n--- Summary Statistics ---")
    summary_cols = ['co2_g_per_kwh'] + share_cols + ['renewable_share_pct']
    print(df_clean[summary_cols].describe())
    
    # Display missing values
    print("\n--- Missing Values (Top 10) ---")
    missing = df_clean.isna().mean().sort_values(ascending=False).head(10)
    print(missing)
    
    # Display correlation matrix
    print("\n--- Correlation Matrix ---")
    corr_cols = ['co2_g_per_kwh', 'gas_share_pct', 'hydro_share_pct',
                 'solar_share_pct', 'wind_share_pct']
    print(df_clean[corr_cols].corr())
    
    # Region-level aggregates
    print("\n--- Region-level Aggregates ---")
    region_agg = df_clean.groupby('region', as_index=False).agg(
        avg_co2=('co2_g_per_kwh', 'mean'),
        median_co2=('co2_g_per_kwh', 'median'),
        n_observations=('co2_g_per_kwh', 'size')
    )
    print(region_agg)
    
    print("\nPreprocessing complete!")
    print("="*60 + "\n")
    
    return df_clean


def writing(moments, col):
    print("="*60)
    print("STATISTICAL ANALYSIS RESULTS")
    print("="*60)
    print(f'\nFor the attribute "{col}":')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')
    
    # Interpret skewness
    if moments[2] < -0.5:
        skew_interp = "left skewed"
    elif moments[2] > 0.5:
        skew_interp = "right skewed"
    else:
        skew_interp = "not skewed"
    
    # Interpret kurtosis
    if moments[3] < -1:
        kurt_interp = "platykurtic"
    elif moments[3] > 1:
        kurt_interp = "leptokurtic"
    else:
        kurt_interp = "mesokurtic"
    
    print(f'The data was {skew_interp} and {kurt_interp}.')
    print("="*60 + "\n")
    
    return


def main():
    # Load the dataset
    print("Loading Nigerian Energy Dataset...")
    df = pd.read_csv('data.csv',
                     encoding='ascii')
    
    # Preprocess the data
    df = preprocessing(df)
    
    # Choose column for statistical analysis
    col = 'co2_g_per_kwh'
    
    # Generate plots
    print("Generating visualizations...")
    plot_relational_plot(df)
    print("✓ Line plot saved")
    
    plot_statistical_plot(df)
    print("✓ Heatmap saved")
    
    plot_categorical_plot(df)
    print("✓ Pie chart saved")
    
    # Perform statistical analysis
    moments = statistical_analysis(df, col)
    
    # Write results
    writing(moments, col)
    
    print("Analysis complete! All plots saved and statistics calculated.")
    
    return


if __name__ == '__main__':
    main()
