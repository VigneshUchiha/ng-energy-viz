# Analysis and visualization for Nigerian Energy Dataset
# Performs data cleaning, EDA, and comprehensive plotting

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load and Clean Data
print("Loading data...")
df_ng_energy = pd.read_csv('nigerian_energy_and_utilities_carbon_footprint.csv', encoding='ascii')

print("Cleaning and transforming data...")
df_ed = df_ng_energy.copy()
df_ed['timestamp'] = pd.to_datetime(df_ed['timestamp'], errors='coerce')

# Remove rows with missing critical fields
critical_cols = ['timestamp','region','co2_g_per_kwh']
df_ed = df_ed.dropna(subset=critical_cols)
print(f"After removing missing critical fields: {len(df_ed)} rows")

# Clip shares to [0,100] and coerce to numeric
share_cols = ['gas_share_pct','hydro_share_pct','solar_share_pct','wind_share_pct']
for c in share_cols:
    df_ed[c] = pd.to_numeric(df_ed[c], errors='coerce').clip(lower=0, upper=100)

# Remove duplicates
before_dups = len(df_ed)
df_ed = df_ed.drop_duplicates()
after_dups = len(df_ed)
print(f"Duplicates removed: {before_dups - after_dups}")

# Step 2: Feature Engineering
print("Performing feature engineering...")
df_ed['date'] = df_ed['timestamp'].dt.date
df_ed['year'] = df_ed['timestamp'].dt.year
df_ed['month'] = df_ed['timestamp'].dt.to_period('M').astype(str)
df_ed['renewable_share_pct'] = df_ed[['hydro_share_pct','solar_share_pct','wind_share_pct']].sum(axis=1)
df_ed['fossil_share_pct'] = df_ed['gas_share_pct']

# Sanity check for total shares
df_ed = df_ed[(df_ed['renewable_share_pct'] + df_ed['fossil_share_pct']).between(0, 120)]
print(f"After sanity check: {len(df_ed)} rows")

cleaned_df = df_ed.copy()  # Use cleaned data for analysis

# Step 3: EDA - Exploratory Data Analysis
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

print("\n--- Sample Data ---")
print(cleaned_df.head(10))

print("\n--- Summary Statistics ---")
desc_out = cleaned_df[['co2_g_per_kwh'] + share_cols + ['renewable_share_pct']].describe()
print(desc_out)

print("\n--- Missing Values ---")
missing_out = cleaned_df.isna().mean().sort_values(ascending=False).head(10)
print(missing_out)

print("\n--- Region-level Aggregates ---")
region_agg = cleaned_df.groupby('region', as_index=False).agg(
    avg_co2=('co2_g_per_kwh','mean'),
    p50_co2=('co2_g_per_kwh','median'),
    n_obs=('co2_g_per_kwh','size')
)
print(region_agg)

# Step 4: Visualizations
print("\n" + "="*50)
print("VISUALIZATIONS")
print("="*50)

# Prepare data for plotting (raw data for some plots)
df_plot = df_ng_energy.copy()
df_plot = df_plot.sort_values('timestamp')
df_plot['timestamp'] = pd.to_datetime(df_plot['timestamp'], errors='coerce')

# 1) EDA Visual: Average CO2 by region (cleaned data)
print("1. Average CO2 by region")
plt.figure(figsize=(7,4))
sns.barplot(data=region_agg.sort_values('avg_co2', ascending=False), x='region', y='avg_co2', color='#4e79a7')
plt.title('Avg CO2 g/kWh by region (cleaned)')
plt.xlabel('Region')
plt.ylabel('Avg CO2 g/kWh')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 2) Histogram of CO2 intensity
print("2. Histogram of CO2 intensity")
plt.figure(figsize=(7,4))
sns.histplot(df_plot['co2_g_per_kwh'], bins=30, kde=True, color='#4e79a7')
plt.title('Histogram of CO2 intensity (g/kWh)')
plt.xlabel('CO2 g/kWh')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 3) Bar chart: average CO2 by region (all data)
print("3. Bar chart: Average CO2 by region")
region_avg = df_plot.groupby('region', as_index=False)['co2_g_per_kwh'].mean()
plt.figure(figsize=(7,4))
sns.barplot(data=region_avg, x='region', y='co2_g_per_kwh', color='#59a14f')
plt.title('Average CO2 intensity by region')
plt.xlabel('Region')
plt.ylabel('Avg CO2 g/kWh')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 4) Pie chart: overall average generation mix
print("4. Pie chart: Energy generation mix")
mix_cols = ['gas_share_pct','hydro_share_pct','solar_share_pct','wind_share_pct']
overall_mix = df_plot[mix_cols].mean().clip(lower=0)
plt.figure(figsize=(5,5))
plt.pie(overall_mix.values, labels=[c.replace('_',' ').title() for c in overall_mix.index], autopct='%1.1f%%', startangle=140)
plt.title('Overall average generation mix')
plt.tight_layout()
plt.show()

# 5) Line plot: CO2 over time (median by day)
print("5. Line plot: CO2 over time")
df_daily = df_plot.set_index('timestamp').resample('D')['co2_g_per_kwh'].median().dropna().reset_index()
plt.figure(figsize=(8,4))
sns.lineplot(data=df_daily, x='timestamp', y='co2_g_per_kwh', color='#e15759')
plt.title('Daily median CO2 intensity over time')
plt.xlabel('Date')
plt.ylabel('CO2 g/kWh')
plt.tight_layout()
plt.show()

# 6) Scatter plot: CO2 vs gas share
print("6. Scatter plot: Gas share vs CO2 intensity")
plt.figure(figsize=(7,4))
sns.scatterplot(data=df_plot.sample(min(5000, len(df_plot)), random_state=42), x='gas_share_pct', y='co2_g_per_kwh', hue='region', alpha=0.6)
plt.title('Scatter: Gas share vs CO2 intensity')
plt.xlabel('Gas share (%)')
plt.ylabel('CO2 g/kWh')
plt.tight_layout()
plt.show()

# 7) Heatmap: correlation matrix
print("7. Heatmap: Correlation matrix")
corr_cols = ['co2_g_per_kwh','gas_share_pct','hydro_share_pct','solar_share_pct','wind_share_pct']
corr = df_plot[corr_cols].corr(numeric_only=True)
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation heatmap')
plt.tight_layout()
plt.show()

# 8) Violin plot: CO2 by region
print("8. Violin plot: CO2 intensity by region")
plt.figure(figsize=(8,4))
sns.violinplot(data=df_plot, x='region', y='co2_g_per_kwh', inner='box', scale='width')
plt.title('Violin plot: CO2 intensity by region')
plt.xlabel('Region')
plt.ylabel('CO2 g/kWh')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

print("\nAnalysis complete! All visualizations displayed.")
