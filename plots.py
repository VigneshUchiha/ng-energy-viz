# Create the requested set of visualizations from the Nigerian energy dataset.
# Plot: histogram, bar chart, pie chart, line, scatter, heatmap, and violin plots

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df_ng_energy = pd.read_csv('nigerian_energy_and_utilities_carbon_footprint.csv', encoding='ascii')
df_ng_energy['timestamp'] = pd.to_datetime(df_ng_energy['timestamp'], errors='coerce')

# Ensure sorted for line plots
df_plot = df_ng_energy.copy()
df_plot = df_plot.sort_values('timestamp')

# 1) Histogram of CO2 intensity
plt.figure(figsize=(7,4))
sns.histplot(df_plot['co2_g_per_kwh'], bins=30, kde=True, color='#4e79a7')
plt.title('Histogram of CO2 intensity (g/kWh)')
plt.xlabel('CO2 g/kWh')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# 1b) Bar chart: average CO2 by region
region_avg = df_plot.groupby('region', as_index=False)['co2_g_per_kwh'].mean()
plt.figure(figsize=(7,4))
sns.barplot(data=region_avg, x='region', y='co2_g_per_kwh', color='#59a14f')
plt.title('Average CO2 intensity by region')
plt.xlabel('Region')
plt.ylabel('Avg CO2 g/kWh')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 1c) Pie chart: overall average generation mix
mix_cols = ['gas_share_pct','hydro_share_pct','solar_share_pct','wind_share_pct']
overall_mix = df_plot[mix_cols].mean().clip(lower=0)
plt.figure(figsize=(5,5))
plt.pie(overall_mix.values, labels=[c.replace('_',' ').title() for c in overall_mix.index], autopct='%1.1f%%', startangle=140)
plt.title('Overall average generation mix')
plt.tight_layout()
plt.show()

# 2) Line plot: CO2 over time (median by day)
df_daily = df_plot.set_index('timestamp').resample('D')['co2_g_per_kwh'].median().dropna().reset_index()
plt.figure(figsize=(8,4))
sns.lineplot(data=df_daily, x='timestamp', y='co2_g_per_kwh', color='#e15759')
plt.title('Daily median CO2 intensity over time')
plt.xlabel('Date')
plt.ylabel('CO2 g/kWh')
plt.tight_layout()
plt.show()

# Scatter plot: CO2 vs gas share
plt.figure(figsize=(7,4))
sns.scatterplot(data=df_plot.sample(min(5000, len(df_plot)), random_state=42), x='gas_share_pct', y='co2_g_per_kwh', hue='region', alpha=0.6)
plt.title('Scatter: Gas share vs CO2 intensity')
plt.xlabel('Gas share (%)')
plt.ylabel('CO2 g/kWh')
plt.tight_layout()
plt.show()

# 3) Heatmap: correlation matrix
corr_cols = ['co2_g_per_kwh','gas_share_pct','hydro_share_pct','solar_share_pct','wind_share_pct']
corr = df_plot[corr_cols].corr(numeric_only=True)
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation heatmap')
plt.tight_layout()
plt.show()

# Violin plot: CO2 by region
plt.figure(figsize=(8,4))
sns.violinplot(data=df_plot, x='region', y='co2_g_per_kwh', inner='box', scale='width')
plt.title('Violin plot: CO2 intensity by region')
plt.xlabel('Region')
plt.ylabel('CO2 g/kWh')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

print('Plotted histogram, bar, pie, line, scatter, heatmap, and violin plots.')
