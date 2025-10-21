# Clean and transform + EDA for the Nigerian energy dataset, then show quick visuals to guide next steps.
# Steps:
# - Load (if not already), parse timestamp, handle duplicates/NA, basic sanity checks
# - Feature engineering: date parts, renewable share, fossil share
# - EDA: head, basic summary, missingness, and quick region-level aggregates + one small viz

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df_ng_energy = pd.read_csv('nigerian_energy_and_utilities_carbon_footprint.csv', encoding='ascii')

# Clean: parse timestamp and drop obvious bad rows
df_ed = df_ng_energy.copy()
df_ed['timestamp'] = pd.to_datetime(df_ed['timestamp'], errors='coerce')

# Remove rows with missing critical fields
critical_cols = ['timestamp','region','co2_g_per_kwh']
df_ed = df_ed.dropna(subset=critical_cols)

# Clip shares to [0,100] and coerce to numeric
share_cols = ['gas_share_pct','hydro_share_pct','solar_share_pct','wind_share_pct']
for c in share_cols:
    df_ed[c] = pd.to_numeric(df_ed[c], errors='coerce').clip(lower=0, upper=100)

# Remove duplicates
before_dups = len(df_ed)
df_ed = df_ed.drop_duplicates()
after_dups = len(df_ed)

# Feature engineering
df_ed['date'] = df_ed['timestamp'].dt.date
df_ed['year'] = df_ed['timestamp'].dt.year
df_ed['month'] = df_ed['timestamp'].dt.to_period('M').astype(str)
df_ed['renewable_share_pct'] = df_ed[['hydro_share_pct','solar_share_pct','wind_share_pct']].sum(axis=1)
df_ed['fossil_share_pct'] = df_ed['gas_share_pct']

# Sanity: filter rows where total shares are plausible (0-120 to be tolerant)
df_ed = df_ed[(df_ed['renewable_share_pct'] + df_ed['fossil_share_pct']).between(0, 120)]

# Store cleaned dataframe for later plotting
cleaned_df = df_ed.copy()

# EDA summaries
head_out = cleaned_df.head(10)
desc_out = cleaned_df[['co2_g_per_kwh'] + share_cols + ['renewable_share_pct']].describe()
missing_out = cleaned_df.isna().mean().sort_values(ascending=False).head(10)

# Region-level aggregates
region_agg = cleaned_df.groupby('region', as_index=False).agg(
    avg_co2=('co2_g_per_kwh','mean'),
    p50_co2=('co2_g_per_kwh','median'),
    n_obs=('co2_g_per_kwh','size')
)

# Quick small viz: average CO2 by region
plt.figure(figsize=(7,4))
sns.barplot(data=region_agg.sort_values('avg_co2', ascending=False), x='region', y='avg_co2', color='#4e79a7')
plt.title('Avg CO2 g/kWh by region (cleaned)')
plt.xlabel('Region')
plt.ylabel('Avg CO2 g/kWh')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

print(head_out)
print(desc_out)
print(missing_out)
print(region_agg)
print('Cleaned and engineered features; produced head/summary/missingness and a region bar chart.')
