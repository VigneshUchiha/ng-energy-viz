# Analysis and visualization for Nigerian Electricity Data (Excel file)
# Performs data cleaning, EDA, and comprehensive plotting

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

FILEPATH = 'Nigeria Electricity Data.xlsx'

# Helper function to read all sheets
def read_all_sheets_from_excel(filepath):
    xls = pd.ExcelFile(filepath)
    dataframes = {}
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        dataframes[sheet_name] = df
    return dataframes

# This cell loads the Excel file using the provided helper, inspects sheets, and shows heads for quick EDA setup.
dataframes = read_all_sheets_from_excel(FILEPATH)
for k, v in dataframes.items():
    print('dataframe: ' + k)
    print(v.head(15))

# Load all sheets into dataframes (already done) and perform basic cleaning/transformations.
# Then create combined tidy tables and produce EDA visualizations (hist/bar/pie, line/scatter, heatmap/box/violin).

# Read all sheets
xls = pd.ExcelFile(FILEPATH)
all_sheets = {}
for name in xls.sheet_names:
    df = pd.read_excel(FILEPATH, sheet_name=name)
    all_sheets[name] = df

print(list(all_sheets.keys()))
print('Loaded sheets')

# Clean and transform sheets into tidy dataframes, then create EDA plots:
# - Standardize column names
# - Combine state-level sheets into one long table
# - Create summary tables for Regions, Residences, Totals
# - Plot: bar/pie, line/scatter, and heatmap/box/violin

xls = pd.ExcelFile('Nigeria Electricity Data.xlsx')
raw = {s: pd.read_excel('Nigeria Electricity Data.xlsx', sheet_name=s) for s in xls.sheet_names}

for k in raw:
    raw[k].columns = [c.strip() for c in raw[k].columns]

state_sheets = ['2013 DHS','2015 MIS','2018 DHS','2021 MIS']
state_df = pd.concat([raw[s] for s in state_sheets], ignore_index=True)
state_df = state_df.rename(columns={'Households with electricity':'Households','Population with electricity':'Population'})
state_df['Households'] = pd.to_numeric(state_df['Households'], errors='coerce')
state_df['Population'] = pd.to_numeric(state_df['Population'], errors='coerce')

regions = raw['Regions'].rename(columns={'Characteristic':'Region','Households with electricity':'Households','Population with electricity':'Population'})
resid = raw['Residences'].rename(columns={'Characteristic':'Residence','Households with electricity':'Households','Population with electricity':'Population'})
totals = raw['Totals'].rename(columns={'Characteristic':'Level','Households with electricity':'Households','Population with electricity':'Population'})

print(state_df.head())
print(regions.head())
print(resid.head())
print(totals.head())
print('Cleaned and combined tables ready')

# Create EDA visualizations: histogram/bar/pie, line/scatter, heatmap/box/violin using the cleaned tables prepared above.
# We assume state_df, regions, resid, totals exist from prior cell.

# 1) Bar chart: Average state household electricity by survey (top 10 states in 2021)
sns.set(style='whitegrid')
latest = state_df[state_df['Survey'] == '2021 MIS']
latest_top = latest.nlargest(10, 'Households')
plt.figure(figsize=(8,5))
sns.barplot(data=latest_top, x='Households', y='State', palette='viridis')
plt.title('Top 10 States by Household Electricity Access (2021 MIS)')
plt.xlabel('Households with electricity (%)')
plt.ylabel('State')
plt.tight_layout()
plt.show()
print('Plotted bar chart of top 10 states for 2021 MIS')

# 2) Pie chart: Regional share of Household electricity (weighted by mean)
reg_2021 = regions[regions['Survey'] == '2021 MIS'].copy()
plt.figure(figsize=(6,6))
plt.pie(reg_2021['Households'], labels=reg_2021['Region'], autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title('Regional distribution of household electricity access (2021 MIS)')
plt.tight_layout()
plt.show()
print('Plotted pie chart for regions (2021 MIS)')

# 3) Histogram: Distribution across all states all years of Population electricity access
plt.figure(figsize=(7,4))
sns.histplot(state_df['Population'].dropna(), bins=20, kde=True, color='steelblue')
plt.title('Distribution of population electricity access across states (all years)')
plt.xlabel('Population with electricity (%)')
plt.tight_layout()
plt.show()
print('Plotted histogram of population access across all states and years')

# 4) Line graph: National totals over time (households vs population)
# Extract year from Survey
def extract_year(s):
    try:
        parts = str(s).split(' ')
        return int(parts[0])
    except Exception:
        return np.nan

totals2 = totals.copy()
totals2['Year'] = totals2['Survey'].apply(extract_year)
plot_totals = totals2.dropna(subset=['Year']).sort_values('Year')
plt.figure(figsize=(8,4))
plt.plot(plot_totals['Year'], plot_totals['Households'], marker='o', label='Households')
plt.plot(plot_totals['Year'], plot_totals['Population'], marker='s', label='Population')
plt.title('Nigeria electricity access totals over time')
plt.ylabel('Percent with electricity')
plt.xlabel('Year')
plt.legend()
plt.tight_layout()
plt.show()
print('Plotted line chart for national totals over time')

# 5) Scatter: State households vs population with regression line
plt.figure(figsize=(6,5))
sns.regplot(data=state_df, x='Households', y='Population', scatter_kws={'alpha':0.5}, line_kws={'color':'crimson'})
plt.title('State household vs population electricity access')
plt.tight_layout()
plt.show()
print('Plotted scatter with regression')

# 6) Heatmap: Regional values for households vs population (2021)
reg_pivot = reg_2021.melt(id_vars=['Region'], value_vars=['Households','Population'], var_name='Metric', value_name='Percent')
heat = reg_pivot.pivot(index='Region', columns='Metric', values='Percent')
plt.figure(figsize=(5,4))
sns.heatmap(heat, annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('Regional electricity access (2021 MIS)')
plt.tight_layout()
plt.show()
print('Plotted heatmap of regions (2021)')

# 7) Box and Violin plots: distribution by survey across states
plt.figure(figsize=(8,4))
sns.boxplot(data=state_df, x='Survey', y='Households', color='lightgray')
sns.stripplot(data=state_df, x='Survey', y='Households', color='dodgerblue', alpha=0.4)
plt.title('Distribution of state household access by survey (box + points)')
plt.tight_layout()
plt.show()
print('Plotted box plot with jittered points by survey')

plt.figure(figsize=(8,4))
sns.violinplot(data=state_df, x='Survey', y='Population', inner='quartile', palette='Pastel1')
plt.title('Distribution of state population access by survey (violin)')
plt.tight_layout()
plt.show()
print('Plotted violin plot by survey')
