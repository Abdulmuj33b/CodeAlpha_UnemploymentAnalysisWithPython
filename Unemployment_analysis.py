#!/usr/bin/env python3
"""
Professional Unemployment Analysis Project (India: 2019–2020)

Objective:
- Analyze unemployment rate data (percentage of unemployed people)
- Clean and explore the dataset using Python
- Visualize unemployment trends across time, states, and rural/urban areas
- Quantify and interpret the impact of COVID-19 lockdowns
- Identify temporal and seasonal patterns
- Generate policy-relevant insights suitable for academic or professional submission

Author: Abidoye Abdulmujeeb Abiola
Date: 19-12-2025
"""

# =========================
# 1. LIBRARIES & SETTINGS
# =========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_ind

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (12, 6),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)

COLORS = {"national": "#8E1B1B", "rural": "#1F6F78", "urban": "#E36414"}


# =========================
# 2. DATA LOADING & CLEANING
# =========================

def load_and_clean_data(file_path: Path, debug: bool = False) -> pd.DataFrame:
    """Load unemployment data with robust handling for mislabeled CSV/XLS files.
    If debug=True, prints dataset snapshots at each major cleaning stage.
    """

    df = None

    try:
        df = pd.read_excel(file_path)
        if debug:
            print("\n[DEBUG] Raw data loaded via read_excel:")
            print(df.head())
    except Exception:
        pass

    if df is None:
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if debug:
                print("\n[DEBUG] Raw data loaded via read_csv (utf-8):")
                print(df.head())
        except UnicodeDecodeError:
            df = pd.read_csv(file_path, encoding='latin-1')
            if debug:
                print("\n[DEBUG] Raw data loaded via read_csv (latin-1):")
                print(df.head())

    if df is None:
        raise ValueError("Unable to read dataset")

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("[()%]", "", regex=True)
    )

    if debug:
        print("\n[DEBUG] After standardizing column names:")
        print(df.head())

    # Explicitly handle potential duplicate 'region' column names after standardization
    # If both 'region' and 'region.1' exist, assume 'region' (state) is primary
    # and drop 'region.1'. This avoids creating duplicate column labels.
    if 'region' in df.columns and 'region.1' in df.columns:
        df = df.drop(columns=['region.1'])
        if debug:
            print("\n[DEBUG] Dropped 'region.1' to avoid name conflict:")
            print(df.head())

    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if 'date' in lc:
            col_map[col] = 'date'
        elif 'unemploy' in lc:
            col_map[col] = 'estimated_unemployment_rate'
        elif 'region' in lc or 'state' in lc:
            col_map[col] = 'region'
        elif 'area' in lc or 'urban' in lc or 'rural' in lc:
            col_map[col] = 'area'

    df = df.rename(columns=col_map)

    if debug:
        print("\n[DEBUG] After column mapping:")
        print(df.head())

    if 'area' not in df.columns:
        df['area'] = 'All'

    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['estimated_unemployment_rate'] = pd.to_numeric(
        df['estimated_unemployment_rate'], errors='coerce'
    )

    for col in ['area', 'region']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.title().str.strip()

    if debug:
        print("\n[DEBUG] After type conversion:")
        print(df.head())

    df = df.dropna(subset=['date', 'region', 'estimated_unemployment_rate'])

    if debug:
        print("\n[DEBUG] Final cleaned dataset:")
        print(df.head())
        print("Final shape:", df.shape)

    return df


# =========================
# 3. FEATURE ENGINEERING
# =========================

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['covid_period'] = np.where(
        df['date'] < '2020-03-01',
        'Pre-COVID',
        np.where(df['date'] < '2020-07-01', 'Lockdown', 'Post-Lockdown'),
    )
    return df


# =========================
# 4. ANALYSIS FUNCTIONS
# =========================

def state_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(['region', 'area'])
        .agg(avg_unemployment=('estimated_unemployment_rate', 'mean'))
        .reset_index()
    )


def covid_impact_analysis(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(['covid_period', 'area'])['estimated_unemployment_rate']
        .mean()
        .reset_index()
    )

    pivot = summary.pivot(
        index='area', columns='covid_period', values='estimated_unemployment_rate'
    )
    pivot['absolute_change'] = pivot['Lockdown'] - pivot['Pre-COVID']
    pivot['percentage_change'] = (pivot['absolute_change'] / pivot['Pre-COVID']) * 100

    return pivot.reset_index()


# =========================
# 5. VISUALIZATIONS
# =========================

def plot_national_trend(df: pd.DataFrame):
    trend = df.groupby('date')['estimated_unemployment_rate'].mean()
    plt.plot(trend.index, trend.values, color=COLORS['national'], linewidth=2)
    plt.axvspan(pd.Timestamp('2020-03-25'), pd.Timestamp('2020-06-30'), alpha=0.2)
    plt.title('National Unemployment Rate Trend (2019–2020)')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.tight_layout()
    plt.savefig('national_trend.png', dpi=300)
    plt.close()


def plot_rural_urban_comparison(df: pd.DataFrame):
    sns.lineplot(data=df, x='date', y='estimated_unemployment_rate', hue='area')
    plt.title('Rural vs Urban Unemployment Trends')
    plt.tight_layout()
    plt.savefig('rural_urban_trend.png', dpi=300)
    plt.close()


def plot_monthly_seasonality(df: pd.DataFrame):
    monthly = df.groupby(['month', 'area'])['estimated_unemployment_rate'].mean().reset_index()
    sns.lineplot(data=monthly, x='month', y='estimated_unemployment_rate', hue='area', marker='o')
    plt.title('Seasonal Unemployment Pattern')
    plt.tight_layout()
    plt.savefig('seasonality.png', dpi=300)
    plt.close()


# =========================
# 6. FORECASTING & TESTING
# =========================

def arima_forecast(df: pd.DataFrame, steps: int = 6) -> pd.DataFrame:
    ts = df.groupby('date')['estimated_unemployment_rate'].mean().asfreq('M').interpolate() # Changed 'MS' to 'M'
    adf_p = adfuller(ts.dropna())[1]

    model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12), enforce_stationarity=False)
    results = model.fit(disp=False)
    forecast = results.get_forecast(steps=steps)
    out = forecast.summary_frame()
    out['adf_pvalue'] = adf_p

    plt.plot(ts, label='Observed')
    plt.plot(out['mean'], '--', label='Forecast')
    plt.fill_between(out.index, out['mean_ci_lower'], out['mean_ci_upper'], alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sarima_forecast.png', dpi=300)
    plt.close()

    return out


def lockdown_significance_test(df: pd.DataFrame) -> pd.DataFrame:
    pre = df[df['covid_period'] == 'Pre-COVID']['estimated_unemployment_rate']
    lock = df[df['covid_period'] == 'Lockdown']['estimated_unemployment_rate']
    t, p = ttest_ind(pre, lock, equal_var=False)
    return pd.DataFrame(
        {
            'test': ['Welch t-test'],
            't_statistic': [t],
            'p_value': [p],
            'significant_at_5pct': [p < 0.05],
        }
    )


# =========================
# 7. MAIN PIPELINE
# =========================

def main():
    print("Starting analysis...")

    files = [
        f
        for ext in ('*.xls', '*.xlsx', '*.csv')
        for f in Path('.').glob(ext)
        if 'unemployment' in f.name.lower()
    ]
    if not files:
        raise FileNotFoundError("No unemployment dataset found")

    df = load_and_clean_data(files[0], debug=True)
    df = add_time_features(df)

    state_level_summary(df).to_csv('state_summary.csv', index=False)
    covid_impact_analysis(df).to_csv('covid_impact.csv')
    arima_forecast(df).to_csv('sarima_forecast.csv')
    lockdown_significance_test(df).to_csv('significance_test.csv', index=False)

    plot_national_trend(df)
    plot_rural_urban_comparison(df)
    plot_monthly_seasonality(df)

    df.to_csv('cleaned_unemployment_data.csv', index=False)
    print("Analysis completed successfully")


if __name__ == '__main__':
    main()
