% Professional Unemployment Analysis — Slide Deck
% Author: Your Name
% Date: December 2025

# Slide 1: Title & Objective

- Professional Unemployment Analysis (India: 2019–2020)
- Objective: measure COVID-19 lockdown impact, seasonal patterns, and short-term forecasting

# Slide 2: Data & Cleaning

- Source files: `Unemployment in India.xls`, `Unemployment_Rate_upto_11_2020.xls`
- Key cleaning steps: header normalization, date parsing (day-first), numeric coercion, deduplication

# Slide 3: Methodology

- Aggregation: national, state (region), area (rural/urban)
- Feature engineering: month, year, COVID period (Pre-COVID, Lockdown, Post-Lockdown)
- Tests: Welch's t-test for lockdown vs pre-COVID
- Forecasting: SARIMA with ARIMA and AR(1) OLS fallbacks

# Slide 4: Key Visuals

- National trend (show `national_trend.png`)
- Rural vs Urban (show `rural_urban_trend.png`)
- Seasonality (show `seasonality.png`)

# Slide 5: COVID Impact & Stats

- Pivot: average unemployment by `covid_period` × `area`
- Welch t-test results: p-value and interpretation (significant at 5%)

# Slide 6: Forecast Results

- Forecast method: stabilized SARIMA (fallbacks used where necessary)
- Show `sarima_forecast.png` and summarize next 6 months

# Slide 7: Policy Recommendations & Limitations

- Recommendations: urban counter-cyclical programs, expand rural safety nets, skills + digital inclusion
- Limitations: sample length, measurement of informal sector, aggregation choices
