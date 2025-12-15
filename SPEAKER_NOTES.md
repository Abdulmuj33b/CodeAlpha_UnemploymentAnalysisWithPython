# Speaker Notes — Professional Unemployment Analysis

Slide 1 — Title & Objective
- Say: "Hello — I'm presenting a professional analysis of unemployment in India for 2019–2020. The goals: quantify the lockdown impact, identify seasonal/temporal patterns, and produce short-term forecasts to inform policy."

Slide 2 — Data & Cleaning
- Say: "We used two main source files. Many files have inconsistent headers; we normalize columns, parse dates with day-first, coerce numeric types, and drop invalid rows. The pipeline auto-discovers files with 'unemployment' in their names and concatenates them, deduplicating records."

Slide 3 — Methodology
- Say: "We created time features (year, month), and defined COVID periods: Pre-COVID (before 2020-03-01), Lockdown (Mar–Jun 2020), Post-Lockdown (after Jul 2020). For inference we use Welch's t-test to compare means, and for forecasting we attempt SARIMA with fallbacks to ARIMA and a simple AR(1) OLS when models don't converge."

Slide 4 — Key Visuals
- Show `national_trend.png`: "This shows a spike in unemployment during the lockdown period."
- Show `rural_urban_trend.png`: "Rural and urban dynamics differ; urban is more volatile."
- Show `seasonality.png`: "There are recurring monthly patterns indicating informal-sector seasonality."

Slide 5 — COVID Impact & Stats
- Say: "We aggregated average unemployment across defined COVID periods. The Welch t-test rejects the null (p < 0.05), indicating a statistically significant increase during lockdown. The pivot table (`covid_impact_summary.csv`) quantifies absolute and percentage changes by area."

Slide 6 — Forecast Results
- Say: "We fit a stabilized SARIMA model where possible. If the SARIMA fit was unstable, we tried ARIMA, then AR(1) OLS. The final SARIMA forecast is shown; interpret with caution due to sample period and possible structural breaks from the pandemic."

Slide 7 — Policy Recommendations & Limitations
- Say: "Recommend counter-cyclical urban employment programs, strengthen rural safety nets, and invest in remote-work skills. Limitations: data representativeness (informal sector), sample length, and model assumptions. Next steps: add covariates and state-level hierarchical modeling."

Slides checklist
- Run `python Unemployment_analysis.py` to regenerate figures.
- Open `policy_insights.txt` to get the 1–2 minute summary.
