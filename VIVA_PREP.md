# Viva / Presentation Preparation — Professional Unemployment Analysis

This document gives a short script, slide outline, and likely viva questions with compact answers to help you present and defend this analysis.

Presentation (7–10 minutes)
- Slide 1 (30s): Title, your name, objective — analyze unemployment (2019–2020), COVID impact, seasonal patterns, forecasting.
- Slide 2 (45s): Data sources — `Unemployment in India.xls` and `Unemployment_Rate_upto_11_2020.xls`. Mention cleaning steps (header normalization, date parsing, numeric coercion).
- Slide 3 (60s): Methodology — aggregation, feature engineering (month, covid_period), state and area breakdowns, Welch t-test for lockdown impact, SARIMA forecasting with ARIMA/AR(1) fallbacks.
- Slide 4 (60s): Key visuals — national trend, rural vs urban, seasonality (show `national_trend.png` and `rural_urban_trend.png`).
- Slide 5 (60s): Statistical results — t-test summary and COVID impact pivot (`covid_impact_summary.csv`).
- Slide 6 (60s): Forecasts — show `sarima_forecast.png`, explain model and fallbacks, and discuss forecast uncertainty.
- Slide 7 (45s): Policy takeaways / limitations / next steps.

Viva: Likely Questions & Suggested Short Answers
- Q: What data cleaning steps did you perform?
  - A: Normalized headers, parsed date with day-first, coerced numeric columns, filled/filtered invalid records, standardized area/region strings.

- Q: How did you define COVID periods?
  - A: Pre-COVID: before 2020-03-01. Lockdown: 2020-03-01 to 2020-06-30. Post-Lockdown: after 2020-07-01. These windows align with major lockdown dates and allow aggregation for comparison.

- Q: Why SARIMA and what are its limitations here?
  - A: SARIMA captures trend and seasonality. Its limitations include sensitivity to short samples and convergence issues; hence ARIMA and AR(1) OLS fallbacks were implemented for robustness.

- Q: How do you ensure forecasts are reliable?
  - A: Use diagnostics (ADF p-value, residual checks, confidence intervals), check model convergence, prefer quarterly aggregation when monthly series are sparse, and compare fallbacks.

- Q: Why use Welch's t-test?
  - A: Welch's t-test does not assume equal variances between groups (pre-COVID vs lockdown), making it robust for heteroskedastic data.

- Q: How would you improve the analysis?
  - A: Incorporate more recent data, control for labour-force participation shifts, add state-level covariates (industry composition, urbanization), and try hierarchical time-series models.

Quick defense talking points
- Emphasize data-driven, reproducible pipeline; fallbacks ensure results are robust when data quality varies.
- Acknowledge limitations (sampling, informal sector measurement issues) and briefly outline mitigation strategies.

Files to have ready during viva
- `national_trend.png`, `rural_urban_trend.png`, `seasonality.png`, `sarima_forecast.png`, `policy_insights.txt`, and `combined_cleaned_unemployment.csv`.

Short checklist before presenting
- Re-run `python Unemployment_analysis.py` locally to produce fresh plots.
- Open `policy_insights.txt` and practice two-minute summary.

Good luck — ask if you want a concise 5-slide PDF export or speaker notes for each slide.
