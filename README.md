# World Cup Match Forecasting Engine

A quant-style probabilistic football forecasting system for FIFA World Cup matches.

This project estimates calibrated pre-match probabilities for football matches using historical international results, team-strength ratings, feature engineering, model backtesting, and tournament simulation.

The goal is not to claim perfect prediction. The goal is to build a transparent forecasting engine similar to how a quant research, sports analytics, or trading technology team might build, test, monitor, and explain a probabilistic signal model.

## MVP Scope

The first version will include:

- Historical match results ingestion
- Custom Elo model
- Basic pre-match feature table
- Logistic regression baseline
- Chronological backtest
- Prediction report output

## Planned Upgrades

- Poisson expected-goals model
- Calibrated ensemble model
- Tournament Monte Carlo simulator
- Model card and backtest report
- FastAPI service
- Dashboard
- CI with GitHub Actions

## Important Limitation

This project uses public and reproducible data only. It avoids claims of certainty and focuses on calibrated probabilities, uncertainty, and transparent assumptions.
