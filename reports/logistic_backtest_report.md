# Backtest Report: Logistic Regression Baseline

## Purpose

This report summarizes a chronological backtest for the current World Cup Match
Forecasting Engine model.

The model is evaluated as a probabilistic forecasting system, not as a perfect
match-result predictor. The goal is to estimate pre-match win/draw/loss
probabilities and evaluate whether those probabilities are reasonable,
transparent, and testable.

## Backtest Summary

The latest backtest produced accuracy of 0.5902, log loss of 0.8903, multiclass Brier score of 0.5240.

The latest run may use either the small repository sample dataset or a larger
real international-results dataset, depending on the local processed data file.
Metric values should be interpreted in the context of the data source, feature
set, train/test split, and known model limitations. The report demonstrates that
the pipeline can train, predict, score, and generate auditable outputs
end-to-end.

## Metrics

| Metric | Value |
|---|---:|
| train_rows | 34603 |
| test_rows | 14830 |
| accuracy | 0.5902 |
| log_loss | 0.8903 |
| multiclass_brier_score | 0.5240 |

## Recent Match-Level Predictions

| Date | Match | Score | Actual Outcome | Predicted Outcome | Probabilities |
|---|---|---:|---|---|---|
| 2026-06-16 | France vs Senegal | 3-1 | home_win | home_win | H 61.2% / D 23.5% / A 15.2% |
| 2026-06-16 | Iraq vs Norway | 1-4 | away_win | away_win | H 24.4% / D 24.5% / A 51.1% |
| 2026-06-17 | England vs Croatia | 4-2 | home_win | home_win | H 46.6% / D 28.0% / A 25.4% |
| 2026-06-17 | Ghana vs Panama | 1-0 | home_win | away_win | H 20.6% / D 21.8% / A 57.6% |
| 2026-06-17 | Portugal vs DR Congo | 1-1 | draw | home_win | H 76.4% / D 15.7% / A 7.8% |
| 2026-06-17 | Uzbekistan vs Colombia | 1-3 | away_win | away_win | H 14.9% / D 23.4% / A 61.7% |
| 2026-06-18 | Canada vs Qatar | 6-0 | home_win | home_win | H 76.4% / D 16.0% / A 7.5% |
| 2026-06-18 | Czech Republic vs South Africa | 1-1 | draw | home_win | H 53.5% / D 21.7% / A 24.8% |
| 2026-06-18 | Mexico vs South Korea | 1-0 | home_win | home_win | H 53.9% / D 27.7% / A 18.3% |
| 2026-06-18 | Switzerland vs Bosnia and Herzegovina | 4-1 | home_win | home_win | H 69.9% / D 17.7% / A 12.4% |

## Model Inputs

The current baseline uses pre-match features generated from custom Elo ratings
and match context fields:

- Home-team pre-match Elo rating
- Away-team pre-match Elo rating
- Elo rating difference
- Absolute Elo rating difference
- Elo expected home/away score
- Neutral-site flag
- World Cup match flag
- Tournament importance weight

## Leakage Controls

The feature table is built chronologically. Elo ratings are captured before each
match update, and final scores are stored separately as result/target columns.
This prevents final scores from leaking into pre-match features.

## Current Limitations

- The baseline can run on either sample data or a larger real international
  results dataset.
- The current backtest uses a simple chronological split and should be upgraded
  with explicit train/evaluation cutoff dates.
- Current features are mostly team-strength and match-context features.
- Player availability, injuries, travel, rest, and market odds are planned
  future upgrades.
- Logistic regression is used as an interpretable baseline, not the final model.

## Next Improvements

- Add larger public historical results data.
- Add rolling form features.
- Add Poisson expected-goals model.
- Add calibrated ensemble forecasts.
- Add Monte Carlo tournament simulation.
- Add calibration plots and model cards.
