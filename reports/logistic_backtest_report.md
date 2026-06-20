# Backtest Report: Logistic Regression Baseline

## Purpose

This report summarizes a chronological backtest for the current World Cup Match
Forecasting Engine model.

The model is evaluated as a probabilistic forecasting system, not as a perfect
match-result predictor. The goal is to estimate pre-match win/draw/loss
probabilities and evaluate whether those probabilities are reasonable,
transparent, and testable.

## Backtest Summary

The latest backtest produced accuracy of 0.3333, log loss of 0.9440, multiclass Brier score of 0.6231.

Because this project is currently using a small reproducible sample dataset, the
absolute metric values should not be interpreted as evidence of real predictive
power. At this stage, the report demonstrates that the pipeline can train,
predict, score, and generate auditable outputs end-to-end.

## Metrics

| Metric | Value |
|---|---:|
| train_rows | 7 |
| test_rows | 3 |
| accuracy | 0.3333 |
| log_loss | 0.9440 |
| multiclass_brier_score | 0.6231 |

## Recent Match-Level Predictions

| Date | Match | Score | Actual Outcome | Predicted Outcome | Probabilities |
|---|---|---:|---|---|---|
| 2022-11-25 | Netherlands vs Ecuador | 1-1 | draw | home_win | H 54.9% / D 34.9% / A 10.1% |
| 2022-12-03 | Netherlands vs United States | 3-1 | home_win | home_win | H 48.8% / D 32.4% / A 18.8% |
| 2022-12-18 | Argentina vs France | 3-3 | draw | home_win | H 53.8% / D 34.6% / A 11.6% |

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

- The sample dataset is intentionally small and exists only to make the repo
  runnable for reviewers.
- Real evaluation requires a larger historical international-match dataset.
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
