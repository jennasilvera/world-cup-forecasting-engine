# Model Selection and Validation Notes

This document explains how the World Cup Forecasting Engine selects its default forecasting model, feature set, and tuned parameters.

The goal is not to maximize a single lucky backtest. The goal is to select a model that is stable across historical time periods, produces calibrated probabilities, and avoids obvious leakage.

## Validation Philosophy

The project uses rolling-origin validation rather than random train/test splits.

For each cutoff date, the model trains only on matches before the cutoff and evaluates on matches in the following 365-day window. This simulates the actual forecasting problem: predicting future matches using only information available at the time.

The current rolling validation cutoffs are:

```text
2018-01-01
2019-01-01
2020-01-01
2021-01-01
2022-01-01
2023-01-01
2024-01-01
2025-01-01
2026-01-01
```

The main selection metric is **log loss**, because the project forecasts probabilities rather than only hard winners. Brier score is used as a secondary probability-quality metric. Accuracy is reported, but it is not the primary optimization target.

## Model Candidates

The current model candidates are:

| Model | Purpose |
|---|---|
| Logistic regression | Transparent calibrated baseline |
| Gradient boosting | Nonlinear tree-based challenger |
| Random forest | Ensemble tree-based challenger |

Rolling validation showed that logistic regression had the best overall probability quality.

| Model | Folds | Test Rows | Accuracy | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 9 | 8,134 | 0.5947 | 0.8809 | 0.5185 |
| Gradient boosting | 9 | 8,134 | 0.5894 | 0.8914 | 0.5245 |
| Random forest | 9 | 8,134 | 0.5876 | 0.8915 | 0.5244 |

Because logistic regression won on log loss, Brier score, and accuracy, it remains the default model.

## Feature Sets

The model uses leakage-safe features available before kickoff.

Current feature families:

| Feature Group | Description |
|---|---|
| Elo strength | Pre-match home and away Elo ratings, Elo difference, expected Elo score |
| Match context | Neutral site flag, World Cup flag, tournament importance |
| Rolling form | Recent points per match and goal difference per match |
| Attack/defense form | Recent goals for and goals against per match |

Feature ablation confirmed that the full feature set performs best by probability quality.

| Feature Set | Active Features | Folds | Accuracy | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| All features | 25 | 9 | 0.5940 | 0.8790 | 0.5175 |
| Elo + context + points form | 17 | 9 | 0.5949 | 0.8807 | 0.5185 |
| Elo + context | 9 | 9 | 0.5920 | 0.8843 | 0.5212 |
| Elo only | 6 | 9 | 0.5946 | 0.8852 | 0.5213 |

The full feature set slightly reduces accuracy versus the simpler form model, but improves log loss and Brier score. Since this project is probability-focused, the full feature set is preferred.

## Tuned Logistic Parameters

After adding attack/defense form features, logistic regression was retuned across:

```text
sample_weight_half_life_days: 365, 730, 1095, 1460, 2190, 2920
logistic_c: 0.25, 0.5, 1.0, 2.0, 4.0
```

The best rolling log-loss result was:

| Half-Life Days | Logistic C | Folds | Accuracy | Log Loss | Brier |
|---:|---:|---:|---:|---:|---:|
| 2190 | 4.0 | 9 | 0.5941 | 0.8790 | 0.5175 |

The selected default configuration is therefore:

```text
model_type = logistic
sample_weight_half_life_days = 2190
logistic_c = 4.0
feature_set = all_features
```

## Interpretation of Forecasts

The output probabilities should be interpreted as model-implied estimates, not guarantees.

For example, a team with a 55% win probability is the most likely winner, but the remaining 45% probability is still assigned to draw or opponent win outcomes. This is especially important in football, where draws and low-scoring variance make deterministic prediction difficult.

The model is designed for analytical forecasting, validation, and portfolio demonstration. It is not betting advice.

## Current Default Forecast Command

```bash
python -m wc_forecast forecast-fixtures \
  data/sample/world_cup_2026_fixtures_sample.csv \
  --train-cutoff-date 2026-01-01 \
  --rating-cutoff-date 2026-06-19 \
  --output outputs/world_cup_2026_forecasts.csv
```

The tuned defaults are already applied automatically:

```text
sample_weight_half_life_days = 2190
logistic_c = 4.0
```

## Why This Matters

This validation process demonstrates:

- Time-aware train/test separation
- Avoidance of future leakage
- Model comparison across multiple historical windows
- Hyperparameter tuning with rolling validation
- Feature ablation to justify model complexity
- Probability-oriented evaluation using log loss and Brier score

This makes the project more credible than a simple winner-prediction demo.
