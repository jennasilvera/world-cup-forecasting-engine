# Match Prediction Report: Argentina vs France

## Forecast Summary

This report combines the current logistic-regression baseline with the Poisson
expected-goals model.

The output should be interpreted as a probabilistic model report, not as a claim
that the match result can be predicted with certainty.

## Logistic Regression Baseline

| Outcome | Probability |
|---|---:|
| Argentina win | 5.5% |
| Draw | 92.0% |
| France win | 2.6% |

Model confidence: **High**

## Poisson Expected-Goals Forecast

| Field | Value |
|---|---:|
| Argentina expected goals | 3.333 |
| France expected goals | 4.167 |
| Argentina win probability | 31.2% |
| Draw probability | 14.3% |
| France win probability | 54.5% |
| Most likely scoreline | 3-4 |
| Scoreline probability | 4.3% |

## Model Layer Comparison

| Signal | Value |
|---|---:|
| Logistic Argentina win probability | 5.5% |
| Poisson Argentina win probability | 31.2% |
| Absolute home-win disagreement | 25.7% |

## Interpretation

The logistic model uses engineered pre-match features such as Elo rating
difference, expected Elo score, neutral-site status, and tournament importance.

The Poisson model estimates expected goals from historical team attack and
defense profiles, then converts expected goals into scoreline probabilities.

When the two layers disagree, the forecast should be treated with more caution.
In a future ensemble model, this disagreement can become an explicit uncertainty
or risk feature.

## Caveats

- The current committed dataset is intentionally small for reproducible demo use.
- The current probabilities are pipeline outputs, not validated betting signals.
- Real predictive evaluation requires a larger historical international dataset.
- Injury, lineup, rest, travel, market odds, and squad-strength features are not
  included yet.
- The Poisson model is a transparent baseline and does not yet include advanced
  hierarchical or time-decay effects.
