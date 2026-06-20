# Prediction Ledger Performance Report

## Purpose

This report summarizes the prediction ledger after forecasts have been logged
and, where available, settled with final match results.

The ledger is designed to support auditability, calibration review, expected
value tracking, and future betting-style evaluation.

## Summary Metrics

| Metric | Value |
|---|---:|
| Total logged predictions | 4 |
| Settled predictions | 2 |
| Candidate edges logged | 4 |
| Settled candidate edges | 2 |
| Hit rate on settled candidate edges | 100.0% |
| Total flat-stake return | 4.800 |
| Flat-stake ROI | 240.0% |
| Average expected value | 0.806 |
| Average model edge | 24.7% |
| Average closing-line movement | 0.150 |

## Recent Settled Predictions

| Prediction ID | Match | Decision | Pick | Final | Return | EV | Edge |
|---|---|---|---|---|---:|---:|---:|
| c6d33a4d | Argentina vs France | candidate_edge | draw | draw | 2.400 | 0.806 | 24.7% |
| 48f2c02f | Argentina vs France | candidate_edge | draw | draw | 2.400 | 0.806 | 24.7% |

## Interpretation

The performance report treats rows with `decision = candidate_edge` as the
forecasting agent's candidate betting-style signals.

Rows with `decision = no_edge` are still useful for calibration and monitoring,
but they should not be counted as active betting decisions unless a strategy
explicitly says otherwise.

## Caveats

- The committed demo ledger is not evidence of real betting profitability.
- The current sample dataset is intentionally small.
- Real evaluation requires many timestamped, out-of-sample predictions.
- Closing-line value should be interpreted over large samples, not individual
  examples.
- This report is an audit and evaluation artifact, not betting advice.
