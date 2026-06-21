#!/usr/bin/env bash
set -euo pipefail

echo "== World Cup Forecasting Engine Demo =="

echo
echo "1. Health check"
python -m wc_forecast health

echo
echo "2. Ingest real international results"
python -m wc_forecast ingest-real-results

echo
echo "3. Build model features"
python -m wc_forecast build-features

echo
echo "4. Run tuned rolling backtest"
python -m wc_forecast rolling-backtest \
  --sample-weight-half-life-days 2190 \
  --model-types logistic \
  --evaluation-window-days 365 \
  --output outputs/rolling_backtest_metrics.csv

echo
echo "5. Run feature ablation"
python -m wc_forecast ablate-features \
  --sample-weight-half-life-days 2190 \
  --logistic-c 4.0 \
  --evaluation-window-days 365 \
  --output outputs/feature_ablation_results.csv

echo
echo "6. Generate sample World Cup fixture forecasts"
python -m wc_forecast forecast-fixtures \
  data/sample/world_cup_2026_fixtures_sample.csv \
  --train-cutoff-date 2026-01-01 \
  --rating-cutoff-date 2026-06-19 \
  --output outputs/world_cup_2026_forecasts.csv

echo
echo "Demo complete. Key outputs:"
echo "- outputs/rolling_backtest_metrics.csv"
echo "- outputs/feature_ablation_results.csv"
echo "- outputs/world_cup_2026_forecasts.csv"
