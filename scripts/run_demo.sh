#!/usr/bin/env bash
set -euo pipefail

RESULTS_SOURCE="${1:-}"
PROCESSED_RESULTS="data/processed/results.csv"

echo "== World Cup Forecasting Engine Demo =="

echo
echo "1. Health check"
python -m wc_forecast health

echo
echo "2. Prepare international results data"

if [[ -n "$RESULTS_SOURCE" ]]; then
  if [[ ! -f "$RESULTS_SOURCE" ]]; then
    echo "ERROR: Results source not found: $RESULTS_SOURCE"
    echo "Usage: ./scripts/run_demo.sh path/to/results.csv"
    exit 1
  fi

  echo "Ingesting raw results source: $RESULTS_SOURCE"
  python -m wc_forecast ingest-real-results "$RESULTS_SOURCE"

elif [[ -f "$PROCESSED_RESULTS" ]]; then
  echo "No raw source supplied. Using existing processed results:"
  echo "$PROCESSED_RESULTS"

else
  echo "ERROR: No raw results source supplied and no processed results found."
  echo
  echo "Run one of:"
  echo "  ./scripts/run_demo.sh path/to/results.csv"
  echo
  echo "or first create:"
  echo "  data/processed/results.csv"
  exit 1
fi

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
