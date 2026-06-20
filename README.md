# World Cup Match Forecasting Engine

[![World Cup Forecasting CI](https://github.com/jennasilvera/world-cup-forecasting-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/jennasilvera/world-cup-forecasting-engine/actions/workflows/ci.yml)

A quant-style probabilistic football forecasting system for FIFA World Cup matches.

This project estimates pre-match win/draw/loss probabilities using historical international match results, custom Elo ratings, model-ready feature engineering, chronological backtesting, and recruiter-facing model reports.

The project is intentionally framed as a probabilistic forecasting engine, not a tool that claims to perfectly predict football matches.

## Why This Project Exists

Football outcomes are noisy, low-scoring, and highly uncertain. A serious forecasting system should therefore focus on:

- Calibrated probabilities
- Transparent assumptions
- Chronological backtesting
- Baseline comparisons
- Leakage prevention
- Uncertainty-aware reporting
- Reproducible data pipelines

This project is designed to resemble how a quantitative research, sports analytics, trading technology, or ML engineering team might build and evaluate a signal model.

## Current MVP

The current version includes:

- Historical results ingestion
- Data validation
- Custom Elo model
- Match-importance weighting
- Margin-of-victory Elo adjustment
- Neutral-site handling
- Pre-match feature table generation
- Logistic regression baseline
- Chronological train/test backtest
- Accuracy, log loss, and multiclass Brier score metrics
- Markdown backtest report generation
- Poisson expected-goals model
- Scoreline probability forecasting
- Analyst-style match prediction reports
- Unit tests and linting

## Repository Structure

    world-cup-forecasting-engine/
      data/
        sample/                 # Tiny committed dataset for reproducible demo runs
        raw/                    # Local raw data, ignored by Git
        processed/              # Local processed outputs, ignored by Git
      reports/                  # Markdown model/backtest reports
      outputs/                  # Local model outputs, ignored by Git
      src/
        wc_forecast/
          data/                 # Ingestion and validation
          features/             # Pre-match feature engineering
          models/               # Elo and ML models
          reporting/            # Backtest/model report generation
          simulation/           # Planned tournament simulation logic
          cli.py                # Project command-line interface
      tests/                    # Unit tests

## Modeling Approach

The project currently uses a layered modeling workflow.

### 1. Historical match ingestion

- Loads match results from CSV
- Validates required fields
- Cleans dates, scores, teams, tournament names, and neutral-site flags
- Adds match outcome labels

### 2. Custom Elo model

- Initializes unseen teams at a default rating
- Produces pre-match expected scores
- Adjusts for neutral vs non-neutral matches
- Applies higher match weight to FIFA World Cup matches
- Applies margin-of-victory scaling

### 3. Pre-match feature table

- Captures Elo ratings before each match update
- Builds relative team-strength features
- Separates pre-match features from final-score result columns

### 4. Logistic regression baseline

- Trains on historical pre-match features
- Predicts three-class outcome probabilities:
  - home/team A win
  - draw
  - away/team B win

### 5. Chronological backtest

- Splits matches by time rather than randomly
- Evaluates the model on later matches
- Reports accuracy, log loss, and multiclass Brier score

### 6. Poisson expected-goals model

- Estimates team attack strength from historical goals scored
- Estimates team defensive weakness from historical goals conceded
- Produces expected goals for both teams
- Converts expected goals into scoreline probabilities
- Derives home win, draw, and away win probabilities from the score matrix

### 7. Match prediction report

- Combines logistic-regression probabilities with Poisson expected-goals output
- Shows most likely scoreline
- Compares disagreement between model layers
- Adds caveats about sample size, limitations, and future features

## Leakage Prevention

The feature table is built chronologically.

For each match:

1. The model records the teams' pre-match Elo ratings.
2. The feature row is created.
3. Only after that does the Elo model update using the final result.

This prevents final scores from leaking into pre-match features.

Final scores and outcomes are retained only as result/target columns for supervised learning and evaluation.

## Quickstart

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .

Run the full MVP pipeline:

    python -m wc_forecast ingest-results data/sample/historical_results_sample.csv
    python -m wc_forecast build-elo
    python -m wc_forecast build-features
    python -m wc_forecast backtest-logistic
    python -m wc_forecast report-backtest
    python -m wc_forecast predict-poisson Argentina France
    python -m wc_forecast report-match Argentina France

Run tests and linting:

    pytest
    ruff check .

## CLI Commands

    python -m wc_forecast health
    python -m wc_forecast ingest-results data/sample/historical_results_sample.csv
    python -m wc_forecast build-elo
    python -m wc_forecast build-features
    python -m wc_forecast backtest-logistic
    python -m wc_forecast report-backtest
    python -m wc_forecast predict-poisson Argentina France
    python -m wc_forecast report-match Argentina France

## Example Outputs

The pipeline writes local model artifacts such as:

    data/processed/results.csv
    data/processed/features.csv
    outputs/elo_ratings.csv
    outputs/elo_history.csv
    outputs/logistic_backtest_predictions.csv
    outputs/logistic_backtest_metrics.csv
    reports/logistic_backtest_report.md

The backtest report includes:

- Model purpose
- Metric summary
- Recent match-level predictions
- Probability estimates
- Expected-goals estimates
- Most likely scoreline
- Model layer comparison
- Model inputs
- Leakage controls
- Current limitations
- Planned improvements

## Current Limitations

This project is currently an MVP. The committed sample dataset is intentionally small so reviewers can run the project immediately.

The current results should not be interpreted as evidence of real-world predictive power.

Current limitations:

- Small demo dataset
- Limited feature set
- No injury or lineup data yet
- No market-implied odds yet
- Poisson model is still a transparent baseline
- No calibrated ensemble yet
- No tournament simulation yet

## Planned Upgrades

Next planned additions:

- Larger public historical results dataset
- Rolling form features
- Strength-of-schedule features
- FIFA ranking features
- Calibrated ensemble model
- Reliability/calibration plots
- Monte Carlo tournament simulator
- Group-stage and knockout advancement probabilities
- FastAPI prediction service
- Streamlit or React dashboard
- Full model card
- CI with GitHub Actions

## Data Use Statement

This project uses public, reproducible, non-proprietary data. The committed sample dataset is intentionally small and included only for demonstration purposes.

Future data integrations should avoid unclear scraping practices and should document source, license, access date, and transformation assumptions.

## Professional Framing

This project does not claim to perfectly predict football matches.

It demonstrates how to build a reproducible probabilistic forecasting pipeline with:

- Pre-match feature generation
- Quant-style rating systems
- Chronological model evaluation
- Proper leakage controls
- Transparent reporting
- Clean Python engineering practices
