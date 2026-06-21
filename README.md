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
- Lightweight ensemble forecast layer
- Forecast entropy and model-disagreement signals
- Monte Carlo group-stage simulation
- Group advancement probability outputs
- Market-implied probability and expected value evaluation
- Batch market odds slate evaluation and ranked edge output
- Strategy policy filtering for actionable edges
- Fractional Kelly stake sizing with exposure caps
- Batch prediction ledger logging for candidate edges
- Batch prediction settlement from final results and closing odds
- Timestamped prediction ledger for forecast auditability
- Ledger settlement with final score, closing odds, and realized return
- Prediction ledger performance report
- Markdown group-stage simulation report
- Unit tests and linting

## One-Command Demo

Run the full reproducible demo pipeline:

    make demo

This executes:

- historical result ingestion
- Elo rating generation
- pre-match feature generation
- logistic-regression backtest
- backtest report generation
- Poisson expected-goals prediction
- match prediction report generation
- market edge and expected value evaluation
- batch market odds slate evaluation
- strategy policy filtering
- stake sizing
- batch prediction ledger logging
- batch prediction settlement
- timestamped prediction ledger logging
- prediction settlement and realized return calculation
- prediction ledger performance reporting
- Monte Carlo group-stage simulation
- group-stage simulation report generation

You can also run quality checks with:

    make check

## Documentation and Reports

Key generated and maintained project documents:

- [Backtest Report](reports/logistic_backtest_report.md)
- [Match Prediction Report](reports/match_prediction_report.md)
- [Group-Stage Simulation Report](reports/group_stage_simulation_report.md)
- Prediction Ledger Performance Report generated at `outputs/prediction_ledger_report.md`
- [Model Card](reports/model_card.md)
- [Real Forecasting Agent Roadmap](docs/real_forecasting_agent_roadmap.md)
- [Data Source Registry](docs/data_source_registry.md)
- [Betting Evaluation Framework](docs/betting_evaluation_framework.md)
- [Assumptions and Limitations](reports/assumptions.md)

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

### 8. Ensemble forecast layer

- Blends logistic-regression probabilities with Poisson probabilities
- Produces a final home/draw/away probability forecast
- Reports predicted outcome and confidence label
- Calculates normalized probability entropy
- Calculates maximum model disagreement across outcome classes
- Treats model-layer disagreement as an uncertainty signal

### 9. Monte Carlo group-stage simulation

- Loads group-stage fixture definitions from CSV
- Uses the Poisson expected-goals model to sample match scorelines
- Simulates group standings repeatedly
- Applies points, goal difference, and goals-for ranking logic
- Estimates each team's probability of advancing from the group
- Outputs average points, average goal difference, and average goals for

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
    python -m wc_forecast evaluate-market Argentina France --home-odds 2.20 --draw-odds 3.40 --away-odds 3.50
    python -m wc_forecast batch-evaluate-market data/sample/market_odds_sample.csv
    python -m wc_forecast apply-strategy-policy outputs/batch_market_edges.csv
    python -m wc_forecast size-stakes outputs/strategy_policy_edges.csv
    python -m wc_forecast log-batch-predictions outputs/stake_sizing_edges.csv
    python -m wc_forecast settle-batch-predictions data/sample/settlement_results_sample.csv
    python -m wc_forecast log-prediction Argentina France --home-odds 2.20 --draw-odds 3.40 --away-odds 3.50
    PREDICTION_ID=$(tail -n 1 outputs/prediction_ledger.csv
outputs/prediction_ledger_report.md | cut -d',' -f1)
    python -m wc_forecast settle-prediction "$PREDICTION_ID" --final-home-score 1 --final-away-score 1 --closing-home-odds 2.10 --closing-draw-odds 3.25 --closing-away-odds 3.60
    python -m wc_forecast report-ledger
    python -m wc_forecast simulate-group-stage --n-simulations 1000
    python -m wc_forecast report-group-stage

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
    python -m wc_forecast simulate-group-stage --n-simulations 1000

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
- Ensemble forecast
- Probability entropy
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
- Current ensemble is a transparent weighted-average baseline, not yet calibrated
- Group-stage simulation exists; knockout simulation is not implemented yet

## Planned Upgrades

Next planned additions:

- Larger public historical results dataset
- Rolling form features
- Strength-of-schedule features
- FIFA ranking features
- Calibrated ensemble model with validation-based weights
- Reliability/calibration plots
- Knockout-stage Monte Carlo simulator
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

## System Design and Forecasting Agent Architecture

### Core Design Principles

#### Leakage prevention

The feature pipeline is chronological. Pre-match features are calculated before the match result is known. Elo ratings are updated after each match, but the model only receives the pre-match rating state for prediction.

#### Probabilistic forecasting

The engine does not only output a winner. It produces probability distributions over:

- home win
- draw
- away win
- scorelines
- group-stage advancement probabilities

This allows the system to support calibration, risk analysis, and expected value calculations.

#### Separation of model and decision layers

The model forecast is not treated as an automatic action.

The system separates:

    forecast probability
    → market-implied probability
    → edge detection
    → strategy policy filtering
    → stake sizing
    → ledger logging
    → settlement
    → performance review

This makes the project closer to a real research or trading process, where a signal must pass risk and quality gates before becoming an action.

### Main Components

#### Data ingestion

The ingestion layer validates historical match results before writing processed data. It checks schema quality, missing values, team validity, score values, and outcome construction.

Primary command:

    python -m wc_forecast ingest-results data/sample/historical_results_sample.csv

#### Elo rating engine

The Elo module generates team strength ratings using chronological match results. It supports tournament weighting, margin-of-victory adjustment, and neutral-site handling.

Primary command:

    python -m wc_forecast build-elo

#### Feature engineering

The feature builder creates pre-match model features from historical data and Elo state. It is designed to avoid future leakage.

Primary command:

    python -m wc_forecast build-features

#### Logistic model backtest

The logistic model provides a supervised learning baseline with chronological train/test splitting. It reports accuracy, log loss, and multiclass Brier score.

Primary command:

    python -m wc_forecast backtest-logistic

#### Poisson expected-goals model

The Poisson model estimates expected goals and scoreline probabilities. It supports match-level probability forecasts and scoreline analysis.

Primary command:

    python -m wc_forecast predict-poisson Argentina France

#### Ensemble forecast

The ensemble combines model outputs into a blended probability forecast. It tracks confidence, entropy, and model disagreement.

This gives the strategy layer more than just a single probability estimate. It also receives quality and uncertainty indicators.

#### Market odds and edge detection

The market layer converts decimal odds into implied probabilities, removes market overround, compares model probabilities against market fair probabilities, and calculates expected value.

Primary commands:

    python -m wc_forecast evaluate-market Argentina France --home-odds 2.20 --draw-odds 3.40 --away-odds 3.50
    python -m wc_forecast batch-evaluate-market data/sample/market_odds_sample.csv

#### Strategy policy layer

The strategy policy layer filters raw candidate edges using risk and quality gates:

- minimum edge
- minimum expected value
- maximum entropy
- maximum model disagreement
- maximum market overround
- allowed confidence levels

Primary command:

    python -m wc_forecast apply-strategy-policy outputs/batch_market_edges.csv

#### Stake sizing

The stake sizing layer applies fractional Kelly sizing with caps on single-bet exposure and total portfolio exposure.

Primary command:

    python -m wc_forecast size-stakes outputs/strategy_policy_edges.csv

#### Prediction ledger

The ledger records each forecast and decision for auditability. It stores model probabilities, market probabilities, edges, expected values, strategy decisions, stake sizing fields, final outcomes, closing odds, and realized returns.

Primary commands:

    python -m wc_forecast log-batch-predictions outputs/stake_sizing_edges.csv
    python -m wc_forecast settle-batch-predictions data/sample/settlement_results_sample.csv
    python -m wc_forecast report-ledger

#### Group-stage simulation

The simulation layer uses forecast probabilities and scoreline sampling to estimate group-stage standings and advancement probabilities.

Primary command:

    python -m wc_forecast simulate-group-stage --n-simulations 500

### One-Command Demo

The project includes a full demo pipeline:

    make demo

The demo runs ingestion, ratings, features, model backtesting, match forecasting, market edge evaluation, strategy policy filtering, stake sizing, ledger logging, settlement, reporting, and group-stage simulation.

The project also includes:

    make check

which runs linting and the full test suite.

### Outputs

Important generated outputs include:

    data/processed/results.csv
    data/processed/features.csv
    outputs/elo_ratings.csv
    outputs/logistic_backtest_predictions.csv
    outputs/logistic_backtest_metrics.csv
    outputs/poisson_prediction.csv
    outputs/match_prediction.csv
    outputs/market_edge.csv
    outputs/batch_market_edges.csv
    outputs/strategy_policy_edges.csv
    outputs/stake_sizing_edges.csv
    outputs/prediction_ledger.csv
    outputs/prediction_ledger_report.md
    outputs/group_stage_simulation.csv
    reports/logistic_backtest_report.md
    reports/match_prediction_report.md
    reports/group_stage_simulation_report.md

### Testing and CI

The repository includes automated tests for:

- data ingestion
- Elo ratings
- feature engineering
- model backtesting
- Poisson forecasting
- ensemble blending
- market odds de-vigging
- batch edge evaluation
- strategy policy filtering
- fractional Kelly stake sizing
- ledger logging
- settlement
- stake-weighted reporting
- group-stage simulation
- CLI import health

The CI workflow runs linting, tests, and the sample forecasting pipeline.

### Current Limitations

The committed sample dataset is intentionally small and synthetic/demo-oriented. The current outputs should be interpreted as workflow validation, not evidence of real predictive accuracy or betting profitability.

A real production-grade version would require:

- larger historical results datasets
- team/player availability data
- fixture metadata
- travel/rest/contextual features
- market odds snapshots with timestamps
- out-of-sample prediction history
- calibration tracking
- robust feature versioning
- model version registry
- automated data refresh
- monitoring and alerting

### Future Extensions

Potential future extensions include:

- richer feature store
- player-level availability model
- injury and suspension ingestion
- market movement tracking
- closing-line value dashboards
- calibration plots
- model registry
- automated scheduled predictions
- API service for forecasts
- dashboard for match and portfolio views
- database-backed prediction ledger

### Summary

This project demonstrates more than a basic machine learning model. It implements an end-to-end forecasting and decision workflow with modeling, risk filters, stake sizing, audit logging, settlement, and performance reporting.

The main value of the project is the architecture: it shows how a prediction system can be structured like a research-grade forecasting agent rather than a one-off notebook.


## Reproducible Demo

Run the full modeling workflow with an existing processed results file:

    ./scripts/run_demo.sh

Or provide a raw international results CSV path:

    ./scripts/run_demo.sh path/to/results.csv

The demo performs:

1. Project health check
2. Results data preparation
3. Feature engineering
4. Tuned rolling backtest
5. Feature ablation validation
6. Sample World Cup fixture forecasting

Key outputs:

- `outputs/rolling_backtest_metrics.csv`
- `outputs/feature_ablation_results.csv`
- `outputs/world_cup_2026_forecasts.csv`

## Model Selection

The default forecasting model is selected using rolling-origin validation across historical cutoff dates, feature ablation, and logistic hyperparameter tuning.

Current default:

- Model: logistic regression
- Recency half-life: 2,190 days
- Logistic regularization C: 4.0
- Feature set: Elo, match context, rolling form, and attack/defense form

See [`docs/model_selection.md`](docs/model_selection.md) for validation results and rationale.

