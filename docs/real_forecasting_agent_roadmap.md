# Real Forecasting Agent Roadmap

## Purpose

This document defines the long-term direction for the World Cup Match Forecasting Engine.

The current repository is a reproducible forecasting scaffold. It includes data validation, Elo ratings, feature engineering, logistic regression, Poisson expected-goals modeling, ensemble probabilities, Monte Carlo group-stage simulation, reports, tests, CI, and a one-command demo.

The long-term goal is to evolve this scaffold into a real pre-match forecasting agent that can ingest legally available data, generate calibrated probabilities, compare those probabilities against market-implied odds, track predictions over time, and evaluate whether the model has evidence of edge.

## Important Framing

This project should not claim that it can perfectly predict football matches.

A serious forecasting agent should estimate probabilities, not certainties.

For betting-oriented evaluation, the key question is not:

Can the model pick the winner?

The key question is:

Can the model produce calibrated pre-match probabilities that outperform market-implied probabilities after removing bookmaker margin, over a large out-of-sample sample?

## Current System

The current system includes:

- historical result ingestion
- custom Elo model
- leakage-aware pre-match feature table
- logistic regression baseline
- chronological backtest
- Poisson expected-goals model
- scoreline probabilities
- ensemble forecast layer
- forecast entropy
- model-disagreement signal
- Monte Carlo group-stage simulation
- Markdown reports
- model card and assumptions documentation
- CI and one-command demo workflow

This is a strong base, but it is not yet a validated betting model.

## Target System

The target system should become a pre-match forecasting agent with these capabilities:

1. Pull or load legal public/licensed data.
2. Build timestamp-safe pre-match feature rows.
3. Generate match-level probabilities.
4. Compare model probabilities against market-implied probabilities.
5. Calculate expected value.
6. Store every prediction in a ledger.
7. Track calibration, closing-line value, and realized performance.
8. Simulate tournament outcomes from calibrated probabilities.
9. Produce reports explaining forecasts, edge/no-edge decisions, uncertainty, and caveats.

## Forecasting Agent Architecture

### 1. Data Ingestion Layer

Future data sources should include:

- historical international match results
- FIFA rankings and ranking points
- team Elo ratings
- tournament schedules, venues, and groups
- player availability data
- expected lineups
- injuries and suspensions
- squad composition
- player minutes and fatigue
- travel distance
- rest days
- weather forecasts
- venue altitude
- pitch or stadium context where legally available
- betting odds
- odds movement
- market closing odds

### 2. Data Source Registry

Every data source should be documented with:

- source name
- source type
- license or usage notes
- update frequency
- access method
- whether it is available before kickoff
- whether it is historical, live, manual, or forecasted
- leakage risk
- reliability rating

### 3. Feature Store

Every feature should include:

- feature name
- formula
- data source
- timestamp
- match id
- team id
- whether it is pre-match safe
- missingness flag
- fallback behavior

### 4. Prediction Ledger

Every model prediction should be saved with:

- match id
- team A
- team B
- kickoff time
- prediction timestamp
- model version
- feature version
- model probabilities
- market probabilities
- expected value
- recommended action
- closing odds
- final result
- realized profit/loss if a strategy is being tested

The prediction ledger is essential. Without it, the project cannot prove whether the model had real forecasting edge.

### 5. Modeling Layer

Future models should include:

- naive baseline
- FIFA ranking baseline
- Elo baseline
- Poisson expected-goals model
- Dixon-Coles style score model
- gradient boosting classifier
- calibrated ensemble model
- market-aware model
- Bayesian or uncertainty-aware layer

The model should be allowed to return:

- home/team A edge
- draw edge
- away/team B edge
- no edge

A serious betting model should often say no edge.

### 6. Market Comparison Layer

The system should convert odds into implied probabilities, remove bookmaker margin, and compare the model against the market.

It should calculate:

- raw implied probability
- de-vigged implied probability
- model probability
- model edge
- expected value
- closing-line value
- realized return
- calibration by odds bucket

### 7. Simulation Layer

The tournament simulator should eventually include:

- group-stage simulation
- full FIFA tie-breaker logic
- knockout bracket simulation
- penalty shootout assumptions
- path-dependent probability updates
- probability of reaching each round
- championship probability

### 8. Reporting Layer

Reports should include:

- match forecast report
- backtest report
- calibration report
- market edge report
- prediction ledger report
- group-stage simulation report
- tournament simulation report
- model card
- assumptions and limitations

## Micro-Context Features

The model may eventually consider very granular contextual factors, such as:

- hotel quality
- bed quality proxy
- travel comfort
- time zone change
- training facility quality
- weather
- venue altitude
- humidity
- pitch condition
- travel distance
- crowd context
- local climate familiarity

However, these features should only be used if they pass four tests:

1. The data is legal and ethical to obtain.
2. The data is available before kickoff.
3. Historical equivalents exist for backtesting.
4. The feature adds signal out-of-sample.

If a feature cannot be backtested historically, it should be treated as a manual qualitative adjustment or low-confidence scenario input, not as a core model feature.

## Betting-Grade Evaluation Requirements

Before the model can be treated as betting-relevant, it must show:

- strong out-of-sample calibration
- log loss improvement over baselines
- Brier score improvement over baselines
- performance versus de-vigged market probabilities
- positive expected value over a large sample
- closing-line value evidence
- stable performance by time period
- stable performance by league/tournament context
- clear drawdown analysis
- no evidence of leakage

## What Would Make This Credible

The most credible next upgrades are:

1. Add historical betting odds ingestion.
2. Add implied probability and de-vig calculations.
3. Add a prediction ledger.
4. Add calibration evaluation.
5. Add rolling/expanding backtests.
6. Add richer historical international match data.
7. Add weather and venue features.
8. Add player availability manual inputs.
9. Add market edge reports.
10. Add full tournament simulation.

## What Not To Claim

Do not claim:

- the model guarantees profitable betting
- the model can predict every match
- the current sample dataset proves accuracy
- the current demo outputs are real betting signals

Professional framing:

This project estimates calibrated pre-match probabilities using historical team strength, match context, market-implied expectations, expected-goals modeling, and tournament simulation. Betting-oriented use requires larger data, strict timestamping, market comparison, calibration testing, and a prediction ledger.
