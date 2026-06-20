# Betting Evaluation Framework

## Purpose

This document defines how the forecasting engine should be evaluated if the long-term goal is betting-relevant decision support.

The current project is not a betting system. It is a forecasting scaffold. Betting-oriented use requires market comparison, calibration, prediction timestamping, and a large out-of-sample prediction ledger.

## Core Principle

A betting model does not need to predict every winner.

A betting model needs probabilities that are better than the market often enough to identify positive expected value opportunities.

## Required Concepts

### Implied Probability

For decimal odds:

    implied_probability = 1 / decimal_odds

Example:

    odds = 2.50
    implied_probability = 1 / 2.50 = 0.40

### Bookmaker Margin

For a three-outcome football market:

    raw_home_probability = 1 / home_odds
    raw_draw_probability = 1 / draw_odds
    raw_away_probability = 1 / away_odds

    margin = raw_home_probability + raw_draw_probability + raw_away_probability

If margin is greater than 1, the bookmaker has an overround.

### De-Vigged Probability

A simple normalization method:

    fair_home_probability = raw_home_probability / margin
    fair_draw_probability = raw_draw_probability / margin
    fair_away_probability = raw_away_probability / margin

This gives a market-implied fair probability estimate.

### Model Edge

    edge = model_probability - market_fair_probability

A positive edge suggests the model believes the outcome is more likely than the market implies.

### Expected Value

For decimal odds:

    expected_value = model_probability * decimal_odds - 1

Positive expected value means the model's probability estimate implies a favorable price.

## Betting Decision Rule

A serious forecasting agent should not always produce a bet.

Example rule:

- calculate model probability
- calculate fair market probability
- calculate expected value
- require minimum edge threshold
- require minimum confidence
- require acceptable model disagreement
- require acceptable feature completeness
- otherwise return no edge

Example output:

| Field | Value |
|---|---:|
| Model probability | 42.0% |
| Market fair probability | 36.5% |
| Edge | 5.5% |
| Expected value | 8.0% |
| Decision | Candidate edge |
| Confidence | Medium |
| Warning | Lineup data missing |

## Required Evaluation Metrics

### Forecasting Metrics

- accuracy
- log loss
- Brier score
- ranked probability score
- calibration curve
- expected calibration error
- prediction entropy
- performance by confidence bucket

### Market Comparison Metrics

- model log loss vs market log loss
- model Brier score vs market Brier score
- edge bucket performance
- expected value vs realized return
- closing-line value
- performance by bookmaker
- performance by odds range
- performance by favorite/draw/underdog

### Betting Strategy Metrics

- flat-stake ROI
- yield
- hit rate
- average odds
- maximum drawdown
- profit factor
- number of bets
- average edge
- realized profit/loss
- volatility of returns

## Prediction Ledger Requirements

Every prediction should be stored before the match starts.

Required columns:

- prediction_id
- model_version
- feature_version
- match_id
- prediction_timestamp
- kickoff_timestamp
- team_a
- team_b
- tournament
- venue
- model_prob_team_a_win
- model_prob_draw
- model_prob_team_b_win
- market_prob_team_a_win
- market_prob_draw
- market_prob_team_b_win
- odds_team_a_win
- odds_draw
- odds_team_b_win
- edge_team_a_win
- edge_draw
- edge_team_b_win
- expected_value_team_a_win
- expected_value_draw
- expected_value_team_b_win
- recommended_action
- confidence
- entropy
- max_model_disagreement
- feature_missingness_rate
- closing_odds_team_a_win
- closing_odds_draw
- closing_odds_team_b_win
- final_team_a_score
- final_team_b_score
- final_outcome
- realized_return

## Closing-Line Value

Closing-line value measures whether the model consistently beats the final market price.

A model can lose money over a small sample and still show promise if it consistently beats closing lines.

A model can win over a small sample and still be weak if it does not beat closing lines.

## Leakage Risks

Common leakage failures:

- using closing odds as features for early predictions
- using final lineups before they were actually announced
- using post-match xG
- using post-match injuries or cards
- using final standings for pre-match group-stage predictions
- using actual weather when the model would only have had forecast weather
- using manually entered information without timestamps

## Minimum Bar Before Real Betting Use

The model should not be considered betting-relevant until it has:

- hundreds or thousands of out-of-sample predictions
- no known leakage
- documented prediction timestamps
- comparison against de-vigged market probabilities
- calibration analysis
- closing-line value analysis
- positive expected value evidence
- drawdown analysis
- stable results across time periods
- reproducible data and model versions

## Professional Output Language

Use:

    The model identifies a potential positive expected value edge.

Do not use:

    This bet will win.

Use:

    The model probability is higher than the de-vigged market-implied probability.

Do not use:

    The sportsbook is wrong.

Use:

    No edge under current assumptions.

Do not use:

    Force a pick.
