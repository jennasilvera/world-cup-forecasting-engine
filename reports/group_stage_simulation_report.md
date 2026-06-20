# Group-Stage Simulation Report

## Purpose

This report summarizes a Monte Carlo group-stage simulation for the World Cup
Match Forecasting Engine.

The simulation repeatedly samples match scorelines from the Poisson
expected-goals model, builds group standings, applies ranking logic, and
estimates each team's probability of advancing from its group.

## Simulation Setup

| Field | Value |
|---|---:|
| Simulations per team/group estimate | 500 |
| Ranking logic | Points, goal difference, goals for, team name |
| Match scoring model | Poisson expected-goals baseline |
| Advancement rule | Top teams by simulated group rank |

## Advancement Summary

### Group A

| Team | Advance Probability | Avg Points | Avg GD | Avg GF |
|---|---:|---:|---:|---:|
| Spain | 99.6% | 7.06 | 9.32 | 9.96 |
| Brazil | 98.8% | 6.89 | 5.67 | 6.25 |
| Argentina | 1.2% | 1.23 | -7.80 | 3.77 |
| France | 0.4% | 1.84 | -7.20 | 4.49 |
### Group B

| Team | Advance Probability | Avg Points | Avg GD | Avg GF |
|---|---:|---:|---:|---:|
| Ecuador | 80.0% | 6.02 | 4.82 | 7.53 |
| England | 70.8% | 5.78 | 3.58 | 9.64 |
| Netherlands | 48.8% | 4.74 | 2.36 | 7.51 |
| Iran | 0.4% | 0.59 | -10.76 | 3.90 |

## Interpretation

Advance probability estimates represent the share of simulations in which a team
finished in an advancing position.

Average points, average goal difference, and average goals for help explain
whether a team is advancing through consistent simulated performance or through
thin margins.

## Caveats

- The current committed fixture file is a small reproducible demo sample.
- The current historical results dataset is intentionally small.
- These probabilities are pipeline outputs, not validated betting or trading
  signals.
- FIFA tie-breaker logic is simplified.
- Knockout-stage simulation is not implemented yet.
- Future versions should use a larger historical dataset, calibrated model
  weights, rest/travel context, market odds, squad strength, and player
  availability inputs.
