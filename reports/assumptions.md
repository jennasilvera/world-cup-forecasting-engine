# Assumptions and Limitations

## Core Assumptions

This project assumes that historical team performance contains useful signal for future match probabilities.

The current MVP uses simplified assumptions:

- Team strength can be approximated with Elo ratings.
- Goals can be modeled with a Poisson process.
- Historical attack and defense rates are useful proxies for expected goals.
- Weighted averaging can provide a transparent first ensemble baseline.
- Group-stage advancement probabilities can be estimated through repeated simulation.

## Pre-Match Information Assumption

The model should only use information available before kickoff.

Allowed pre-match information may include:

- Historical results before the match
- Pre-match Elo ratings
- FIFA rankings available before the match
- Published fixtures
- Known venue and neutral-site status
- Rest days known before the match
- Market odds available before kickoff
- Manually entered injuries or suspensions available before kickoff

Disallowed information includes:

- Final score
- Post-match expected goals
- Post-match possession
- Post-match shots
- Post-match cards
- Post-match standings
- Any statistic only known after the match has started or ended

## Current Sample Dataset Assumption

The committed sample dataset exists only for reproducibility.

It is not large enough to support real performance claims.

## Simulation Assumptions

The current group-stage simulator assumes:

- Scorelines can be sampled from the Poisson model.
- Group ranking uses points, goal difference, goals for, then team name.
- The top teams by simulated group rank advance.
- Tie-breakers are simplified and do not fully match FIFA rules yet.

## Reporting Assumptions

Markdown reports are intended to make outputs reviewable by recruiters, engineers, analysts, and interviewers.

Reports should include caveats whenever the dataset is small, the model is uncalibrated, or the output is only a demonstration artifact.

## Future Assumptions to Revisit

Future versions should revisit:

- Whether Poisson assumptions fit international football data
- Whether team ratings should decay over time
- Whether recent form should be weighted more heavily
- Whether confederation strength should be modeled explicitly
- Whether market odds should be used as a benchmark or feature
- Whether ensemble weights should be learned through validation
