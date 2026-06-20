# Data Source Registry

## Purpose

This registry defines the data sources required to evolve the forecasting engine from a reproducible scaffold into a real pre-match forecasting agent.

Every future data source should be evaluated for:

- legality
- reproducibility
- update frequency
- timestamp availability
- leakage risk
- historical coverage
- modeling value

## Data Source Categories

| Category | Examples | Priority | Notes |
|---|---|---:|---|
| Match results | historical international results | High | Core training data |
| Fixtures | World Cup schedule, groups, venues | High | Required for prediction and simulation |
| Team strength | Elo, FIFA rankings | High | Core baseline signal |
| Market odds | opening odds, closing odds | High | Needed for betting-grade evaluation |
| Player data | appearances, minutes, squad strength | Medium | Valuable but harder to maintain |
| Availability | injuries, suspensions, expected lineups | High | Strong signal if timestamped |
| Rest/travel | rest days, distance, time zones | Medium | Useful context signal |
| Weather | temperature, wind, humidity, rain | Medium | Useful for match context |
| Venue | altitude, host nation, stadium | Medium | Useful but must avoid overfitting |
| Micro-context | hotel quality, bed quality proxies | Low/Experimental | Must be legal, historical, and validated |

## Source Evaluation Template

Each source should be documented with:

- Source name:
- Source URL or provider:
- Data type:
- Free/public, paid, or manual:
- License/terms notes:
- Historical coverage:
- Refresh frequency:
- Available before kickoff:
- Timestamped:
- Leakage risk:
- Reliability:
- Planned use:
- Fallback if unavailable:

## Initial Candidate Sources

### Historical Results

Purpose:

- train Elo
- train baseline models
- evaluate historical performance
- create rolling form features

Required fields:

- match date
- home/team A
- away/team B
- score
- tournament
- neutral-site flag
- venue/location where available

Leakage risk:

- Low if only final results from matches before prediction timestamp are used.

### FIFA Rankings

Purpose:

- team strength baseline
- ranking difference feature
- FIFA points difference feature

Required fields:

- ranking date
- team
- rank
- points

Leakage risk:

- Medium if ranking publication date is not respected.

### Betting Odds

Purpose:

- market baseline
- implied probability features
- de-vigged probability comparison
- edge calculation
- closing-line value

Required fields:

- bookmaker
- match
- timestamp or odds type
- home/team A odds
- draw odds
- away/team B odds
- opening/closing indicator if available

Leakage risk:

- High if closing odds are used as training features for predictions supposedly made before closing.

Safe usage:

- Use opening odds or odds available at prediction timestamp as features.
- Use closing odds for evaluation, not pre-match features, unless the prediction timestamp is immediately before kickoff.

### Weather

Purpose:

- match context
- fatigue and style interaction
- temperature/humidity/wind/rain features

Required fields:

- venue
- match date/time
- forecast timestamp
- temperature
- humidity
- wind speed
- precipitation probability
- weather condition

Leakage risk:

- Medium if actual observed weather is used instead of forecast weather for historical prediction rows.

Safe usage:

- Use historical forecasts where available.
- If unavailable, use actual historical weather only as a proxy and document the limitation.

### Player Availability

Purpose:

- injury/suspension impact
- lineup strength
- goalkeeper/captain/star player availability

Required fields:

- team
- player
- status
- source date
- match date
- expected availability
- confidence

Leakage risk:

- High if final starting lineups are used too early.

Safe usage:

- Use only information available at the prediction timestamp.
- Store manual updates in timestamped CSV files.

### Hotel and Sleep Quality Proxies

Purpose:

- experimental micro-context signal
- travel comfort proxy
- recovery quality proxy

Potential fields:

- team
- hotel city
- hotel rating
- review-derived sleep-quality proxy
- travel distance from venue
- altitude difference
- noise risk proxy

Leakage risk:

- High if data is private, speculative, or not historically reproducible.

Safe usage:

- Do not use private location data.
- Do not scrape unclear sources.
- Use only legal public or manually entered data.
- Treat as experimental unless historically backtested.

## Feature Admission Rules

A source should not become a core model feature unless:

1. It is legally usable.
2. It is reproducible.
3. It is available before kickoff.
4. It can be historically backtested.
5. It improves out-of-sample metrics.
6. It does not introduce leakage.
7. It has fallback handling for missing values.

## Priority Build Order

1. Historical results
2. Market odds
3. FIFA rankings
4. Prediction ledger
5. Weather/venue data
6. Rest and travel features
7. Player availability manual inputs
8. Calibration reports
9. Advanced player/squad data
10. Experimental micro-context features
