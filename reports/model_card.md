# Model Card: World Cup Match Forecasting Engine

## Model Name

World Cup Match Forecasting Engine

## Project Type

Probabilistic football forecasting system for FIFA World Cup-style matches.

## Intended Use

This project estimates pre-match probabilities for football outcomes:

- Team A / home win
- Draw
- Team B / away win

It also produces:

- Expected goals estimates
- Most likely scoreline
- Ensemble forecast probabilities
- Forecast entropy
- Model-layer disagreement
- Group-stage advancement probabilities

The system is intended as a portfolio project demonstrating quantitative modeling, ML engineering, backtesting, leakage prevention, and reporting practices.

## Not Intended For

This project is not intended for:

- Real betting decisions
- Financial trading decisions
- Claims of certain match prediction
- Production deployment without larger datasets and validation
- Automated decision-making involving real money

## Modeling Layers

### Elo Rating Layer

The Elo layer estimates team strength from historical match results.

It includes:

- Default initial team ratings
- Match-importance weights
- Margin-of-victory adjustment
- Neutral-site handling
- Pre-match expected score estimates

### Feature Engineering Layer

The feature builder replays historical matches chronologically and captures only pre-match information.

Current feature examples:

- Home-team pre-match Elo
- Away-team pre-match Elo
- Elo rating difference
- Absolute Elo difference
- Elo expected score
- Neutral-site flag
- Tournament importance

### Logistic Regression Layer

The logistic regression model predicts a three-class outcome:

- home_win
- draw
- away_win

It serves as an interpretable baseline rather than the final desired model.

### Poisson Expected-Goals Layer

The Poisson model estimates expected goals for each team based on historical attack and defensive weakness.

It converts expected goals into:

- Scoreline probability table
- Home win probability
- Draw probability
- Away win probability
- Most likely scoreline

### Ensemble Layer

The current ensemble is a transparent weighted average of:

- Logistic regression probabilities
- Poisson outcome probabilities

It also reports:

- Final predicted outcome
- Confidence label
- Normalized entropy
- Maximum model disagreement

## Data

The committed sample data is intentionally small and used only to make the repository reproducible for reviewers.

The current sample data should not be interpreted as sufficient for real predictive evaluation.

Future versions should integrate larger public datasets, such as:

- Historical international match results
- FIFA rankings
- Elo-style public team ratings or custom expanded Elo history
- Public betting odds where legally usable
- Manually curated injuries, suspensions, rest, travel, and squad data

## Leakage Prevention

The project is designed to avoid post-match leakage.

For each historical match:

1. Pre-match team ratings are recorded.
2. Feature rows are created.
3. Match results are added only as target/result columns.
4. Elo ratings are updated only after the row is created.

Final scores, outcomes, and post-match statistics should never be used as model features for pre-match prediction.

## Evaluation

Current evaluation includes:

- Chronological train/test split
- Accuracy
- Log loss
- Multiclass Brier score
- Match-level prediction outputs
- Markdown backtest report

Future evaluation should add:

- Rolling-window validation
- Expanding-window validation
- Calibration curves
- Expected calibration error
- Confusion matrix
- Performance by favorite/underdog
- Performance by tournament type
- Performance by confidence bucket
- Market baseline comparison

## Known Limitations

Current limitations:

- Small demo dataset
- Simplified Poisson model
- No real player availability features
- No live injury or lineup data
- No market odds integration
- No calibrated ensemble weights
- Simplified group-stage tie-breakers
- No knockout-stage simulation yet
- No weather, venue altitude, travel, or rest features yet

## Ethical and Legal Data Use

This project should only use public, legal, reproducible data sources.

Data sources should be documented with:

- Source name
- Access method
- Access date
- License or usage constraints
- Transformation assumptions

The project should avoid scraping sources where terms of service are unclear.

## Reliability Notes

Forecast outputs should be interpreted as uncertain probabilities.

High confidence does not mean certainty. Model disagreement and entropy should be reviewed before interpreting any forecast.

## Future Improvements

Planned improvements:

- Larger historical dataset
- Rolling form features
- Strength-of-schedule features
- FIFA ranking features
- Market-implied probability features
- Calibrated ensemble weights
- Calibration plots
- Tournament knockout simulator
- FastAPI prediction service
- Dashboard
- Automated model cards and report generation
