from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.data.ingest_results import load_historical_results
from wc_forecast.features.build_features import FEATURE_COLUMNS, build_match_features
from wc_forecast.models.classifier import OUTCOME_ORDER, train_logistic_regression
from wc_forecast.models.elo import EloModel, match_importance_weight
from wc_forecast.models.ensemble import combine_match_prediction
from wc_forecast.models.poisson import PoissonGoalsModel


def _format_probability(value: float) -> str:
    """Format a probability as a percentage string."""

    return f"{value * 100:.1f}%"


def _model_confidence(probabilities: dict[str, float]) -> str:
    """Assign a simple confidence label from the strongest class probability."""

    strongest_probability = max(probabilities.values())

    if strongest_probability >= 0.60:
        return "High"

    if strongest_probability >= 0.45:
        return "Medium"

    return "Low"


def build_current_match_features(
    results: pd.DataFrame,
    home_team: str,
    away_team: str,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
) -> pd.DataFrame:
    """Build one current pre-match feature row from historical results."""

    elo_model = EloModel()
    elo_model.fit(results)

    elo_prediction = elo_model.predict_match(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
    )

    is_world_cup = tournament.strip().lower() == "fifa world cup"

    row = {
        "home_elo_pre": elo_prediction.home_rating,
        "away_elo_pre": elo_prediction.away_rating,
        "elo_diff_home_minus_away": (
            elo_prediction.home_rating - elo_prediction.away_rating
        ),
        "abs_elo_diff": abs(elo_prediction.home_rating - elo_prediction.away_rating),
        "elo_expected_home_score": elo_prediction.expected_home_score,
        "elo_expected_away_score": elo_prediction.expected_away_score,
        "is_neutral": int(neutral),
        "is_world_cup": int(is_world_cup),
        "tournament_importance": match_importance_weight(tournament),
    }

    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def generate_match_prediction(
    results: pd.DataFrame,
    home_team: str,
    away_team: str,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
) -> dict[str, object]:
    """Generate a combined logistic-regression and Poisson match forecast."""

    historical_features = build_match_features(results)
    classifier = train_logistic_regression(historical_features)

    current_features = build_current_match_features(
        results=results,
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        neutral=neutral,
    )

    raw_probabilities = classifier.predict_proba(current_features[FEATURE_COLUMNS])[0]
    classifier_probabilities = {outcome: 0.0 for outcome in OUTCOME_ORDER}

    for outcome, probability in zip(classifier.classes_, raw_probabilities, strict=True):
        classifier_probabilities[str(outcome)] = float(probability)

    poisson_model = PoissonGoalsModel()
    poisson_model.fit(results)
    poisson_prediction = poisson_model.predict_match(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
    )

    prediction = {
        "home_team": home_team,
        "away_team": away_team,
        "tournament": tournament,
        "neutral": neutral,
        "logistic_prob_home_win": classifier_probabilities["home_win"],
        "logistic_prob_draw": classifier_probabilities["draw"],
        "logistic_prob_away_win": classifier_probabilities["away_win"],
        "poisson_expected_home_goals": poisson_prediction.expected_home_goals,
        "poisson_expected_away_goals": poisson_prediction.expected_away_goals,
        "poisson_prob_home_win": poisson_prediction.prob_home_win,
        "poisson_prob_draw": poisson_prediction.prob_draw,
        "poisson_prob_away_win": poisson_prediction.prob_away_win,
        "most_likely_score": poisson_prediction.most_likely_score,
        "most_likely_score_probability": (
            poisson_prediction.most_likely_score_probability
        ),
        "logistic_confidence": _model_confidence(classifier_probabilities),
        "home_win_model_disagreement": abs(
            classifier_probabilities["home_win"] - poisson_prediction.prob_home_win
        ),
    }

    ensemble = combine_match_prediction(prediction)

    prediction.update(
        {
            "ensemble_prob_home_win": ensemble.prob_home_win,
            "ensemble_prob_draw": ensemble.prob_draw,
            "ensemble_prob_away_win": ensemble.prob_away_win,
            "ensemble_predicted_outcome": ensemble.predicted_outcome,
            "ensemble_confidence": ensemble.confidence,
            "ensemble_entropy": ensemble.entropy,
            "max_model_disagreement": ensemble.max_model_disagreement,
        }
    )

    return prediction


def prediction_to_frame(prediction: dict[str, object]) -> pd.DataFrame:
    """Convert one prediction dictionary to a single-row dataframe."""

    return pd.DataFrame([prediction])


def render_match_prediction_report(prediction: dict[str, object]) -> str:
    """Render an analyst-style Markdown report for one match prediction."""

    home_team = str(prediction["home_team"])
    away_team = str(prediction["away_team"])

    logistic_home = float(prediction["logistic_prob_home_win"])
    logistic_draw = float(prediction["logistic_prob_draw"])
    logistic_away = float(prediction["logistic_prob_away_win"])

    poisson_home = float(prediction["poisson_prob_home_win"])
    poisson_draw = float(prediction["poisson_prob_draw"])
    poisson_away = float(prediction["poisson_prob_away_win"])

    scoreline_probability = _format_probability(
        float(prediction["most_likely_score_probability"])
    )
    home_win_disagreement = _format_probability(
        float(prediction["home_win_model_disagreement"])
    )
    ensemble_home = float(prediction["ensemble_prob_home_win"])
    ensemble_draw = float(prediction["ensemble_prob_draw"])
    ensemble_away = float(prediction["ensemble_prob_away_win"])
    ensemble_entropy = float(prediction["ensemble_entropy"])
    max_disagreement = _format_probability(
        float(prediction["max_model_disagreement"])
    )

    return f"""# Match Prediction Report: {home_team} vs {away_team}

## Forecast Summary

This report combines the current logistic-regression baseline with the Poisson
expected-goals model.

The output should be interpreted as a probabilistic model report, not as a claim
that the match result can be predicted with certainty.

## Logistic Regression Baseline

| Outcome | Probability |
|---|---:|
| {home_team} win | {_format_probability(logistic_home)} |
| Draw | {_format_probability(logistic_draw)} |
| {away_team} win | {_format_probability(logistic_away)} |

Model confidence: **{prediction["logistic_confidence"]}**

## Poisson Expected-Goals Forecast

| Field | Value |
|---|---:|
| {home_team} expected goals | {float(prediction["poisson_expected_home_goals"]):.3f} |
| {away_team} expected goals | {float(prediction["poisson_expected_away_goals"]):.3f} |
| {home_team} win probability | {_format_probability(poisson_home)} |
| Draw probability | {_format_probability(poisson_draw)} |
| {away_team} win probability | {_format_probability(poisson_away)} |
| Most likely scoreline | {prediction["most_likely_score"]} |
| Scoreline probability | {scoreline_probability} |

## Ensemble Forecast

| Outcome | Probability |
|---|---:|
| {home_team} win | {_format_probability(ensemble_home)} |
| Draw | {_format_probability(ensemble_draw)} |
| {away_team} win | {_format_probability(ensemble_away)} |

| Field | Value |
|---|---:|
| Predicted outcome | {prediction["ensemble_predicted_outcome"]} |
| Ensemble confidence | {prediction["ensemble_confidence"]} |
| Normalized entropy | {ensemble_entropy:.3f} |
| Max model disagreement | {max_disagreement} |

## Model Layer Comparison

| Signal | Value |
|---|---:|
| Logistic {home_team} win probability | {_format_probability(logistic_home)} |
| Poisson {home_team} win probability | {_format_probability(poisson_home)} |
| Absolute home-win disagreement | {home_win_disagreement} |

## Interpretation

The logistic model uses engineered pre-match features such as Elo rating
difference, expected Elo score, neutral-site status, and tournament importance.

The Poisson model estimates expected goals from historical team attack and
defense profiles, then converts expected goals into scoreline probabilities.

When the two layers disagree, the forecast should be treated with more caution.
In a future ensemble model, this disagreement can become an explicit uncertainty
or risk feature.

## Caveats

- The current committed dataset is intentionally small for reproducible demo use.
- The current probabilities are pipeline outputs, not validated betting signals.
- Real predictive evaluation requires a larger historical international dataset.
- Injury, lineup, rest, travel, market odds, and squad-strength features are not
  included yet.
- The Poisson model is a transparent baseline and does not yet include advanced
  hierarchical or time-decay effects.
"""


def save_match_prediction_report(
    results_path: str | Path,
    home_team: str,
    away_team: str,
    prediction_output_path: str | Path,
    report_output_path: str | Path,
    tournament: str = "FIFA World Cup",
    neutral: bool = True,
) -> tuple[Path, Path]:
    """Generate and save a match prediction CSV and Markdown report."""

    results = load_historical_results(results_path)
    prediction = generate_match_prediction(
        results=results,
        home_team=home_team,
        away_team=away_team,
        tournament=tournament,
        neutral=neutral,
    )

    prediction_destination = Path(prediction_output_path)
    report_destination = Path(report_output_path)

    prediction_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.parent.mkdir(parents=True, exist_ok=True)

    prediction_to_frame(prediction).to_csv(prediction_destination, index=False)
    report_destination.write_text(render_match_prediction_report(prediction))

    return prediction_destination, report_destination
