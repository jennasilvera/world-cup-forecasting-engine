from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline

from wc_forecast.features.build_features import FEATURE_COLUMNS
from wc_forecast.models.classifier import (
    PREDICTION_COLUMNS,
    train_logistic_regression,
)
from wc_forecast.models.elo import EloModel

REQUIRED_FIXTURE_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "tournament",
    "neutral",
]

TEAM_ALIASES = {
    "Curacao": "Curaçao",
    "Czechia": "Czech Republic",
    "DR Congo": "DR Congo",
    "Ivory Coast": "Ivory Coast",
    "Cape Verde": "Cape Verde",
    "South Korea": "South Korea",
    "United States": "United States",
    "USA": "United States",
}

FORECAST_OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "tournament",
    "neutral",
    "home_elo_rating",
    "away_elo_rating",
    "home_rating_source",
    "away_rating_source",
    "rating_warning",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
    "predicted_outcome",
    "predicted_winner",
    "model_confidence",
]


def build_ratings_before_cutoff(
    results: pd.DataFrame,
    rating_cutoff_date: str,
) -> pd.DataFrame:
    """Build Elo ratings using only matches before the rating cutoff date."""

    cutoff = pd.to_datetime(rating_cutoff_date, errors="raise")

    historical_results = results.copy()
    historical_results["date"] = pd.to_datetime(
        historical_results["date"],
        errors="raise",
    )
    historical_results = historical_results[historical_results["date"] < cutoff].copy()

    if historical_results.empty:
        raise ValueError("No historical result rows found before rating_cutoff_date.")

    model = EloModel()
    model.fit(historical_results)

    return model.ratings_table()


def validate_fixture_slate(fixtures: pd.DataFrame) -> None:
    """Validate fixture slate schema before forecasting."""

    missing_columns = sorted(set(REQUIRED_FIXTURE_COLUMNS) - set(fixtures.columns))

    if missing_columns:
        raise ValueError(f"Fixture slate missing columns: {missing_columns}")

    if fixtures.empty:
        raise ValueError("Fixture slate is empty.")

    if fixtures[["home_team", "away_team"]].isna().any().any():
        raise ValueError("Fixture slate contains missing team names.")

    if (fixtures["home_team"] == fixtures["away_team"]).any():
        raise ValueError("Fixture slate contains identical teams.")


def train_model_before_cutoff(
    features: pd.DataFrame,
    train_cutoff_date: str,
    sample_weight_half_life_days: float | None = None,
) -> Pipeline:
    """Train model using only rows before the forecast cutoff date."""

    cutoff = pd.to_datetime(train_cutoff_date, errors="raise")

    training_features = features.copy()
    training_features["date"] = pd.to_datetime(
        training_features["date"],
        errors="raise",
    )

    training_features = training_features[training_features["date"] < cutoff].copy()

    if training_features.empty:
        raise ValueError("No training rows found before train_cutoff_date.")

    return train_logistic_regression(
        training_features,
        sample_weight_half_life_days=sample_weight_half_life_days,
    )


def build_fixture_forecast_features(
    fixtures: pd.DataFrame,
    ratings: pd.DataFrame,
) -> pd.DataFrame:
    """Create model feature rows for unplayed fixtures."""

    validate_fixture_slate(fixtures)

    fixture_features = fixtures[REQUIRED_FIXTURE_COLUMNS].copy()
    fixture_features["date"] = pd.to_datetime(fixture_features["date"], errors="raise")
    fixture_features["neutral"] = fixture_features["neutral"].map(_coerce_bool)

    ratings_lookup = _ratings_lookup(ratings)

    rows: list[dict[str, object]] = []

    for fixture in fixture_features.itertuples(index=False):
        home_rating = _team_rating(ratings_lookup, fixture.home_team)
        away_rating = _team_rating(ratings_lookup, fixture.away_team)
        home_rating_source = _rating_source(ratings_lookup, fixture.home_team)
        away_rating_source = _rating_source(ratings_lookup, fixture.away_team)
        home_expected = _elo_expected_score(home_rating, away_rating)
        away_expected = 1.0 - home_expected

        feature_values = _empty_feature_values()
        _set_feature_value(feature_values, "home_elo", home_rating)
        _set_feature_value(feature_values, "away_elo", away_rating)
        _set_feature_value(feature_values, "elo_diff", home_rating - away_rating)
        _set_feature_value(feature_values, "abs_elo_diff", abs(home_rating - away_rating))
        _set_feature_value(feature_values, "home_expected", home_expected)
        _set_feature_value(feature_values, "away_expected", away_expected)
        _set_feature_value(feature_values, "neutral", float(fixture.neutral))
        _set_feature_value(
            feature_values,
            "world_cup",
            float("World Cup" in str(fixture.tournament)),
        )
        _set_feature_value(
            feature_values,
            "tournament_weight",
            _tournament_weight(str(fixture.tournament)),
        )

        row = {
            "date": fixture.date,
            "home_team": fixture.home_team,
            "away_team": fixture.away_team,
            "tournament": fixture.tournament,
            "neutral": fixture.neutral,
            "home_elo_rating": home_rating,
            "away_elo_rating": away_rating,
            "home_rating_source": home_rating_source,
            "away_rating_source": away_rating_source,
            "rating_warning": _rating_warning(
                home_team=str(fixture.home_team),
                away_team=str(fixture.away_team),
                home_rating_source=home_rating_source,
                away_rating_source=away_rating_source,
            ),
        }
        row.update(feature_values)
        rows.append(row)

    return pd.DataFrame(rows)


def forecast_fixtures(
    fixtures: pd.DataFrame,
    features: pd.DataFrame,
    ratings: pd.DataFrame,
    train_cutoff_date: str,
    sample_weight_half_life_days: float | None = None,
) -> pd.DataFrame:
    """Train before cutoff and forecast a fixture slate."""

    model = train_model_before_cutoff(
        features=features,
        train_cutoff_date=train_cutoff_date,
        sample_weight_half_life_days=sample_weight_half_life_days,
    )

    fixture_features = build_fixture_forecast_features(
        fixtures=fixtures,
        ratings=ratings,
    )

    probabilities = _predict_fixture_probabilities(
        model=model,
        fixture_features=fixture_features,
    )

    forecasts = fixture_features[
        [
            "date",
            "home_team",
            "away_team",
            "tournament",
            "neutral",
            "home_elo_rating",
            "away_elo_rating",
            "home_rating_source",
            "away_rating_source",
            "rating_warning",
        ]
    ].copy()

    forecasts["prob_home_win"] = probabilities["prob_home_win"]
    forecasts["prob_draw"] = probabilities["prob_draw"]
    forecasts["prob_away_win"] = probabilities["prob_away_win"]

    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    forecasts["predicted_outcome"] = forecasts[probability_columns].idxmax(axis=1)
    forecasts["predicted_outcome"] = forecasts["predicted_outcome"].str.replace(
        "prob_",
        "",
        regex=False,
    )

    forecasts["predicted_winner"] = forecasts.apply(_predicted_winner, axis=1)
    forecasts["model_confidence"] = forecasts[probability_columns].max(axis=1)

    forecasts["date"] = pd.to_datetime(forecasts["date"]).dt.date.astype(str)

    return forecasts[FORECAST_OUTPUT_COLUMNS]


def save_fixture_forecasts(
    fixtures_path: str | Path,
    features_path: str | Path,
    ratings_path: str | Path,
    output_path: str | Path,
    train_cutoff_date: str,
    sample_weight_half_life_days: float | None = None,
) -> pd.DataFrame:
    """Load inputs, forecast fixtures, and save forecast CSV."""

    fixtures = pd.read_csv(fixtures_path)
    features = pd.read_csv(features_path)
    ratings = pd.read_csv(ratings_path)

    forecasts = forecast_fixtures(
        fixtures=fixtures,
        features=features,
        ratings=ratings,
        train_cutoff_date=train_cutoff_date,
        sample_weight_half_life_days=sample_weight_half_life_days,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(destination, index=False)

    return forecasts


def save_fixture_forecasts_from_results(
    fixtures_path: str | Path,
    features_path: str | Path,
    results_path: str | Path,
    output_path: str | Path,
    train_cutoff_date: str,
    rating_cutoff_date: str | None = None,
    sample_weight_half_life_days: float | None = None,
) -> pd.DataFrame:
    """Load inputs, build cutoff-safe ratings, forecast fixtures, and save CSV."""

    fixtures = pd.read_csv(fixtures_path)
    features = pd.read_csv(features_path)
    results = pd.read_csv(results_path)

    effective_rating_cutoff = rating_cutoff_date or train_cutoff_date
    ratings = build_ratings_before_cutoff(
        results=results,
        rating_cutoff_date=effective_rating_cutoff,
    )

    forecasts = forecast_fixtures(
        fixtures=fixtures,
        features=features,
        ratings=ratings,
        train_cutoff_date=train_cutoff_date,
        sample_weight_half_life_days=sample_weight_half_life_days,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(destination, index=False)

    return forecasts


def _predict_fixture_probabilities(
    model: Pipeline,
    fixture_features: pd.DataFrame,
) -> pd.DataFrame:
    """Predict aligned home/draw/away probabilities for fixture feature rows."""

    raw_probabilities = model.predict_proba(fixture_features[FEATURE_COLUMNS])
    model_classes = list(model.classes_)

    probabilities = pd.DataFrame(
        0.0,
        index=fixture_features.index,
        columns=PREDICTION_COLUMNS,
    )

    for class_index, class_name in enumerate(model_classes):
        probabilities[f"prob_{class_name}"] = raw_probabilities[:, class_index]

    return probabilities.reset_index(drop=True)


def _ratings_lookup(ratings: pd.DataFrame) -> dict[str, float]:
    """Convert ratings table into a normalized lookup."""

    required = {"team", "elo_rating"}
    missing = sorted(required - set(ratings.columns))

    if missing:
        raise ValueError(f"Ratings table missing columns: {missing}")

    return {
        str(row.team).strip(): float(row.elo_rating)
        for row in ratings.itertuples(index=False)
    }


def _rating_source(ratings_lookup: dict[str, float], team: str) -> str:
    """Return whether a rating came from direct lookup, alias lookup, or fallback."""

    cleaned = str(team).strip()
    normalized_team = _normalize_team_name(cleaned)

    if normalized_team in ratings_lookup and normalized_team != cleaned:
        return "alias_lookup"

    if normalized_team in ratings_lookup:
        return "rating_lookup"

    return "fallback_1500"


def _rating_warning(
    home_team: str,
    away_team: str,
    home_rating_source: str,
    away_rating_source: str,
) -> str:
    """Return a warning when a forecast uses fallback Elo ratings."""

    fallback_teams = []

    if home_rating_source == "fallback_1500":
        fallback_teams.append(home_team)

    if away_rating_source == "fallback_1500":
        fallback_teams.append(away_team)

    if not fallback_teams:
        return ""

    return "fallback_rating_used:" + ",".join(fallback_teams)


def _team_rating(ratings_lookup: dict[str, float], team: str) -> float:
    """Return team rating, resolving aliases before using fallback rating."""

    normalized_team = _normalize_team_name(team)

    return ratings_lookup.get(normalized_team, 1500.0)


def _normalize_team_name(team: str) -> str:
    """Normalize common fixture team aliases to historical-results names."""

    cleaned = str(team).strip()

    return TEAM_ALIASES.get(cleaned, cleaned)


def _elo_expected_score(home_rating: float, away_rating: float) -> float:
    """Calculate Elo expected score for the home team."""

    return 1.0 / (1.0 + 10.0 ** ((away_rating - home_rating) / 400.0))


def _empty_feature_values() -> dict[str, float]:
    """Initialize all model features to zero."""

    return {column: 0.0 for column in FEATURE_COLUMNS}


def _set_feature_value(
    feature_values: dict[str, float],
    token: str,
    value: float,
) -> None:
    """Set feature columns whose names contain a token."""

    for column in FEATURE_COLUMNS:
        if token in column:
            feature_values[column] = value


def _tournament_weight(tournament: str) -> float:
    """Approximate tournament importance for fixture feature generation."""

    normalized = tournament.lower()

    if "world cup" in normalized and "qualification" not in normalized:
        return 1.0

    if "qualification" in normalized:
        return 0.75

    if "nations league" in normalized:
        return 0.65

    if "friendly" in normalized:
        return 0.35

    return 0.50


def _coerce_bool(value: object) -> bool:
    """Convert common CSV bool-like values to bool."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes", "y"}:
        return True

    if normalized in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Cannot convert value to bool: {value}")


def _predicted_winner(row: pd.Series) -> str:
    """Map predicted outcome to the named team or draw."""

    if row["predicted_outcome"] == "home_win":
        return str(row["home_team"])

    if row["predicted_outcome"] == "away_win":
        return str(row["away_team"])

    return "Draw"
