from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from wc_forecast.data.ingest_results import load_historical_results

DEFAULT_MAX_GOALS = 10
DEFAULT_MIN_EXPECTED_GOALS = 0.20
DEFAULT_MAX_EXPECTED_GOALS = 5.00
DEFAULT_HOME_ADVANTAGE_MULTIPLIER = 1.10


@dataclass(frozen=True)
class PoissonPrediction:
    """Expected-goals and scoreline probability forecast for one match."""

    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    most_likely_score: str
    most_likely_score_probability: float


def poisson_pmf(goals: int, expected_goals: float) -> float:
    """Compute Poisson probability mass for a goal count."""

    if goals < 0:
        raise ValueError("goals must be non-negative.")

    if expected_goals <= 0:
        raise ValueError("expected_goals must be positive.")

    return math.exp(-expected_goals) * (expected_goals**goals) / math.factorial(goals)


def _clip_expected_goals(value: float) -> float:
    """Keep expected-goals estimates inside a practical modeling range."""

    return max(DEFAULT_MIN_EXPECTED_GOALS, min(DEFAULT_MAX_EXPECTED_GOALS, value))


class PoissonGoalsModel:
    """Simple team attack/defense Poisson model for football score forecasting."""

    def __init__(
        self,
        max_goals: int = DEFAULT_MAX_GOALS,
        home_advantage_multiplier: float = DEFAULT_HOME_ADVANTAGE_MULTIPLIER,
    ) -> None:
        if max_goals < 3:
            raise ValueError("max_goals must be at least 3.")

        self.max_goals = max_goals
        self.home_advantage_multiplier = home_advantage_multiplier
        self.global_goals_per_team_match = 1.25
        self.attack_strength: dict[str, float] = {}
        self.defense_weakness: dict[str, float] = {}

    def fit(self, results: pd.DataFrame) -> None:
        """Estimate team attack and defense strengths from historical results."""

        home_rows = pd.DataFrame(
            {
                "team": results["home_team"],
                "goals_for": results["home_score"],
                "goals_against": results["away_score"],
            }
        )
        away_rows = pd.DataFrame(
            {
                "team": results["away_team"],
                "goals_for": results["away_score"],
                "goals_against": results["home_score"],
            }
        )

        team_match_rows = pd.concat([home_rows, away_rows], ignore_index=True)
        team_match_rows["goals_for"] = pd.to_numeric(
            team_match_rows["goals_for"],
            errors="raise",
        )
        team_match_rows["goals_against"] = pd.to_numeric(
            team_match_rows["goals_against"],
            errors="raise",
        )

        self.global_goals_per_team_match = float(team_match_rows["goals_for"].mean())

        if self.global_goals_per_team_match <= 0:
            raise ValueError("Cannot fit Poisson model with zero average goals.")

        team_stats = team_match_rows.groupby("team").agg(
            goals_for_per_match=("goals_for", "mean"),
            goals_against_per_match=("goals_against", "mean"),
        )

        self.attack_strength = (
            team_stats["goals_for_per_match"] / self.global_goals_per_team_match
        ).to_dict()

        self.defense_weakness = (
            team_stats["goals_against_per_match"] / self.global_goals_per_team_match
        ).to_dict()

    def _team_attack(self, team: str) -> float:
        return float(self.attack_strength.get(team, 1.0))

    def _team_defense_weakness(self, team: str) -> float:
        return float(self.defense_weakness.get(team, 1.0))

    def expected_goals(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> tuple[float, float]:
        """Estimate expected goals for both teams before kickoff."""

        home_multiplier = 1.0 if neutral else self.home_advantage_multiplier
        away_multiplier = 1.0 if neutral else 1.0 / self.home_advantage_multiplier

        expected_home = (
            self.global_goals_per_team_match
            * self._team_attack(home_team)
            * self._team_defense_weakness(away_team)
            * home_multiplier
        )
        expected_away = (
            self.global_goals_per_team_match
            * self._team_attack(away_team)
            * self._team_defense_weakness(home_team)
            * away_multiplier
        )

        return _clip_expected_goals(expected_home), _clip_expected_goals(expected_away)

    def scoreline_probabilities(
        self,
        expected_home_goals: float,
        expected_away_goals: float,
    ) -> pd.DataFrame:
        """Return normalized scoreline probabilities up to max_goals."""

        rows: list[dict[str, float | int | str]] = []

        for home_goals in range(self.max_goals + 1):
            home_probability = poisson_pmf(home_goals, expected_home_goals)

            for away_goals in range(self.max_goals + 1):
                away_probability = poisson_pmf(away_goals, expected_away_goals)
                probability = home_probability * away_probability

                if home_goals > away_goals:
                    outcome = "home_win"
                elif home_goals == away_goals:
                    outcome = "draw"
                else:
                    outcome = "away_win"

                rows.append(
                    {
                        "home_goals": home_goals,
                        "away_goals": away_goals,
                        "scoreline": f"{home_goals}-{away_goals}",
                        "outcome": outcome,
                        "probability": probability,
                    }
                )

        scorelines = pd.DataFrame(rows)
        total_probability = float(scorelines["probability"].sum())

        if total_probability <= 0:
            raise ValueError("Scoreline probabilities sum to zero.")

        scorelines["probability"] = scorelines["probability"] / total_probability

        return scorelines

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = True,
    ) -> PoissonPrediction:
        """Predict expected goals, outcome probabilities, and likely scoreline."""

        expected_home_goals, expected_away_goals = self.expected_goals(
            home_team=home_team,
            away_team=away_team,
            neutral=neutral,
        )

        scorelines = self.scoreline_probabilities(
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
        )

        outcome_probabilities = scorelines.groupby("outcome")["probability"].sum()
        most_likely = scorelines.sort_values(
            "probability",
            ascending=False,
        ).iloc[0]

        return PoissonPrediction(
            home_team=home_team,
            away_team=away_team,
            expected_home_goals=expected_home_goals,
            expected_away_goals=expected_away_goals,
            prob_home_win=float(outcome_probabilities.get("home_win", 0.0)),
            prob_draw=float(outcome_probabilities.get("draw", 0.0)),
            prob_away_win=float(outcome_probabilities.get("away_win", 0.0)),
            most_likely_score=str(most_likely["scoreline"]),
            most_likely_score_probability=float(most_likely["probability"]),
        )


def save_poisson_prediction(
    results_path: str | Path,
    home_team: str,
    away_team: str,
    output_path: str | Path,
    neutral: bool = True,
) -> Path:
    """Fit Poisson model from historical results and save one match prediction."""

    results = load_historical_results(results_path)

    model = PoissonGoalsModel()
    model.fit(results)
    prediction = model.predict_match(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
    )

    prediction_row = pd.DataFrame(
        [
            {
                "home_team": prediction.home_team,
                "away_team": prediction.away_team,
                "neutral": neutral,
                "expected_home_goals": prediction.expected_home_goals,
                "expected_away_goals": prediction.expected_away_goals,
                "prob_home_win": prediction.prob_home_win,
                "prob_draw": prediction.prob_draw,
                "prob_away_win": prediction.prob_away_win,
                "most_likely_score": prediction.most_likely_score,
                "most_likely_score_probability": (
                    prediction.most_likely_score_probability
                ),
            }
        ]
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    prediction_row.to_csv(destination, index=False)

    return destination
