from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from wc_forecast.data.ingest_results import load_historical_results
from wc_forecast.models.poisson import PoissonGoalsModel

REQUIRED_FIXTURE_COLUMNS = [
    "group",
    "home_team",
    "away_team",
    "neutral",
]


@dataclass(frozen=True)
class GroupStageSimulationResult:
    """Container for group-stage simulation outputs."""

    summary: pd.DataFrame
    standings: pd.DataFrame


def _parse_bool(value: object) -> bool:
    """Parse common boolean values from fixture CSV files."""

    if isinstance(value, bool):
        return value

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes", "y"}:
        return True

    if normalized in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Invalid neutral value: {value!r}")


def load_group_fixtures(input_path: str | Path) -> pd.DataFrame:
    """Load and validate group-stage fixture definitions."""

    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Group-stage fixture file not found: {path}")

    fixtures = pd.read_csv(path)
    missing_columns = sorted(set(REQUIRED_FIXTURE_COLUMNS) - set(fixtures.columns))

    if missing_columns:
        raise ValueError(f"Fixture table missing required columns: {missing_columns}")

    if fixtures.empty:
        raise ValueError("Fixture table is empty.")

    cleaned = fixtures.copy()

    for column in ["group", "home_team", "away_team"]:
        cleaned[column] = cleaned[column].astype(str).str.strip()

        if cleaned[column].eq("").any():
            raise ValueError(f"Fixture column contains blank values: {column}")

    if (cleaned["home_team"] == cleaned["away_team"]).any():
        raise ValueError("A team cannot play itself in fixture table.")

    cleaned["neutral"] = cleaned["neutral"].map(_parse_bool)

    return cleaned


def _initial_standings(fixtures: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Create blank standings rows for all teams in a group."""

    teams = sorted(set(fixtures["home_team"]) | set(fixtures["away_team"]))

    return {
        team: {
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "points": 0,
        }
        for team in teams
    }


def _apply_match_result(
    standings: dict[str, dict[str, int]],
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int,
) -> None:
    """Update standings after one simulated match."""

    standings[home_team]["played"] += 1
    standings[away_team]["played"] += 1

    standings[home_team]["goals_for"] += home_goals
    standings[home_team]["goals_against"] += away_goals

    standings[away_team]["goals_for"] += away_goals
    standings[away_team]["goals_against"] += home_goals

    standings[home_team]["goal_difference"] = (
        standings[home_team]["goals_for"] - standings[home_team]["goals_against"]
    )
    standings[away_team]["goal_difference"] = (
        standings[away_team]["goals_for"] - standings[away_team]["goals_against"]
    )

    if home_goals > away_goals:
        standings[home_team]["wins"] += 1
        standings[away_team]["losses"] += 1
        standings[home_team]["points"] += 3
    elif home_goals < away_goals:
        standings[away_team]["wins"] += 1
        standings[home_team]["losses"] += 1
        standings[away_team]["points"] += 3
    else:
        standings[home_team]["draws"] += 1
        standings[away_team]["draws"] += 1
        standings[home_team]["points"] += 1
        standings[away_team]["points"] += 1


def _sample_scoreline(
    model: PoissonGoalsModel,
    home_team: str,
    away_team: str,
    neutral: bool,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Sample one scoreline from the Poisson scoreline distribution."""

    expected_home_goals, expected_away_goals = model.expected_goals(
        home_team=home_team,
        away_team=away_team,
        neutral=neutral,
    )
    scorelines = model.scoreline_probabilities(
        expected_home_goals=expected_home_goals,
        expected_away_goals=expected_away_goals,
    )

    sampled_index = int(
        rng.choice(
            len(scorelines),
            p=scorelines["probability"].to_numpy(),
        )
    )
    sampled_row = scorelines.iloc[sampled_index]

    return int(sampled_row["home_goals"]), int(sampled_row["away_goals"])


def _rank_group(standings: dict[str, dict[str, int]]) -> pd.DataFrame:
    """Rank one group by points, goal difference, goals for, then team name."""

    rows = []

    for team, stats in standings.items():
        row = {"team": team}
        row.update(stats)
        rows.append(row)

    table = pd.DataFrame(rows)
    table = table.sort_values(
        ["points", "goal_difference", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)
    table["rank"] = table.index + 1

    return table


def simulate_group_stage(
    results: pd.DataFrame,
    fixtures: pd.DataFrame,
    n_simulations: int = 1_000,
    qualifiers_per_group: int = 2,
    seed: int = 42,
) -> GroupStageSimulationResult:
    """Run Monte Carlo group-stage simulations from a fixture table."""

    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    if qualifiers_per_group <= 0:
        raise ValueError("qualifiers_per_group must be positive.")

    model = PoissonGoalsModel()
    model.fit(results)

    rng = np.random.default_rng(seed)
    all_standings: list[pd.DataFrame] = []

    for simulation_id in range(1, n_simulations + 1):
        for group_name, group_fixtures in fixtures.groupby("group", sort=True):
            standings = _initial_standings(group_fixtures)

            for fixture in group_fixtures.itertuples(index=False):
                home_goals, away_goals = _sample_scoreline(
                    model=model,
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    neutral=bool(fixture.neutral),
                    rng=rng,
                )
                _apply_match_result(
                    standings=standings,
                    home_team=fixture.home_team,
                    away_team=fixture.away_team,
                    home_goals=home_goals,
                    away_goals=away_goals,
                )

            ranked = _rank_group(standings)
            ranked["group"] = str(group_name)
            ranked["simulation_id"] = simulation_id
            ranked["advanced"] = ranked["rank"] <= qualifiers_per_group
            all_standings.append(ranked)

    standings_table = pd.concat(all_standings, ignore_index=True)

    summary = (
        standings_table.groupby(["group", "team"])
        .agg(
            simulations=("simulation_id", "nunique"),
            advance_probability=("advanced", "mean"),
            avg_points=("points", "mean"),
            avg_goal_difference=("goal_difference", "mean"),
            avg_goals_for=("goals_for", "mean"),
        )
        .reset_index()
        .sort_values(
            ["group", "advance_probability", "avg_points", "avg_goal_difference"],
            ascending=[True, False, False, False],
        )
        .reset_index(drop=True)
    )

    return GroupStageSimulationResult(summary=summary, standings=standings_table)


def save_group_stage_simulation(
    results_path: str | Path,
    fixtures_path: str | Path,
    output_path: str | Path,
    n_simulations: int = 1_000,
    qualifiers_per_group: int = 2,
    seed: int = 42,
) -> Path:
    """Load inputs, run group-stage simulation, and save summary CSV."""

    results = load_historical_results(results_path)
    fixtures = load_group_fixtures(fixtures_path)

    simulation_result = simulate_group_stage(
        results=results,
        fixtures=fixtures,
        n_simulations=n_simulations,
        qualifiers_per_group=qualifiers_per_group,
        seed=seed,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    simulation_result.summary.to_csv(destination, index=False)

    return destination
