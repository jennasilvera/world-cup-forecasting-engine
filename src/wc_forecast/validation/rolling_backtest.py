from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from wc_forecast.features.build_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_table,
)
from wc_forecast.models.classifier import (
    OUTCOME_ORDER,
    train_logistic_regression,
)

DEFAULT_ROLLING_CUTOFF_DATES = [
    "2018-01-01",
    "2019-01-01",
    "2020-01-01",
    "2021-01-01",
    "2022-01-01",
    "2023-01-01",
    "2024-01-01",
    "2025-01-01",
    "2026-01-01",
]

DEFAULT_MODEL_TYPES = [
    "logistic",
    "gradient_boosting",
    "random_forest",
]

LOG_LOSS_LABEL_ORDER = sorted(OUTCOME_ORDER)


ROLLING_BACKTEST_COLUMNS = [
    "model_type",
    "cutoff_date",
    "evaluation_window_days",
    "train_rows",
    "test_rows",
    "accuracy",
    "log_loss",
    "multiclass_brier_score",
]


def run_rolling_backtest(
    features: pd.DataFrame,
    cutoff_dates: list[str] | None = None,
    model_types: list[str] | None = None,
    evaluation_window_days: int = 365,
    sample_weight_half_life_days: float | None = None,
    logistic_c: float = 1.0,
) -> pd.DataFrame:
    """Run rolling-origin backtests across multiple historical cutoffs."""

    validate_feature_table(features)

    if evaluation_window_days <= 0:
        raise ValueError("evaluation_window_days must be positive.")

    active_cutoff_dates = cutoff_dates or DEFAULT_ROLLING_CUTOFF_DATES
    active_model_types = model_types or DEFAULT_MODEL_TYPES

    dated_features = features.copy()
    dated_features["date"] = pd.to_datetime(dated_features["date"], errors="raise")

    rows: list[dict[str, object]] = []

    for model_type in active_model_types:
        for cutoff_date in active_cutoff_dates:
            cutoff = pd.to_datetime(cutoff_date, errors="raise")
            evaluation_end = cutoff + pd.Timedelta(days=evaluation_window_days)

            train = dated_features[dated_features["date"] < cutoff].copy()
            test = dated_features[
                (dated_features["date"] >= cutoff)
                & (dated_features["date"] < evaluation_end)
            ].copy()

            if train.empty or test.empty:
                continue

            if train[TARGET_COLUMN].nunique() < len(OUTCOME_ORDER):
                continue

            model = train_logistic_regression(
                train,
                sample_weight_half_life_days=sample_weight_half_life_days,
                model_type=model_type,
                logistic_c=logistic_c,
            )

            probabilities = _predict_aligned_probabilities(model, test)
            predicted_outcomes = probabilities.idxmax(axis=1)

            rows.append(
                {
                    "model_type": model_type,
                    "cutoff_date": cutoff.date().isoformat(),
                    "evaluation_window_days": evaluation_window_days,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "accuracy": accuracy_score(
                        test[TARGET_COLUMN],
                        predicted_outcomes,
                    ),
                    "log_loss": log_loss(
                        test[TARGET_COLUMN],
                        probabilities[LOG_LOSS_LABEL_ORDER],
                        labels=LOG_LOSS_LABEL_ORDER,
                    ),
                    "multiclass_brier_score": _multiclass_brier_score(
                        actual_outcomes=test[TARGET_COLUMN],
                        predicted_probabilities=probabilities,
                    ),
                }
            )

    result = pd.DataFrame(rows, columns=ROLLING_BACKTEST_COLUMNS)

    if result.empty:
        raise ValueError("Rolling backtest produced no valid folds.")

    return result


def save_rolling_backtest(
    features_path: str | Path,
    output_path: str | Path,
    cutoff_dates: list[str] | None = None,
    model_types: list[str] | None = None,
    evaluation_window_days: int = 365,
    sample_weight_half_life_days: float | None = None,
    logistic_c: float = 1.0,
) -> pd.DataFrame:
    """Load features, run rolling backtest, and save metrics CSV."""

    features = pd.read_csv(features_path)
    result = run_rolling_backtest(
        features=features,
        cutoff_dates=cutoff_dates,
        model_types=model_types,
        evaluation_window_days=evaluation_window_days,
        sample_weight_half_life_days=sample_weight_half_life_days,
        logistic_c=logistic_c,
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)

    return result


def summarize_rolling_backtest(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize rolling backtest metrics by model type."""

    if results.empty:
        raise ValueError("Cannot summarize empty rolling backtest results.")

    summary = (
        results.groupby("model_type", as_index=False)
        .agg(
            folds=("cutoff_date", "count"),
            total_test_rows=("test_rows", "sum"),
            mean_accuracy=("accuracy", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_brier=("multiclass_brier_score", "mean"),
        )
        .sort_values(["mean_log_loss", "mean_brier"])
        .reset_index(drop=True)
    )

    return summary


def _predict_aligned_probabilities(model: object, test: pd.DataFrame) -> pd.DataFrame:
    """Return predicted probabilities aligned to canonical outcome order."""

    raw_probabilities = model.predict_proba(test[FEATURE_COLUMNS])
    classifier = model.named_steps["classifier"]

    probabilities = pd.DataFrame(
        0.0,
        index=test.index,
        columns=OUTCOME_ORDER,
    )

    for class_index, class_name in enumerate(classifier.classes_):
        probabilities[str(class_name)] = raw_probabilities[:, class_index]

    return probabilities


def _multiclass_brier_score(
    actual_outcomes: pd.Series,
    predicted_probabilities: pd.DataFrame,
) -> float:
    """Calculate multiclass Brier score using canonical outcome order."""

    actual = pd.get_dummies(
        pd.Categorical(actual_outcomes, categories=OUTCOME_ORDER)
    ).astype(float)

    actual.columns = OUTCOME_ORDER

    squared_errors = (
        predicted_probabilities[OUTCOME_ORDER].reset_index(drop=True)
        - actual.reset_index(drop=True)
    ) ** 2

    return float(squared_errors.sum(axis=1).mean())
