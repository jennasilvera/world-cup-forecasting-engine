from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wc_forecast.features.build_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    validate_feature_table,
)

OUTCOME_ORDER = ["home_win", "draw", "away_win"]
PREDICTION_COLUMNS = [f"prob_{outcome}" for outcome in OUTCOME_ORDER]
DEFAULT_RECENCY_HALF_LIFE_DAYS = 1_460.0
SUPPORTED_MODEL_TYPES = {"logistic", "gradient_boosting", "random_forest"}


@dataclass(frozen=True)
class BacktestResult:
    """Container for backtest predictions and summary metrics."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame


def _make_classifier(model_type: str) -> object:
    """Create a supported classification model."""

    if model_type == "logistic":
        return LogisticRegression(
            max_iter=1_000,
            random_state=42,
        )

    if model_type == "gradient_boosting":
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )

    raise ValueError(
        f"Unsupported model_type: {model_type}. "
        f"Expected one of: {sorted(SUPPORTED_MODEL_TYPES)}"
    )


def calculate_recency_sample_weights(
    features: pd.DataFrame,
    half_life_days: float = DEFAULT_RECENCY_HALF_LIFE_DAYS,
    reference_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Calculate exponential time-decay weights for historical matches."""

    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive.")

    dated_features = features.copy()
    dated_features["date"] = pd.to_datetime(dated_features["date"], errors="raise")

    if reference_date is None:
        reference_timestamp = dated_features["date"].max()
    else:
        reference_timestamp = pd.to_datetime(reference_date, errors="raise")

    age_days = (reference_timestamp - dated_features["date"]).dt.days.clip(lower=0)
    weights = 0.5 ** (age_days / half_life_days)

    if "tournament_weight" in dated_features.columns:
        tournament_weights = pd.to_numeric(
            dated_features["tournament_weight"],
            errors="coerce",
        ).fillna(1.0)
        weights = weights * tournament_weights.clip(lower=0.01)

    mean_weight = float(weights.mean())

    if mean_weight <= 0:
        raise ValueError("Sample weights have non-positive mean.")

    normalized_weights = weights / mean_weight

    return normalized_weights.astype(float)


def chronological_train_test_split(
    features: pd.DataFrame,
    test_fraction: float = 0.30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split features chronologically into train and test sets."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    sorted_features = features.copy()
    sorted_features["date"] = pd.to_datetime(sorted_features["date"], errors="raise")
    sorted_features = sorted_features.sort_values(
        ["date", "home_team", "away_team"]
    ).reset_index(drop=True)

    test_size = max(1, int(round(len(sorted_features) * test_fraction)))
    train_size = len(sorted_features) - test_size

    if train_size < len(OUTCOME_ORDER):
        raise ValueError(
            "Not enough training rows for a three-class football outcome model."
        )

    train = sorted_features.iloc[:train_size].copy()
    test = sorted_features.iloc[train_size:].copy()

    return train, test


def date_cutoff_train_test_split(
    features: pd.DataFrame,
    cutoff_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split features by date: train before cutoff, test on/after cutoff."""

    cutoff = pd.to_datetime(cutoff_date, errors="raise")

    sorted_features = features.copy()
    sorted_features["date"] = pd.to_datetime(sorted_features["date"], errors="raise")
    sorted_features = sorted_features.sort_values(
        ["date", "home_team", "away_team"]
    ).reset_index(drop=True)

    train = sorted_features[sorted_features["date"] < cutoff].copy()
    test = sorted_features[sorted_features["date"] >= cutoff].copy()

    if len(train) < len(OUTCOME_ORDER):
        raise ValueError(
            "Not enough training rows before cutoff for a three-class football "
            "outcome model."
        )

    if test.empty:
        raise ValueError("No test rows found on or after cutoff_date.")

    return train.reset_index(drop=True), test.reset_index(drop=True)


def _validate_training_outcomes(train_features: pd.DataFrame) -> None:
    """Ensure the training set contains all required outcome classes."""

    observed_outcomes = set(train_features[TARGET_COLUMN])
    missing_outcomes = sorted(set(OUTCOME_ORDER) - observed_outcomes)

    if missing_outcomes:
        raise ValueError(
            "Training data is missing required outcome classes: "
            f"{missing_outcomes}. Add more historical rows or adjust the split."
        )


def train_logistic_regression(
    train_features: pd.DataFrame,
    sample_weight_half_life_days: float | None = None,
    model_type: str = "logistic",
) -> Pipeline:
    """Train a supported three-class football outcome model."""

    validate_feature_table(train_features)
    _validate_training_outcomes(train_features)

    x_train = train_features[FEATURE_COLUMNS]
    y_train = train_features[TARGET_COLUMN]

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", _make_classifier(model_type)),
        ]
    )

    if sample_weight_half_life_days is None:
        model.fit(x_train, y_train)
    else:
        sample_weights = calculate_recency_sample_weights(
            train_features,
            half_life_days=sample_weight_half_life_days,
        )
        model.fit(
            x_train,
            y_train,
            classifier__sample_weight=sample_weights,
        )

    return model


def predict_probabilities(
    model: Pipeline,
    test_features: pd.DataFrame,
) -> pd.DataFrame:
    """Generate aligned three-class probability predictions."""

    validate_feature_table(test_features)

    x_test = test_features[FEATURE_COLUMNS]
    raw_probabilities = model.predict_proba(x_test)
    model_classes = list(model.classes_)

    probability_frame = pd.DataFrame(
        0.0,
        index=test_features.index,
        columns=PREDICTION_COLUMNS,
    )

    for class_index, class_name in enumerate(model_classes):
        probability_frame[f"prob_{class_name}"] = raw_probabilities[:, class_index]

    predicted_indices = probability_frame.to_numpy().argmax(axis=1)
    predicted_outcomes = [OUTCOME_ORDER[index] for index in predicted_indices]

    predictions = test_features[
        [
            "date",
            "home_team",
            "away_team",
            "tournament",
            "home_score",
            "away_score",
            TARGET_COLUMN,
        ]
    ].copy()

    predictions = predictions.rename(columns={TARGET_COLUMN: "actual_outcome"})
    predictions["predicted_outcome"] = predicted_outcomes

    for column in PREDICTION_COLUMNS:
        predictions[column] = probability_frame[column].values

    return predictions.reset_index(drop=True)


def multiclass_brier_score(predictions: pd.DataFrame) -> float:
    """Compute multiclass Brier score for ordered outcome probabilities."""

    probability_matrix = predictions[PREDICTION_COLUMNS].to_numpy()
    actuals = predictions["actual_outcome"].to_numpy()

    one_hot = np.zeros_like(probability_matrix)

    for row_index, actual_outcome in enumerate(actuals):
        outcome_index = OUTCOME_ORDER.index(str(actual_outcome))
        one_hot[row_index, outcome_index] = 1.0

    return float(np.mean(np.sum((probability_matrix - one_hot) ** 2, axis=1)))


def evaluate_predictions(
    predictions: pd.DataFrame,
    train_size: int,
    test_size: int,
) -> pd.DataFrame:
    """Evaluate probability forecasts with classification and calibration metrics."""

    y_true = predictions["actual_outcome"]
    y_pred = predictions["predicted_outcome"]
    log_loss_labels = sorted(OUTCOME_ORDER)
    log_loss_probability_columns = [f"prob_{label}" for label in log_loss_labels]
    probability_matrix = predictions[log_loss_probability_columns].to_numpy()

    metrics = [
        {"metric": "train_rows", "value": float(train_size)},
        {"metric": "test_rows", "value": float(test_size)},
        {"metric": "accuracy", "value": float(accuracy_score(y_true, y_pred))},
        {
            "metric": "log_loss",
            "value": float(log_loss(y_true, probability_matrix, labels=log_loss_labels)),
        },
        {
            "metric": "multiclass_brier_score",
            "value": multiclass_brier_score(predictions),
        },
    ]

    return pd.DataFrame(metrics)


def run_logistic_backtest(
    features: pd.DataFrame,
    test_fraction: float = 0.30,
    cutoff_date: str | None = None,
    sample_weight_half_life_days: float | None = None,
    model_type: str = "logistic",
) -> BacktestResult:
    """Run a chronological logistic-regression backtest."""

    validate_feature_table(features)

    if cutoff_date is None:
        train, test = chronological_train_test_split(
            features=features,
            test_fraction=test_fraction,
        )
    else:
        train, test = date_cutoff_train_test_split(
            features=features,
            cutoff_date=cutoff_date,
        )

    model = train_logistic_regression(
        train,
        sample_weight_half_life_days=sample_weight_half_life_days,
        model_type=model_type,
    )
    predictions = predict_probabilities(model, test)
    metrics = evaluate_predictions(
        predictions=predictions,
        train_size=len(train),
        test_size=len(test),
    )

    return BacktestResult(predictions=predictions, metrics=metrics)


def save_logistic_backtest(
    features_path: str | Path,
    predictions_output_path: str | Path,
    metrics_output_path: str | Path,
    test_fraction: float = 0.30,
    cutoff_date: str | None = None,
    sample_weight_half_life_days: float | None = None,
    model_type: str = "logistic",
) -> BacktestResult:
    """Load features, run backtest, and save predictions and metrics."""

    features = pd.read_csv(features_path)
    result = run_logistic_backtest(
        features=features,
        test_fraction=test_fraction,
        cutoff_date=cutoff_date,
        sample_weight_half_life_days=sample_weight_half_life_days,
        model_type=model_type,
    )

    predictions_destination = Path(predictions_output_path)
    metrics_destination = Path(metrics_output_path)

    predictions_destination.parent.mkdir(parents=True, exist_ok=True)
    metrics_destination.parent.mkdir(parents=True, exist_ok=True)

    result.predictions.to_csv(predictions_destination, index=False)
    result.metrics.to_csv(metrics_destination, index=False)

    return result
