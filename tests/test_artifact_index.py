from __future__ import annotations

from pathlib import Path

import pandas as pd

from wc_forecast.reports.artifact_index import build_artifact_index, save_artifact_index


def test_build_artifact_index_marks_existing_and_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "existing.csv"
    missing = tmp_path / "missing.csv"

    existing.write_text("hello")

    result = build_artifact_index(paths=[existing, missing])

    assert list(result["exists"]) == [True, False]
    assert result.loc[0, "size_bytes"] == 5
    assert pd.isna(result.loc[1, "size_bytes"])


def test_save_artifact_index_writes_csv(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.md"
    output = tmp_path / "index.csv"

    artifact.write_text("# Report")

    result = save_artifact_index(
        output_path=output,
        paths=[artifact],
    )

    assert output.exists()
    assert len(result) == 1
    assert bool(result.loc[0, "exists"])
