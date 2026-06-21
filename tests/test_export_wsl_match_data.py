from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from scripts.export_wsl_match_data import export_match_data, safe_summary, validate_export_columns


def _match_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_date": pd.to_datetime(["2025-09-12", "2025-09-14"]),
            "round_label": ["R2", "R2"],
            "home_team": ["Arsenal", "Chelsea"],
            "away_team": ["Chelsea", "Arsenal"],
            "home_xg": [1.4, 0.8],
            "away_xg": [0.9, 1.1],
            "home_np_xg": [1.2, 0.8],
            "away_np_xg": [0.7, 1.1],
            "home_goals": [2, 0],
            "away_goals": [1, 1],
        }
    )


def test_required_column_validation_passes() -> None:
    validate_export_columns(_match_data())


def test_required_column_validation_fails_clearly() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_export_columns(_match_data().drop(columns=["home_np_xg"]))


def test_output_directory_creation_and_csv_writing(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "wsl_match_data.csv"

    result = export_match_data(output, df=_match_data())

    assert result == output
    assert output.exists()
    exported = pd.read_csv(output)
    assert len(exported) == 2
    assert set(["home_xg", "away_xg"]).issubset(exported.columns)


def test_export_uses_mocked_fetch_without_live_supabase(tmp_path: Path) -> None:
    output = tmp_path / "wsl_match_data.csv"

    with patch("scripts.export_wsl_match_data.fetch_match_data", return_value=_match_data()) as fetch:
        export_match_data(output)

    fetch.assert_called_once()
    assert output.exists()


def test_safe_summary_contains_only_aggregate_fields(tmp_path: Path) -> None:
    output = tmp_path / "wsl_match_data.csv"
    summary = safe_summary(_match_data(), output)

    assert summary == {
        "row_count": 2,
        "min_match_date": "2025-09-12",
        "max_match_date": "2025-09-14",
        "round_labels": ["R2"],
        "output_path": str(output),
    }

