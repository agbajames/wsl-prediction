from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.normalise_shadow_fixtures import main, normalise_shadow_fixture_frame


def test_normalises_common_raw_fixture_columns() -> None:
    raw = pd.DataFrame(
        {
            "date": ["2099-01-01", "2099-01-02"],
            "home": ["Sample City Women", "Sample Rovers Women"],
            "away": ["Sample United Women", "Sample Athletic Women"],
            "matchweek": ["R1", "R1"],
            "season": ["TEMPLATE-SEASON", "TEMPLATE-SEASON"],
            "competition": ["Sample WSL", "Sample WSL"],
            "venue": ["Sample Ground", "Sample Park"],
            "kickoff": ["12:30", "14:00"],
        }
    )

    fixtures = normalise_shadow_fixture_frame(raw)

    assert fixtures.columns.tolist() == [
        "fixture_id",
        "fixture_date",
        "round_label",
        "season",
        "home_team",
        "away_team",
        "competition",
        "venue",
        "kickoff_time",
        "source_notes",
    ]
    assert fixtures["fixture_date"].tolist() == ["2099-01-01", "2099-01-02"]
    assert fixtures["round_label"].tolist() == ["R1", "R1"]
    assert fixtures["fixture_id"].tolist() == [
        "2099-01-01_sample-city-women_vs_sample-united-women",
        "2099-01-02_sample-rovers-women_vs_sample-athletic-women",
    ]


def test_normalisation_accepts_alternate_team_and_date_names() -> None:
    raw = pd.DataFrame(
        {
            "match_date": ["2099-02-01"],
            "home_team_name": ["Sample Home Women"],
            "away_team_name": ["Sample Away Women"],
            "fixture_id": ["sample-fixture-1"],
        }
    )

    fixtures = normalise_shadow_fixture_frame(raw)

    assert fixtures.loc[0, "fixture_id"] == "sample-fixture-1"
    assert fixtures.loc[0, "home_team"] == "Sample Home Women"
    assert fixtures.loc[0, "away_team"] == "Sample Away Women"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (
            pd.DataFrame({"date": ["2099-01-01"], "away": ["Sample Away Women"]}),
            "home team",
        ),
        (
            pd.DataFrame({"date": ["not-a-date"], "home": ["Sample A"], "away": ["Sample B"]}),
            "Fixture date must be parseable",
        ),
        (
            pd.DataFrame({"date": ["2099-01-01"], "home": ["Sample A"], "away": ["Sample A"]}),
            "cannot be identical",
        ),
        (
            pd.DataFrame(
                {
                    "date": ["2099-01-01", "2099-01-01"],
                    "home": ["Sample A", "Sample A"],
                    "away": ["Sample B", "Sample B"],
                }
            ),
            "Duplicate fixture rows",
        ),
        (
            pd.DataFrame(
                {
                    "fixture_id": ["sample-1", "sample-1"],
                    "date": ["2099-01-01", "2099-01-02"],
                    "home": ["Sample A", "Sample C"],
                    "away": ["Sample B", "Sample D"],
                }
            ),
            "Duplicate fixture_id",
        ),
    ],
)
def test_normalisation_rejects_invalid_fixture_inputs(raw: pd.DataFrame, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalise_shadow_fixture_frame(raw)


def test_normalise_shadow_fixtures_cli_writes_canonical_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "normalised.csv"
    raw_path.write_text(
        "date,home,away,matchweek\n"
        "2099-01-01,Sample City Women,Sample United Women,R1\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/normalise_shadow_fixtures.py",
            "--input",
            str(raw_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    assert "Wrote 1 normalised fixture row(s)" in capsys.readouterr().out
    output = pd.read_csv(output_path)
    assert output.loc[0, "home_team"] == "Sample City Women"
    assert output.loc[0, "round_label"] == "R1"
