"""Tests for src/data/ingest_edgar_xbrl.py — HTTP and parsing logic, no live network."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.ingest_edgar_xbrl import (
    CONCEPT_MAP,
    _fetch_company_facts,
    parse_all_raw,
    parse_company_facts,
    pull_xbrl_for_universe,
)


def _sample_facts_json() -> dict:
    """Build a minimal SEC companyfacts JSON for testing."""
    return {
        "cik": 320193,
        "entityName": "APPLE INC",
        "facts": {
            "us-gaap": {
                "Assets": {
                    "label": "Assets",
                    "description": "...",
                    "units": {
                        "USD": [
                            {
                                "start": "2019-09-29",
                                "end": "2020-09-26",
                                "val": 323888000000,
                                "accn": "0000320193-20-000096",
                                "fy": 2020,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2020-10-30",
                            },
                            {
                                "start": "2020-09-27",
                                "end": "2021-09-25",
                                "val": 351002000000,
                                "accn": "0000320193-21-000105",
                                "fy": 2021,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2021-10-29",
                            },
                        ]
                    },
                },
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2019-09-29",
                                "end": "2020-09-26",
                                "val": 57411000000,
                                "accn": "0000320193-20-000096",
                                "fy": 2020,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2020-10-30",
                            }
                        ]
                    }
                },
                # Fallback tag for revenue — older filers may not have Revenues
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {
                                "start": "2014-09-28",
                                "end": "2015-09-26",
                                "val": 233715000000,
                                "accn": "0000320193-15-000119",
                                "fy": 2015,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2015-10-28",
                            }
                        ]
                    }
                },
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {
                    "units": {
                        "shares": [
                            {
                                "end": "2020-10-16",
                                "val": 17001802000,
                                "accn": "0000320193-20-000096",
                                "fy": 2020,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2020-10-30",
                            }
                        ]
                    }
                }
            },
        },
    }


def test_parse_company_facts_basic():
    df = parse_company_facts(_sample_facts_json(), cik=320193)

    assert not df.empty
    # Two Assets entries + one NetIncomeLoss + one revenue (via SalesRevenueNet fallback) + 1 shares
    assert len(df) == 5

    concepts = set(df["concept"].unique())
    assert "assets_total" in concepts
    assert "net_income" in concepts
    assert "revenue" in concepts
    assert "shares_outstanding" in concepts


def test_parse_company_facts_uses_fallback_tag():
    """When primary tag is missing, the parser should use the next candidate."""
    json_data = _sample_facts_json()
    # The sample has SalesRevenueNet but not Revenues; verify it falls through.
    df = parse_company_facts(json_data, cik=320193)
    rev = df[df["concept"] == "revenue"]
    assert len(rev) == 1
    assert rev.iloc[0]["xbrl_tag"] == "SalesRevenueNet"
    assert rev.iloc[0]["value"] == 233715000000


def test_parse_company_facts_filed_is_datetime():
    """filed is the PIT date — must be parsed as datetime, never string."""
    df = parse_company_facts(_sample_facts_json(), cik=320193)
    assert pd.api.types.is_datetime64_any_dtype(df["filed"])
    assert df["filed"].min() == pd.Timestamp("2015-10-28")


def test_parse_company_facts_empty_facts_returns_empty_df():
    df = parse_company_facts({"facts": {}}, cik=999999)
    assert df.empty


def test_parse_company_facts_keeps_dei_shares_taxonomy():
    """shares_outstanding can come from dei taxonomy when us-gaap version is absent."""
    df = parse_company_facts(_sample_facts_json(), cik=320193)
    shares = df[df["concept"] == "shares_outstanding"]
    assert len(shares) == 1
    assert shares.iloc[0]["xbrl_taxonomy"] == "dei"
    assert shares.iloc[0]["units"] == "shares"


def test_parse_company_facts_first_candidate_wins_no_double_count():
    """If both primary and fallback tags exist, only the primary should appear."""
    json_data = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            {"end": "2020-12-31", "val": 100, "filed": "2021-02-15",
                             "accn": "x", "fy": 2020, "fp": "FY", "form": "10-K"}
                        ]
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            {"end": "2020-12-31", "val": 999, "filed": "2021-02-15",
                             "accn": "x", "fy": 2020, "fp": "FY", "form": "10-K"}
                        ]
                    }
                },
            }
        }
    }
    df = parse_company_facts(json_data, cik=1)
    rev = df[df["concept"] == "revenue"]
    assert len(rev) == 1
    assert rev.iloc[0]["xbrl_tag"] == "Revenues"
    assert rev.iloc[0]["value"] == 100  # primary won, not the fallback


def test_concept_map_has_no_duplicate_tags_within_a_concept():
    """Sanity check on the static map."""
    for concept, candidates in CONCEPT_MAP.items():
        seen = set()
        for tax, tag in candidates:
            assert (tax, tag) not in seen, f"duplicate in {concept}: {(tax, tag)}"
            seen.add((tax, tag))


def test_parse_all_raw_writes_parquet(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "CIK0000320193.json").write_text(json.dumps(_sample_facts_json()))

    out = tmp_path / "facts.parquet"
    n_rows = parse_all_raw(raw_dir, out)

    assert n_rows == 5
    assert out.exists()
    df = pd.read_parquet(out)
    assert len(df) == 5
    assert df["cik"].iloc[0] == 320193


def test_parse_all_raw_no_files_returns_zero(tmp_path: Path):
    out = tmp_path / "facts.parquet"
    n = parse_all_raw(tmp_path, out)
    assert n == 0
    assert not out.exists()


def test_fetch_company_facts_calls_token_bucket_correctly():
    """Regression: _fetch_company_facts must use the real TokenBucket API.

    Caught a real bug where this called `bucket.take()` but the method is
    `bucket.acquire()`. Tests that patch _fetch_company_facts itself can't see
    this — we need to exercise the real function with a mocked session/bucket.
    """
    bucket = MagicMock()
    session = MagicMock()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"facts": {}}
    session.get.return_value = response

    _fetch_company_facts(1234, bucket, {"User-Agent": "x"}, session)

    bucket.acquire.assert_called_once()
    session.get.assert_called_once()
    assert "CIK0000001234.json" in session.get.call_args[0][0]


def test_pull_xbrl_for_universe_handles_404(tmp_path: Path):
    """A CIK with no XBRL returns 404 from SEC; we mark it done so we don't retry."""
    universe_path = tmp_path / "universe_ids.parquet"
    pd.DataFrame({"cik": [11111, 22222], "permno": [1, 2], "gvkey": ["A", "B"]}).to_parquet(
        universe_path, index=False
    )

    raw_dir = tmp_path / "raw"
    state_path = tmp_path / "state.txt"

    # Mock _fetch_company_facts: first call returns None (404), second returns data
    with patch("src.data.ingest_edgar_xbrl._fetch_company_facts") as mock_fetch:
        mock_fetch.side_effect = [None, _sample_facts_json()]
        no_xbrl_count = pull_xbrl_for_universe(
            universe_ids_path=universe_path,
            raw_output_dir=raw_dir,
            state_path=state_path,
            user_agent="Test test@example.com",
        )

    assert no_xbrl_count == 1
    assert (raw_dir / "CIK0000022222.json").exists()
    assert not (raw_dir / "CIK0000011111.json").exists()  # 404 means no file written
    state_lines = state_path.read_text().splitlines()
    assert "11111" in state_lines and "22222" in state_lines


def test_pull_xbrl_for_universe_resumes_from_state(tmp_path: Path):
    """If state file already lists a CIK, we don't refetch it."""
    universe_path = tmp_path / "universe_ids.parquet"
    pd.DataFrame({"cik": [11111, 22222]}).to_parquet(universe_path, index=False)

    raw_dir = tmp_path / "raw"
    state_path = tmp_path / "state.txt"
    state_path.write_text("11111\n")  # pretend we already did 11111

    with patch("src.data.ingest_edgar_xbrl._fetch_company_facts") as mock_fetch:
        mock_fetch.return_value = _sample_facts_json()
        pull_xbrl_for_universe(
            universe_ids_path=universe_path,
            raw_output_dir=raw_dir,
            state_path=state_path,
            user_agent="Test test@example.com",
        )

    # Only one call should have happened (for CIK 22222)
    assert mock_fetch.call_count == 1
    assert mock_fetch.call_args[0][0] == 22222
