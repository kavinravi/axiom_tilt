"""Tests for src.data.ingest_macro."""
import pandas as pd

from src.data.ingest_macro import normalize_fred_frame


def test_normalize_fred_frame_long_format():
    idx = pd.date_range("2024-01-01", periods=3, freq="D")
    wide = pd.DataFrame(
        {"DGS3MO": [5.1, 5.2, 5.3], "VIXCLS": [13.0, 13.5, 14.0]},
        index=idx,
    )
    long = normalize_fred_frame(wide)
    assert {"date", "series", "value"}.issubset(long.columns)
    assert len(long) == 6
    assert set(long["series"].unique()) == {"DGS3MO", "VIXCLS"}
