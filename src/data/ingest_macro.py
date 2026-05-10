"""Pull macro / risk-free / regime series from FRED via pandas-datareader.

No API key required for the public CSV endpoint.

Output: data/processed/macro.parquet
Long format: (date, series, value)
"""
from __future__ import annotations

import pandas as pd
from pandas_datareader import data as pdr

from src.utils.config import load_config
from src.utils.io import processed_dir
from src.utils.logging_utils import configure_logging, get_logger


log = get_logger(__name__)


def normalize_fred_frame(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert FRED's wide-format frame to long (date, series, value)."""
    reset = wide.reset_index()
    date_col = reset.columns[0]
    long = reset.melt(id_vars=date_col, var_name="series", value_name="value")
    long = long.rename(columns={date_col: "date"})
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long.dropna(subset=["value"]).reset_index(drop=True)


def main() -> None:
    configure_logging()
    cfg = load_config("data")
    series = cfg["macro"]["series"]
    start = cfg["start_date"]
    end = cfg["end_date"]

    log.info("Pulling FRED series: %s", series)
    wide = pdr.DataReader(series, "fred", start=start, end=end)
    long = normalize_fred_frame(wide)

    out = processed_dir() / "macro.parquet"
    long.to_parquet(out, index=False)
    log.info("Wrote %d rows for %d series to %s", len(long), long["series"].nunique(), out)


if __name__ == "__main__":
    main()
