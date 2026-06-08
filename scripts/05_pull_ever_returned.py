#!/usr/bin/env python3
"""Pull ever-returned cohort stats from Dune -> data/raw/ever_returned.csv."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_settings, require_ever_returned_query_id  # noqa: E402
from data_access import fetch_dune_query  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

OUTPUT_PATH = _ROOT / "data" / "raw" / "ever_returned.csv"
EXPECTED_COLUMNS = (
    "cohort_month",
    "cohort_size",
    "ever_returned",
    "ever_returned_pct",
)


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (LookupError, OSError):
            pass


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dune result missing columns: {missing}")

    out = df[list(EXPECTED_COLUMNS)].copy()
    out["cohort_month"] = pd.to_datetime(out["cohort_month"], utc=True)
    out["cohort_size"] = pd.to_numeric(out["cohort_size"], errors="raise").astype("int64")
    out["ever_returned"] = pd.to_numeric(out["ever_returned"], errors="raise").astype(
        "int64"
    )
    out["ever_returned_pct"] = pd.to_numeric(out["ever_returned_pct"], errors="raise")
    return out.sort_values("cohort_month").reset_index(drop=True)


def print_validation_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Ever returned — validation summary")
    print("=" * 60)
    print(f"total rows (cohort months): {len(df)}")
    labels = df["cohort_month"].dt.strftime("%Y-%m")
    print(f"cohort_month range: {labels.min()} .. {labels.max()}")
    print(f"cohort_size range: {df['cohort_size'].min():,} .. {df['cohort_size'].max():,}")
    print(
        f"ever_returned_pct range: {df['ever_returned_pct'].min():.1f}%"
        f" .. {df['ever_returned_pct'].max():.1f}%"
    )
    pooled = 100.0 * df["ever_returned"].sum() / df["cohort_size"].sum()
    print(f"pooled ever-returned (all rows): {pooled:.1f}%")
    print(f"median per-cohort ever-returned_pct: {df['ever_returned_pct'].median():.1f}%")


def main() -> int:
    _configure_stdout()
    settings = load_settings()
    query_id = require_ever_returned_query_id(settings)

    print(f"Pulling ever-returned stats (query_id={query_id}) …")
    df = _normalize(
        fetch_dune_query(query_id, settings=settings, prefer_cached=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_PATH}")
    print_validation_summary(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
