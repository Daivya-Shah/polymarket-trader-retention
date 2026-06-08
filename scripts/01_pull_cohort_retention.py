#!/usr/bin/env python3
"""Pull cohort retention matrix from Dune and write data/raw/cohort_retention.csv."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_settings, require_cohort_query_id  # noqa: E402
from data_access import run_dune_query  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

OUTPUT_PATH = _ROOT / "data" / "raw" / "cohort_retention.csv"
EXPECTED_COLUMNS = ("cohort_month", "months_since", "active_users")


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
    out["months_since"] = pd.to_numeric(out["months_since"], errors="raise").astype(
        "int64"
    )
    out["active_users"] = pd.to_numeric(out["active_users"], errors="raise").astype(
        "int64"
    )
    return out.sort_values(["cohort_month", "months_since"]).reset_index(drop=True)


def print_validation_summary(df: pd.DataFrame) -> None:
    """Print row counts, cohort span, and cohort sizes at months_since = 0."""
    print("=" * 60)
    print("Cohort retention — validation summary")
    print("=" * 60)

    print(f"total rows: {len(df)}")

    cohorts = df["cohort_month"].dt.to_period("M")
    print(f"distinct cohort_month values: {cohorts.nunique()}")
    print(f"cohort_month range: {cohorts.min()} .. {cohorts.max()}")

    print(f"months_since range: {df['months_since'].min()} .. {df['months_since'].max()}")

    negative = df[df["months_since"] < 0]
    if negative.empty:
        print("negative months_since: none (OK)")
    else:
        print(f"negative months_since: {len(negative)} rows — BUG")
        print(negative.head(10).to_string(index=False))

    sizes = (
        df.loc[df["months_since"] == 0, ["cohort_month", "active_users"]]
        .assign(cohort_month=lambda x: x["cohort_month"].dt.to_period("M").astype(str))
        .rename(columns={"active_users": "cohort_size"})
        .sort_values("cohort_month")
    )
    print("\nCohort sizes (months_since = 0):")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(sizes.to_string(index=False))


def main() -> int:
    _configure_stdout()
    settings = load_settings()
    query_id = require_cohort_query_id(settings)

    print(f"Pulling cohort retention (query_id={query_id}) …")
    df = _normalize(run_dune_query(query_id, settings=settings))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_PATH}")

    print_validation_summary(df)
    return 0 if df[df["months_since"] < 0].empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
