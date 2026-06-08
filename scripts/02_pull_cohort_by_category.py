#!/usr/bin/env python3
"""Pull category-segmented cohort retention from Dune."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_settings, require_cohort_cat_query_id  # noqa: E402
from data_access import fetch_dune_query  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

OUTPUT_PATH = _ROOT / "data" / "raw" / "cohort_retention_by_category.csv"
EXPECTED_COLUMNS = (
    "first_category",
    "cohort_month",
    "months_since",
    "active_users",
)
KEY_COHORTS = ("2024-10", "2024-11", "2026-02", "2026-03")


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (LookupError, OSError):
            pass


def _cohort_label(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.strftime("%Y-%m")


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
    out["first_category"] = out["first_category"].astype(str)
    return out.sort_values(
        ["first_category", "cohort_month", "months_since"]
    ).reset_index(drop=True)


def print_validation_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("Cohort retention by category — validation summary")
    print("=" * 60)

    print(f"total rows: {len(df)}")
    print(f"distinct categories: {sorted(df['first_category'].unique())}")

    labels = _cohort_label(df["cohort_month"])
    print(f"distinct cohort_month values: {labels.nunique()}")
    print(f"cohort_month range: {labels.min()} .. {labels.max()}")
    print(f"months_since range: {df['months_since'].min()} .. {df['months_since'].max()}")

    negative = df[df["months_since"] < 0]
    if negative.empty:
        print("negative months_since: none (OK)")
    else:
        print(f"negative months_since: {len(negative)} rows — BUG")
        print(negative.head(10).to_string(index=False))

    m0 = df.loc[df["months_since"] == 0].copy()
    m0["cohort_label"] = _cohort_label(m0["cohort_month"])
    by_cat = (
        m0.groupby("first_category", as_index=False)["active_users"]
        .sum()
        .rename(columns={"active_users": "acquired_users"})
        .sort_values("acquired_users", ascending=False)
    )
    total_users = int(by_cat["acquired_users"].sum())
    by_cat["pct_of_users"] = (100.0 * by_cat["acquired_users"] / total_users).round(1)
    print(f"\nAcquisition mix by first_category (months_since=0, all cohorts):")
    print(f"total distinct acquired users: {total_users:,}")
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(by_cat.to_string(index=False))

    print("\nKey cohort sizes by first_category (months_since = 0):")
    key = m0[m0["cohort_label"].isin(KEY_COHORTS)].copy()
    key = key.sort_values(["cohort_label", "active_users"], ascending=[True, False])
    pivot = key.pivot_table(
        index="cohort_label",
        columns="first_category",
        values="active_users",
        fill_value=0,
        aggfunc="sum",
    )
    with pd.option_context("display.width", 140):
        print(pivot.to_string())


def main() -> int:
    _configure_stdout()
    settings = load_settings()
    query_id = require_cohort_cat_query_id(settings)

    print(f"Pulling category cohort retention (query_id={query_id}) …")
    df = _normalize(
        fetch_dune_query(query_id, settings=settings, prefer_cached=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df)} rows to {OUTPUT_PATH}")

    print_validation_summary(df)
    return 0 if df[df["months_since"] < 0].empty else 1


if __name__ == "__main__":
    raise SystemExit(main())
