#!/usr/bin/env python3
"""Pull churn training features from Dune -> data/raw/churn_features.csv."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_settings, require_churn_features_query_id  # noqa: E402
from data_access import fetch_dune_query  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)

OUTPUT_PATH = _ROOT / "data" / "raw" / "churn_features.csv"
EXPECTED_COLUMNS = (
    "wallet",
    "cohort_month",
    "first_category",
    "n_trades_w1",
    "active_days_w1",
    "n_markets_w1",
    "n_categories_w1",
    "multi_category_flag",
    "avg_usd",
    "max_usd",
    "total_usd",
    "extreme_price_share",
    "median_hour",
    "hrs_to_2nd",
    "active_m2",
    "active_m3",
    "active_m6b",
    "active_m6",
)
LABEL_COLUMNS = ("active_m2", "active_m3", "active_m6b", "active_m6")
VALID_CATEGORIES = frozenset({"politics", "sports", "crypto", "other"})


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
    out["wallet"] = out["wallet"].astype(str)
    out["cohort_month"] = pd.to_datetime(out["cohort_month"], utc=True)
    out["first_category"] = out["first_category"].astype(str)

    int_cols = (
        "n_trades_w1",
        "active_days_w1",
        "n_markets_w1",
        "n_categories_w1",
        "multi_category_flag",
        *LABEL_COLUMNS,
    )
    for col in int_cols:
        out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")

    float_cols = ("avg_usd", "max_usd", "total_usd", "extreme_price_share", "median_hour")
    for col in float_cols:
        out[col] = pd.to_numeric(out[col], errors="raise")

    out["hrs_to_2nd"] = pd.to_numeric(out["hrs_to_2nd"], errors="coerce")

    dupes = out["wallet"].duplicated().sum()
    if dupes:
        raise ValueError(f"Expected one row per wallet; found {dupes} duplicate wallets")

    bad_cats = set(out["first_category"].unique()) - VALID_CATEGORIES
    if bad_cats:
        raise ValueError(f"Unexpected first_category values: {sorted(bad_cats)}")

    for col in LABEL_COLUMNS:
        if not out[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be 0 or 1")

    if (out["n_trades_w1"] < 1).any():
        raise ValueError("n_trades_w1 must be >= 1 for every wallet")

    return out.sort_values(["cohort_month", "wallet"]).reset_index(drop=True)


def print_validation_summary(df: pd.DataFrame) -> None:
    labels = df["cohort_month"].dt.strftime("%Y-%m")
    single_trade = df["hrs_to_2nd"].isna().sum()

    print("=" * 60)
    print("Churn features — validation summary")
    print("=" * 60)
    print(f"total rows (wallets): {len(df):,}")
    print(f"cohort_month range: {labels.min()} .. {labels.max()}")
    print("label positive rates:")
    for col in LABEL_COLUMNS:
        print(f"  {col}: {100.0 * df[col].mean():.1f}%")
    print(f"wallets with no second trade (hrs_to_2nd null): {single_trade:,}")
    print(
        f"n_trades_w1 range: {df['n_trades_w1'].min()} .. {df['n_trades_w1'].max()}"
    )
    print("first_category counts:")
    for cat, n in df["first_category"].value_counts().items():
        print(f"  {cat}: {n:,}")


def main() -> int:
    _configure_stdout()
    settings = load_settings()
    query_id = require_churn_features_query_id(settings)

    print(f"Pulling churn features (query_id={query_id}) …")
    df = _normalize(
        fetch_dune_query(query_id, settings=settings, prefer_cached=True)
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {len(df):,} rows to {OUTPUT_PATH}")
    print_validation_summary(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
