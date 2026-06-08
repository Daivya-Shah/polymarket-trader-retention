"""
Retention analytics for Polymarket taker cohort matrices.

Our data supports two distinct retention notions:
- *Sequential* month-N retention from the cohort matrix (active in exactly month M / cohort size).
- *Ever returned* from sql/03_ever_returned.sql (active in >=2 distinct calendar months after
  first trade). Compare the latter to Polymarket's "~75% return" claim on similar terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

# Presentation order and colors (use everywhere charts reference categories).
CATEGORY_ORDER: Final[list[str]] = ["politics", "sports", "crypto", "other"]
CATEGORY_COLORS: Final[dict[str, str]] = {
    "politics": "#D1495B",
    "sports": "#3A6EA5",
    "crypto": "#E8A33D",
    "other": "#9AA0A6",
}

# Mature cohort window: each has >=12 observable months by 2026-06; excludes
# right-censored 2026 cohorts and thin early-2024 baselines.
MATURE_COHORT_START: Final[str] = "2024-05"
MATURE_COHORT_END: Final[str] = "2025-05"
ELECTION_COHORT_MONTH: Final[str] = "2024-10"
CLAIMED_RETURN_RATE: Final[float] = 0.75
MAX_MATURE_MONTHS_SINCE: Final[int] = 12
SEQUENTIAL_MILESTONES: Final[tuple[int, ...]] = (1, 3, 6)

# Ever-returned mature window for apples-to-apples vs the ~75% claim.
EVER_RETURNED_MATURE_START: Final[str] = "2024-01"
EVER_RETURNED_MATURE_END: Final[str] = "2025-12"


@dataclass(frozen=True)
class EverReturnedSummary:
    """Pooled and median ever-returned rates over mature cohort months."""

    pooled_rate: float
    median_rate: float
    cohort_start: str
    cohort_end: str
    total_cohort_size: int
    total_ever_returned: int
    n_cohorts: int


def load_cohort_retention(path: Path | str) -> pd.DataFrame:
    """Load overall cohort matrix CSV."""
    df = pd.read_csv(path)
    return _normalize_cohort_df(df, category_col=None)


def load_cohort_by_category(path: Path | str) -> pd.DataFrame:
    """Load category-segmented cohort matrix CSV."""
    df = pd.read_csv(path)
    return _normalize_cohort_df(df, category_col="first_category")


def load_ever_returned(path: Path | str) -> pd.DataFrame:
    """Load per-cohort ever-returned counts from sql/03_ever_returned.sql."""
    required = {
        "cohort_month",
        "cohort_size",
        "ever_returned",
        "ever_returned_pct",
    }
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    out["cohort_month"] = pd.to_datetime(out["cohort_month"], utc=True)
    out["cohort_label"] = out["cohort_month"].dt.strftime("%Y-%m")
    out["cohort_size"] = out["cohort_size"].astype(int)
    out["ever_returned"] = out["ever_returned"].astype(int)
    out["ever_returned_pct"] = pd.to_numeric(out["ever_returned_pct"]) / 100.0
    return out.sort_values("cohort_month").reset_index(drop=True)


def _normalize_cohort_df(
    df: pd.DataFrame, *, category_col: str | None
) -> pd.DataFrame:
    required = {"cohort_month", "months_since", "active_users"}
    if category_col:
        required.add(category_col)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy()
    out["cohort_month"] = pd.to_datetime(out["cohort_month"], utc=True)
    out["months_since"] = out["months_since"].astype(int)
    out["active_users"] = out["active_users"].astype(int)
    out["cohort_label"] = out["cohort_month"].dt.strftime("%Y-%m")
    if category_col:
        out[category_col] = out[category_col].astype(str)
    return out


def cohort_sizes(
    df: pd.DataFrame,
    *,
    category_col: str | None = None,
) -> pd.DataFrame:
    """Cohort sizes = active_users at months_since == 0."""
    m0 = df[df["months_since"] == 0].copy()
    group = ["cohort_label"] if category_col is None else [category_col, "cohort_label"]
    return (
        m0.groupby(group, as_index=False)["active_users"]
        .sum()
        .rename(columns={"active_users": "cohort_size"})
    )


def retention_rates_long(
    df: pd.DataFrame,
    *,
    category_col: str | None = None,
) -> pd.DataFrame:
    """
    Per-cohort sequential retention rates: active_users(m) / cohort_size.

    months_since=0 is 100% by construction.
    """
    group_keys = ["cohort_label"] if category_col is None else [category_col, "cohort_label"]
    sizes = cohort_sizes(df, category_col=category_col)
    merged = df.merge(sizes, on=group_keys, how="inner")
    merged["retention_rate"] = merged["active_users"] / merged["cohort_size"]
    return merged


def _filter_cohort_range(
    df: pd.DataFrame,
    start: str,
    end: str,
    *,
    category_col: str | None = None,
) -> pd.DataFrame:
    mask = (df["cohort_label"] >= start) & (df["cohort_label"] <= end)
    return df.loc[mask].copy()


def mature_average_retention(
    df: pd.DataFrame,
    *,
    category_col: str = "first_category",
    max_months_since: int = MAX_MATURE_MONTHS_SINCE,
) -> pd.DataFrame:
    """
    Mean-of-rates mature curve: equal weight per cohort_month in [2024-05, 2025-05].

    For each category and months_since, average retention_rate across cohorts that
    have that month observed (all mature cohorts have 0..12 by 2026-06).
    """
    subset = _filter_cohort_range(
        df,
        MATURE_COHORT_START,
        MATURE_COHORT_END,
        category_col=category_col,
    )
    rates = retention_rates_long(subset, category_col=category_col)
    rates = rates[rates["months_since"] <= max_months_since]

    avg = (
        rates.groupby([category_col, "months_since"], as_index=False)["retention_rate"]
        .mean()
    )
    table = avg.pivot(
        index="months_since", columns=category_col, values="retention_rate"
    ).reindex(columns=CATEGORY_ORDER)
    table.index.name = "months_since"
    return table.sort_index()


def election_cohort_retention(
    df: pd.DataFrame,
    *,
    category_col: str = "first_category",
    cohort_label: str = ELECTION_COHORT_MONTH,
) -> pd.DataFrame:
    """Retention rates for a single acquisition cohort, by category."""
    subset = df[df["cohort_label"] == cohort_label].copy()
    rates = retention_rates_long(subset, category_col=category_col)
    table = rates.pivot(
        index="months_since", columns=category_col, values="retention_rate"
    ).reindex(columns=CATEGORY_ORDER)
    table.index.name = "months_since"
    return table.sort_index()


def acquisition_mix(
    df: pd.DataFrame,
    *,
    category_col: str = "first_category",
) -> pd.DataFrame:
    """Share of all acquired users by first_category (sum of cohort sizes)."""
    sizes = cohort_sizes(df, category_col=category_col)
    by_cat = (
        sizes.groupby(category_col, as_index=False)["cohort_size"]
        .sum()
        .rename(columns={"cohort_size": "acquired_users"})
    )
    total = by_cat["acquired_users"].sum()
    by_cat["pct_of_users"] = 100.0 * by_cat["acquired_users"] / total
    by_cat = by_cat.sort_values("acquired_users", ascending=False)
    return by_cat


def acquisition_by_month(
    df: pd.DataFrame,
    *,
    start_label: str = MATURE_COHORT_START,
) -> pd.DataFrame:
    """Overall new traders (months_since=0) by cohort month."""
    sizes = cohort_sizes(df)
    out = sizes[sizes["cohort_label"] >= start_label].sort_values("cohort_label")
    return out


def median_sequential_retention(
    df: pd.DataFrame,
    milestones: tuple[int, ...] = SEQUENTIAL_MILESTONES,
) -> pd.Series:
    """
    Median across cohorts of sequential retention at each milestone month.

    For milestone M, only cohorts with max(months_since) >= M are included
    (maturity filter — e.g. month 6 needs 6+ months of observable data).
    """
    rates = retention_rates_long(df)
    medians: dict[int, float] = {}
    for m in milestones:
        cohort_max = rates.groupby("cohort_label")["months_since"].max()
        eligible = cohort_max[cohort_max >= m].index
        at_m = rates[
            (rates["months_since"] == m) & (rates["cohort_label"].isin(eligible))
        ]
        medians[m] = float(at_m["retention_rate"].median())
    return pd.Series(medians, name="median_retention_rate")


def mature_ever_returned_summary(
    df: pd.DataFrame,
    *,
    start: str = EVER_RETURNED_MATURE_START,
    end: str = EVER_RETURNED_MATURE_END,
) -> EverReturnedSummary:
    """
    Ever-returned rates over mature cohorts (default 2024-01 .. 2025-12).

    Pooled rate = sum(ever_returned) / sum(cohort_size).
    Median rate = median of per-cohort ever_returned_pct (equal weight per cohort).
    """
    subset = df[(df["cohort_label"] >= start) & (df["cohort_label"] <= end)].copy()
    if subset.empty:
        raise ValueError(f"No ever-returned rows for cohorts {start} .. {end}")

    total_size = int(subset["cohort_size"].sum())
    total_returned = int(subset["ever_returned"].sum())
    pooled = total_returned / total_size if total_size else 0.0
    median = float(subset["ever_returned_pct"].median())

    return EverReturnedSummary(
        pooled_rate=pooled,
        median_rate=median,
        cohort_start=start,
        cohort_end=end,
        total_cohort_size=total_size,
        total_ever_returned=total_returned,
        n_cohorts=len(subset),
    )


def rate_at_month(table: pd.DataFrame, month: int, category: str) -> float | None:
    """Lookup retention rate from a pivoted table; None if missing."""
    if month not in table.index or category not in table.columns:
        return None
    val = table.loc[month, category]
    if pd.isna(val):
        return None
    return float(val)


def pct_str(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{100.0 * value:.{decimals}f}%"
