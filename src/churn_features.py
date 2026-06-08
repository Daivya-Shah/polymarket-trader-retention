"""
Prepare data/raw/churn_features.csv for churn modeling.

Predicts a configurable retention label (e.g. ``active_m3``) from week-1
behavioral features only. Loads, cleans, encodes, and time-splits — no training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

logger = logging.getLogger(__name__)

# All possible retention labels; only columns present in the CSV are used.
LABEL_COLUMNS: Final[list[str]] = [
    "active_m1",
    "active_m2",
    "active_m3",
    "active_m6",
    "active_m6b",
]
DEFAULT_TARGET: Final[str] = "active_m3"
TARGET: Final[str] = DEFAULT_TARGET  # default for scripts; override via build_feature_matrix

ID_COLS: Final[list[str]] = ["wallet"]

NUMERIC_FEATURES: Final[list[str]] = [
    "n_trades_w1",
    "active_days_w1",
    "n_markets_w1",
    "n_categories_w1",
    "avg_usd",
    "max_usd",
    "total_usd",
    "extreme_price_share",
    "median_hour",
    "hrs_to_2nd",
]
BINARY_FEATURES: Final[list[str]] = ["multi_category_flag", "has_second_trade"]
CATEGORICAL_FEATURES: Final[list[str]] = ["first_category"]

# Behavioral-only set (excludes raw volume/size) for robustness checks later.
VOLUME_FEATURES: Final[list[str]] = ["avg_usd", "max_usd", "total_usd"]

HRS_TO_2ND_NULL_SENTINEL: Final[float] = 24.0 * 365.0  # 8760 hours — never-made-2nd-trade


def label_columns_in(df: pd.DataFrame) -> list[str]:
    """Return retention label columns present in ``df``."""
    return [c for c in LABEL_COLUMNS if c in df.columns]


@dataclass
class ChurnData:
    """Time-based train/test split ready for model fitting."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    train_cohorts: list[pd.Timestamp]
    test_cohorts: list[pd.Timestamp]


def load_churn_frame(path: Path | str) -> pd.DataFrame:
    """
    Load churn features CSV and apply modeling-safe cleaning.

    Null ``hrs_to_2nd`` means no second trade ever — not "instant second trade".
    Adds ``has_second_trade`` and fills null gaps with a large sentinel for trees.
    Coerces all present label columns to int.
    """
    df = pd.read_csv(path)
    labels = label_columns_in(df)
    if not labels:
        raise ValueError(
            f"No label columns found. Expected any of: {LABEL_COLUMNS}"
        )

    out = df.copy()
    out["cohort_month"] = pd.to_datetime(out["cohort_month"], utc=True)

    for col in labels:
        null_count = int(out[col].isna().sum())
        if null_count:
            raise ValueError(f"{col} has {null_count} null values after load; cannot model")
        out[col] = pd.to_numeric(out[col], errors="raise").astype("int64")

    for col in NUMERIC_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="raise")

    out["has_second_trade"] = out["hrs_to_2nd"].notna().astype("int64")
    out["hrs_to_2nd"] = out["hrs_to_2nd"].fillna(HRS_TO_2ND_NULL_SENTINEL)

    if "multi_category_flag" in out.columns:
        out["multi_category_flag"] = pd.to_numeric(
            out["multi_category_flag"], errors="raise"
        ).astype("int64")

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    target: str = DEFAULT_TARGET,
    drop_volume: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Build model matrix ``X`` and target ``y`` from a cleaned churn frame.

    ``target`` selects which retention label is ``y``. All label columns are
    excluded from ``X`` to prevent leakage. One-hot encodes ``first_category``.

    When ``drop_volume`` is True, excludes ``VOLUME_FEATURES`` (avg/max/total_usd)
    for behavioral-only robustness checks.
    """
    labels = label_columns_in(df)
    if target not in labels:
        raise ValueError(
            f"target={target!r} not in dataframe labels {labels}"
        )

    numeric_cols = [
        c for c in NUMERIC_FEATURES
        if c not in VOLUME_FEATURES or not drop_volume
    ]
    missing_num = [c for c in numeric_cols if c not in df.columns]
    missing_bin = [c for c in BINARY_FEATURES if c not in df.columns]
    missing_cat = [c for c in CATEGORICAL_FEATURES if c not in df.columns]
    if missing_num or missing_bin or missing_cat:
        raise ValueError(
            f"Missing feature columns: numeric={missing_num}, "
            f"binary={missing_bin}, categorical={missing_cat}"
        )

    y = df[target].copy()
    numeric = df[numeric_cols].copy()
    binary = df[BINARY_FEATURES].copy()

    cat_dummies = pd.get_dummies(
        df[CATEGORICAL_FEATURES[0]],
        prefix="first_category",
        prefix_sep="_",
        drop_first=False,
        dtype="int64",
    )

    X = pd.concat([numeric, binary, cat_dummies], axis=1)
    feature_names = list(X.columns)

    leaked = set(feature_names) & set(labels)
    assert not leaked, f"Label columns leaked into features: {sorted(leaked)}"

    return X, y, feature_names


def time_based_split(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_fraction: float = 0.25,
) -> ChurnData:
    """
    Chronological cohort split: earliest cohorts → train, latest → test.

    Assigns the most recent cohort months to test until ~``test_fraction`` of
    wallets are in the test set (anti-leakage: test is strictly later in time).
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")

    cohorts = sorted(df["cohort_month"].unique())
    total = len(df)
    test_cohort_set: list[pd.Timestamp] = []
    test_count = 0

    for cohort in reversed(cohorts):
        n = int((df["cohort_month"] == cohort).sum())
        test_cohort_set.append(cohort)
        test_count += n
        if test_count / total >= test_fraction:
            break

    test_cohorts = sorted(test_cohort_set)
    train_cohorts = [c for c in cohorts if c not in test_cohorts]
    if not train_cohorts or not test_cohorts:
        raise ValueError(
            f"Split produced empty train or test set (cohorts={len(cohorts)}, "
            f"test_fraction={test_fraction})"
        )

    test_mask = df["cohort_month"].isin(test_cohorts)
    train_mask = ~test_mask

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_test = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    train_pos = float(y_train.mean())
    test_pos = float(y_test.mean())
    train_labels = df.loc[train_mask, "cohort_month"].dt.strftime("%Y-%m")
    test_labels = df.loc[test_mask, "cohort_month"].dt.strftime("%Y-%m")

    logger.info(
        "Churn split: train=%d test=%d | train cohorts %s..%s | test cohorts %s..%s | "
        "positive rate train=%.3f test=%.3f",
        len(X_train),
        len(X_test),
        train_labels.min(),
        train_labels.max(),
        test_labels.min(),
        test_labels.max(),
        train_pos,
        test_pos,
    )

    return ChurnData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=list(X.columns),
        train_cohorts=train_cohorts,
        test_cohorts=test_cohorts,
    )
