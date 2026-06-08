"""Environment-backed settings for Polymarket growth analysis."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_PATH = _PROJECT_ROOT / ".env"


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise ValueError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env in {_PROJECT_ROOT} and set it."
        )
    return value.strip()


def _parse_int_env(name: str) -> int:
    raw = _require(name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw!r}") from exc


def _parse_optional_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if not value or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {value!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from the environment."""

    dune_api_key: str
    dune_check_query_id: int
    dune_cohort_query_id: int | None = None
    dune_cohort_cat_query_id: int | None = None
    dune_ever_returned_query_id: int | None = None
    dune_churn_features_query_id: int | None = None
    project_root: Path = _PROJECT_ROOT


def load_settings() -> Settings:
    """Load settings from `.env` at the project root."""
    load_dotenv(_ENV_PATH)
    return Settings(
        dune_api_key=_require("DUNE_API_KEY"),
        dune_check_query_id=_parse_int_env("DUNE_CHECK_QUERY_ID"),
        dune_cohort_query_id=_parse_optional_int_env("DUNE_COHORT_QUERY_ID"),
        dune_cohort_cat_query_id=_parse_optional_int_env("DUNE_COHORT_CAT_QUERY_ID"),
        dune_ever_returned_query_id=_parse_optional_int_env("DUNE_EVER_RETURNED_QUERY_ID"),
        dune_churn_features_query_id=_parse_optional_int_env("DUNE_CHURN_FEATURES_QUERY_ID"),
    )


def require_ever_returned_query_id(settings: Settings | None = None) -> int:
    """Return the ever-returned saved-query ID or raise with setup instructions."""
    cfg = settings or load_settings()
    if cfg.dune_ever_returned_query_id is None:
        raise ValueError(
            "Missing DUNE_EVER_RETURNED_QUERY_ID. Save sql/03_ever_returned.sql as a "
            f"public Dune query (small engine), then add the query_id to {_ENV_PATH}."
        )
    return cfg.dune_ever_returned_query_id


def require_cohort_cat_query_id(settings: Settings | None = None) -> int:
    """Return the category-segmented cohort query ID or raise with setup instructions."""
    cfg = settings or load_settings()
    if cfg.dune_cohort_cat_query_id is None:
        raise ValueError(
            "Missing DUNE_COHORT_CAT_QUERY_ID. Save sql/02_cohort_retention_by_category.sql "
            f"as a public saved Dune query (small engine), then add the query_id to {_ENV_PATH}."
        )
    return cfg.dune_cohort_cat_query_id


def require_churn_features_query_id(settings: Settings | None = None) -> int:
    """Return the churn-features saved-query ID or raise with setup instructions."""
    cfg = settings or load_settings()
    if cfg.dune_churn_features_query_id is None:
        raise ValueError(
            "Missing DUNE_CHURN_FEATURES_QUERY_ID. Save sql/04_churn_features.sql as a "
            f"public Dune query (small engine), then add the query_id to {_ENV_PATH}."
        )
    return cfg.dune_churn_features_query_id


def require_cohort_query_id(settings: Settings | None = None) -> int:
    """Return the cohort retention saved-query ID or raise with setup instructions."""
    cfg = settings or load_settings()
    if cfg.dune_cohort_query_id is None:
        raise ValueError(
            "Missing DUNE_COHORT_QUERY_ID. Save sql/01_cohort_retention.sql as a Dune "
            f"query (small engine), copy the query_id from the URL, and add it to {_ENV_PATH}."
        )
    return cfg.dune_cohort_query_id
