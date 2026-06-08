"""Reusable clients for Dune Analytics and the Polymarket Gamma API."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests
from dune_client.client import DuneClient
from dune_client.query import QueryBase

from config import Settings, load_settings

logger = logging.getLogger(__name__)

GAMMA_EVENTS_URL = (
    "https://gamma-api.polymarket.com/events"
    "?active=true&closed=false&order=volume24hr&ascending=false&limit=5"
)


class DataAccessError(Exception):
    """Raised when an external data API call fails."""


def get_dune_client(settings: Settings | None = None) -> DuneClient:
    """Return an authenticated Dune client."""
    cfg = settings or load_settings()
    logger.debug("Initializing Dune client")
    # Free-tier accounts require the "small" engine; "medium" returns 400.
    return DuneClient(
        api_key=cfg.dune_api_key,
        base_url="https://api.dune.com",
        request_timeout=300,
        performance="small",
    )


def _result_rows_to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _is_forbidden_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", None) == 403:
        return True
    return "403" in str(exc) and "Forbidden" in str(exc)


def run_dune_query(
    query_id: int,
    *,
    settings: Settings | None = None,
    client: DuneClient | None = None,
    allow_cached_on_forbidden: bool = True,
) -> pd.DataFrame:
    """
    Execute a saved Dune query by ID and return results as a DataFrame.

    Uses ``DuneClient.run_query`` with ``QueryBase`` (free tier: saved queries only).
    If execution returns 403 (common for temporary/unsaved queries), optionally falls
    back to ``get_latest_result`` when the query was already run in the Dune UI.
    """
    dune = client or get_dune_client(settings)
    query = QueryBase(
        name="polymarket-growth-query",
        query_id=query_id,
    )
    logger.info("Running Dune query_id=%s", query_id)
    try:
        result = dune.run_query(query, performance="small")
    except Exception as exc:
        if allow_cached_on_forbidden and _is_forbidden_error(exc):
            logger.warning(
                "Execute forbidden for query_id=%s (403). Query may be temporary/unsaved. "
                "Trying cached results from the last Dune UI run. Save the query in Dune "
                "to allow API re-execution.",
                query_id,
            )
            try:
                result = dune.get_latest_result(
                    query,
                    max_age_hours=24 * 365 * 10,
                )
            except Exception as cache_exc:
                logger.error(
                    "Cached result fetch also failed for query_id=%s: %s",
                    query_id,
                    cache_exc,
                )
                raise DataAccessError(
                    f"Dune query {query_id} failed to execute (403 Forbidden). "
                    "Open the query in Dune, click Save (not just Run), then retry. "
                    f"Cached fetch error: {cache_exc}"
                ) from cache_exc
        else:
            logger.error("Dune query failed for query_id=%s: %s", query_id, exc)
            raise DataAccessError(
                f"Dune query {query_id} failed: {exc}"
            ) from exc

    rows: list[dict[str, Any]] = result.result.rows
    if not rows:
        logger.warning("Dune query_id=%s returned zero rows", query_id)
        return pd.DataFrame()

    df = _result_rows_to_dataframe(rows)
    logger.info("Dune query_id=%s returned %d rows", query_id, len(df))
    return df


def get_dune_latest_result(
    query_id: int,
    *,
    settings: Settings | None = None,
    client: DuneClient | None = None,
    max_age_hours: int = 24 * 365 * 10,
) -> pd.DataFrame:
    """
    Return the latest cached Dune result without re-executing the query.

    Useful when a saved query was already run in the Dune UI.
    """
    dune = client or get_dune_client(settings)
    query = QueryBase(name="polymarket-growth-query", query_id=query_id)
    logger.info("Fetching latest Dune result for query_id=%s", query_id)
    try:
        result = dune.get_latest_result(query, max_age_hours=max_age_hours)
    except Exception as exc:
        logger.error("Latest result fetch failed for query_id=%s: %s", query_id, exc)
        raise DataAccessError(
            f"Dune latest result for {query_id} failed: {exc}"
        ) from exc

    rows: list[dict[str, Any]] = result.result.rows
    df = _result_rows_to_dataframe(rows)
    logger.info("Dune query_id=%s latest result: %d rows", query_id, len(df))
    return df


def fetch_dune_query(
    query_id: int,
    *,
    settings: Settings | None = None,
    prefer_cached: bool = True,
) -> pd.DataFrame:
    """
    Fetch query results, optionally trying cached results before executing.

    When ``prefer_cached`` is True, attempts ``get_dune_latest_result`` first; on
    failure, falls back to ``run_dune_query`` (execute + wait).
    """
    if prefer_cached:
        try:
            return get_dune_latest_result(query_id, settings=settings)
        except DataAccessError:
            logger.info(
                "Cached result unavailable for query_id=%s; executing query",
                query_id,
            )
    return run_dune_query(query_id, settings=settings)


def get_top_markets(limit: int = 5) -> pd.DataFrame:
    """
    Fetch active Polymarket events from Gamma and return top markets by volume.

    Columns: ``question``, ``volume``, ``liquidity``.
    """
    logger.info("Fetching top markets from Gamma API (limit=%d)", limit)
    try:
        response = requests.get(GAMMA_EVENTS_URL, timeout=30)
        response.raise_for_status()
        events = response.json()
    except requests.RequestException as exc:
        logger.error("Gamma API request failed: %s", exc)
        raise DataAccessError(f"Gamma API request failed: {exc}") from exc

    if not isinstance(events, list):
        raise DataAccessError(
            f"Unexpected Gamma API response type: {type(events).__name__}"
        )

    records: list[dict[str, Any]] = []
    for event in events:
        markets = event.get("markets") or []
        for market in markets:
            records.append(
                {
                    "question": market.get("question"),
                    "volume": _to_float(market.get("volume")),
                    "liquidity": _to_float(market.get("liquidity")),
                }
            )

    if not records:
        logger.warning("Gamma API returned no markets")
        return pd.DataFrame(columns=["question", "volume", "liquidity"])

    df = pd.DataFrame(records)
    df = df.sort_values("volume", ascending=False, na_position="last")
    df = df.head(limit).reset_index(drop=True)
    logger.info("Gamma API: returning %d markets", len(df))
    return df


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
