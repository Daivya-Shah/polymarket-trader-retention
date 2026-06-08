#!/usr/bin/env python3
"""Connectivity check: Polymarket Gamma API and Dune saved query."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# Allow imports from src/ when run as a script without editable install
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from config import load_settings  # noqa: E402
from data_access import get_top_markets, run_dune_query  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)


def _configure_stdout() -> None:
    """Use UTF-8 on Windows so summary checkmarks print correctly."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (LookupError, OSError):
            pass


def _print_gamma() -> bool:
    print("=" * 60)
    print("Gamma API — top markets by 24h volume")
    print("=" * 60)
    try:
        df = get_top_markets(limit=5)
        if df.empty:
            print("(no markets returned)")
        else:
            print(df.to_string(index=False))
        print(f"\nRows: {len(df)}")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def _print_dune() -> bool:
    print("\n" + "=" * 60)
    print("Dune API — saved check query")
    print("=" * 60)
    try:
        settings = load_settings()
        df = run_dune_query(settings.dune_check_query_id, settings=settings)
        if df.empty:
            print("(empty result set)")
        else:
            with pd.option_context(
                "display.max_columns",
                None,
                "display.width",
                120,
                "display.max_colwidth",
                50,
            ):
                print(df.to_string(index=False))
        print(f"\nRows: {len(df)}")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def main() -> int:
    _configure_stdout()
    gamma_ok = _print_gamma()
    dune_ok = _print_dune()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Gamma API:  {'✅ OK' if gamma_ok else '❌ FAILED'}")
    print(f"Dune API:   {'✅ OK' if dune_ok else '❌ FAILED'}")

    if gamma_ok and dune_ok:
        print("\nAll connectivity checks passed.")
        return 0
    print("\nOne or more checks failed. See errors above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
