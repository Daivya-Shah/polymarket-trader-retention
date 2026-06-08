#!/usr/bin/env python3
"""Demo SHAP explainability and wallet scoring for the behavioral churn model."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import shap

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from churn_explain import (  # noqa: E402
    encode_dataframe,
    extract_tree_model,
    load_model_bundle,
    score_wallet,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DATA_PATH = _ROOT / "data" / "raw" / "churn_features.csv"
FIGURE_PATH = _ROOT / "outputs" / "figures" / "churn_shap_summary.png"
SHAP_SAMPLE_SIZE = 2000
RANDOM_STATE = 42


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (LookupError, OSError):
            pass


def _row_to_features(row: pd.Series) -> dict:
    hrs = row["hrs_to_2nd"]
    if pd.isna(hrs):
        hrs_val = None
        has_second = 0
    else:
        hrs_val = float(hrs)
        has_second = 1
    return {
        "n_trades_w1": int(row["n_trades_w1"]),
        "active_days_w1": int(row["active_days_w1"]),
        "n_markets_w1": int(row["n_markets_w1"]),
        "n_categories_w1": int(row["n_categories_w1"]),
        "multi_category_flag": int(row["multi_category_flag"]),
        "extreme_price_share": float(row["extreme_price_share"]),
        "median_hour": float(row["median_hour"]),
        "hrs_to_2nd": hrs_val,
        "has_second_trade": has_second,
        "first_category": str(row["first_category"]),
    }


def _pick_example_wallets(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Select low / high / middle archetype wallets from raw features."""
    low_mask = (
        (df["active_days_w1"] >= 3)
        & (df["multi_category_flag"] == 1)
        & (df["n_categories_w1"] >= 2)
        & (df["hrs_to_2nd"].notna())
        & (df["hrs_to_2nd"] <= 48)
    )
    high_mask = (
        (df["n_trades_w1"] == 1)
        & (df["active_days_w1"] == 1)
        & (df["hrs_to_2nd"].isna())
    )
    if not low_mask.any():
        raise ValueError("No low-risk example wallet found in churn_features.csv")
    if not high_mask.any():
        raise ValueError("No high-risk example wallet found in churn_features.csv")

    low = df.loc[low_mask].iloc[0]
    high = df.loc[high_mask].iloc[0]

    mid_mask = (
        (df["n_trades_w1"].between(3, 10))
        & (df["active_days_w1"] == 2)
        & (df["multi_category_flag"] == 0)
    )
    if not mid_mask.any():
        mid = df.iloc[len(df) // 2]
    else:
        mid = df.loc[mid_mask].iloc[0]

    return {"low-risk": low, "high-risk": high, "middle": mid}


def _print_wallet_demo(label: str, row: pd.Series, result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"{label}: {row['wallet']}")
    print("=" * 60)
    print(
        f"  Week-1: trades={int(row['n_trades_w1'])}  days={int(row['active_days_w1'])}  "
        f"markets={int(row['n_markets_w1'])}  categories={int(row['n_categories_w1'])}  "
        f"first_category={row['first_category']}"
    )
    hrs = row["hrs_to_2nd"]
    hrs_txt = "none" if pd.isna(hrs) else f"{float(hrs):.0f}h"
    print(f"  Second trade: {hrs_txt}  actual active_m3={int(row['active_m3'])}")
    print(f"  P(active_m3)={result['p_active_m3']:.3f}  P(churn)={result['p_churn']:.3f}")
    print(f"  Risk band: {result['risk_band']}")
    print("  Top reasons:")
    for i, reason in enumerate(result["top_reasons"], 1):
        print(f"    {i}. {reason['plain']}  (SHAP={reason['shap']:+.3f})")


def _save_shap_summary(bundle, df: pd.DataFrame, out_path: Path) -> None:
    sample = df.sample(n=min(SHAP_SAMPLE_SIZE, len(df)), random_state=RANDOM_STATE)
    X = encode_dataframe(sample, bundle.feature_names)
    tree = extract_tree_model(bundle.model)
    explainer = shap.TreeExplainer(tree)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=bundle.feature_names,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Wrote SHAP summary plot to %s (%d rows sampled)", out_path, len(sample))


def main() -> int:
    _configure_stdout()
    bundle = load_model_bundle("behavioral")
    df = pd.read_csv(DATA_PATH)

    examples = _pick_example_wallets(df)
    for label, row in examples.items():
        result = score_wallet(_row_to_features(row), bundle)
        _print_wallet_demo(label, row, result)

    _save_shap_summary(bundle, df, FIGURE_PATH)
    print(f"\nSHAP summary saved: {FIGURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
