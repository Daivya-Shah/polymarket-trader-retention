#!/usr/bin/env python3
"""Train behavioral-only churn model (no volume/USD features) — whale robustness check."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import early_stopping
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from churn_features import (  # noqa: E402
    ChurnData,
    build_feature_matrix,
    load_churn_frame,
    time_based_split,
)

TARGET = "active_m3"
VARIANT = "behavioral_only"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE = 42
DATA_PATH = _ROOT / "data" / "raw" / "churn_features.csv"
MODELS_DIR = _ROOT / "models"
CALIB_HOLDOUT_FRACTION = 0.20


def _configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (LookupError, OSError):
            pass


def _temporal_calib_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    holdout_fraction: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    n = len(X_train)
    split_at = int(n * (1.0 - holdout_fraction))
    if split_at <= 0 or split_at >= n:
        raise ValueError(
            f"Invalid calibration split at {split_at} for n={n} "
            f"(holdout_fraction={holdout_fraction})"
        )
    return (
        X_train.iloc[:split_at].reset_index(drop=True),
        y_train.iloc[:split_at].reset_index(drop=True),
        X_train.iloc[split_at:].reset_index(drop=True),
        y_train.iloc[split_at:].reset_index(drop=True),
    )


def _top_decile_churn_metrics(
    y_true: pd.Series,
    proba_active: np.ndarray,
) -> dict[str, float]:
    n_top = max(1, int(len(y_true) * 0.10))
    order = np.argsort(proba_active)
    top_idx = order[:n_top]
    y_top = y_true.iloc[top_idx]
    churn_rate_top = float((y_top == 0).mean())
    base_churn = float((y_true == 0).mean())
    lift = churn_rate_top / base_churn if base_churn else float("nan")
    return {
        "top_decile_churn_precision": churn_rate_top,
        "base_churn_rate": base_churn,
        "top_decile_lift": lift,
        "top_decile_n": n_top,
    }


def _evaluate_lightgbm(
    y_true: pd.Series,
    proba_active: np.ndarray,
) -> dict[str, Any]:
    decile_metrics = _top_decile_churn_metrics(y_true, proba_active)
    return {
        "model": "lightgbm_calibrated_behavioral",
        "roc_auc": float(roc_auc_score(y_true, proba_active)),
        "pr_auc": float(average_precision_score(y_true, proba_active)),
        "brier": float(brier_score_loss(y_true, proba_active)),
        **decile_metrics,
    }


def _print_core_metrics(metrics: dict[str, Any]) -> None:
    print("\n" + "=" * 56)
    print(f"Behavioral-only model (target={TARGET}, no volume features)")
    print("=" * 56)
    print(f"  ROC-AUC:              {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:               {metrics['pr_auc']:.4f}")
    print(f"  Brier:                {metrics['brier']:.4f}")
    print(
        f"  Top-decile churn prec: {metrics['top_decile_churn_precision']:.1%}  "
        f"lift={metrics['top_decile_lift']:.2f}x  "
        f"(base churn {metrics['base_churn_rate']:.1%})"
    )


def _print_feature_importance(
    lgbm: lgb.LGBMClassifier,
    feature_names: list[str],
    *,
    top_n: int = 15,
) -> None:
    gain = lgbm.booster_.feature_importance(importance_type="gain")
    ranked = sorted(
        zip(feature_names, gain, strict=True),
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]
    print(f"\nLightGBM top {top_n} features (gain importance)")
    print("-" * 48)
    print(f"{'feature':<28} {'gain':>10}")
    for name, score in ranked:
        print(f"{name:<28} {score:>10.1f}")


def train_lightgbm(
    data: ChurnData,
) -> tuple[dict[str, Any], Any, lgb.LGBMClassifier]:
    logger.info("Training behavioral-only LightGBM …")
    n_pos = int(data.y_train.sum())
    n_neg = len(data.y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0
    logger.info(
        "scale_pos_weight=%.3f (neg=%d pos=%d)",
        scale_pos_weight,
        n_neg,
        n_pos,
    )

    X_fit, y_fit, X_calib, y_calib = _temporal_calib_split(
        data.X_train,
        data.y_train,
        holdout_fraction=CALIB_HOLDOUT_FRACTION,
    )

    lgbm = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=100,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    lgbm.fit(
        X_fit,
        y_fit,
        eval_set=[(data.X_test, data.y_test)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )

    calibrated = CalibratedClassifierCV(
        FrozenEstimator(lgbm),
        method="isotonic",
    )
    calibrated.fit(X_calib, y_calib)

    proba = calibrated.predict_proba(data.X_test)[:, 1]
    metrics = _evaluate_lightgbm(data.y_test, proba)
    logger.info(
        "Done: PR-AUC=%.4f ROC-AUC=%.4f Brier=%.4f",
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["brier"],
    )
    return metrics, calibrated, lgbm


def main() -> int:
    _configure_stdout()
    np.random.seed(RANDOM_STATE)

    logger.info(
        "Loading churn features (target=%s, variant=%s)",
        TARGET,
        VARIANT,
    )
    df = load_churn_frame(DATA_PATH)
    X, y, feature_names = build_feature_matrix(
        df, target=TARGET, drop_volume=True
    )
    data = time_based_split(df, X, y)
    test_base_churn = float((data.y_test == 0).mean())

    metrics, calibrated, lgbm = train_lightgbm(data)
    _print_core_metrics(metrics)
    _print_feature_importance(lgbm, feature_names)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, MODELS_DIR / "churn_lgbm_behavioral.joblib")
    with open(MODELS_DIR / "feature_names_behavioral.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    payload = {
        "random_state": RANDOM_STATE,
        "variant": VARIANT,
        "target": TARGET,
        "train_size": len(data.X_train),
        "test_size": len(data.X_test),
        "train_positive_rate": float(data.y_train.mean()),
        "test_positive_rate": float(data.y_test.mean()),
        "test_base_churn_rate": test_base_churn,
        "n_features": len(feature_names),
        "lightgbm_calibrated_behavioral": metrics,
    }
    with open(MODELS_DIR / "metrics_behavioral.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Saved behavioral artifacts to %s", MODELS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
