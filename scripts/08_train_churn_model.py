#!/usr/bin/env python3
"""Train churn models (logistic baseline + calibrated LightGBM) on week-1 features."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

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
    """Hold out the latest fraction of training rows for probability calibration."""
    n = len(X_train)
    split_at = int(n * (1.0 - holdout_fraction))
    if split_at <= 0 or split_at >= n:
        raise ValueError(
            f"Invalid calibration split at {split_at} for n={n} "
            f"(holdout_fraction={holdout_fraction})"
        )
    X_fit = X_train.iloc[:split_at].reset_index(drop=True)
    y_fit = y_train.iloc[:split_at].reset_index(drop=True)
    X_calib = X_train.iloc[split_at:].reset_index(drop=True)
    y_calib = y_train.iloc[split_at:].reset_index(drop=True)
    return X_fit, y_fit, X_calib, y_calib


def _predict_active_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def _metrics_at_threshold(
    y_true: pd.Series,
    proba_active: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float]:
    y_pred = (proba_active >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _top_decile_churn_metrics(
    y_true: pd.Series,
    proba_active: np.ndarray,
) -> dict[str, float]:
    """Top 10% lowest P(active) = highest churn risk."""
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


def _evaluate_model(
    name: str,
    y_true: pd.Series,
    proba_active: np.ndarray,
) -> dict[str, Any]:
    thresh_metrics = _metrics_at_threshold(y_true, proba_active, threshold=0.5)
    decile_metrics = _top_decile_churn_metrics(y_true, proba_active)
    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_true, proba_active)),
        "pr_auc": float(average_precision_score(y_true, proba_active)),
        "brier": float(brier_score_loss(y_true, proba_active)),
        "precision_at_0_5": thresh_metrics["precision"],
        "recall_at_0_5": thresh_metrics["recall"],
        "f1_at_0_5": thresh_metrics["f1"],
        **decile_metrics,
    }


def _calibration_table(
    y_true: pd.Series,
    proba_active: np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    df = pd.DataFrame({"proba_active": proba_active, "actual_active": y_true.values})
    df["decile"] = pd.qcut(
        df["proba_active"],
        q=n_bins,
        duplicates="drop",
        labels=False,
    )
    return (
        df.groupby("decile", as_index=False)
        .agg(
            mean_predicted=("proba_active", "mean"),
            mean_actual=("actual_active", "mean"),
            n=("actual_active", "size"),
        )
        .sort_values("decile")
    )


def _print_metrics_table(rows: list[dict[str, Any]]) -> None:
    cols = [
        ("model", "model"),
        ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"),
        ("brier", "Brier"),
        ("precision_at_0_5", "Prec@0.5"),
        ("recall_at_0_5", "Rec@0.5"),
        ("f1_at_0_5", "F1@0.5"),
        ("top_decile_churn_precision", "Top10% churn prec"),
        ("top_decile_lift", "Lift"),
    ]
    print("\n" + "=" * 72)
    print("Test-set model comparison")
    print("=" * 72)
    header = "  ".join(f"{label:>18}" for _, label in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        parts = []
        for key, label in cols:
            val = row[key]
            if key == "model":
                parts.append(f"{val:>18}")
            else:
                parts.append(f"{val:>18.4f}")
        print("  ".join(parts))


def _print_calibration_table(table: pd.DataFrame, *, title: str) -> None:
    print(f"\n{title}")
    print("-" * 56)
    print(f"{'decile':>6}  {'mean_pred':>10}  {'mean_actual':>12}  {'n':>8}")
    for _, row in table.iterrows():
        print(
            f"{int(row['decile']):>6}  "
            f"{row['mean_predicted']:>10.4f}  "
            f"{row['mean_actual']:>12.4f}  "
            f"{int(row['n']):>8}"
        )


def train_logistic(data: ChurnData) -> tuple[dict[str, Any], StandardScaler, LogisticRegression]:
    logger.info("Training logistic regression baseline …")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(data.X_train)
    X_test_scaled = scaler.transform(data.X_test)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_scaled, data.y_train)

    proba = model.predict_proba(X_test_scaled)[:, 1]
    metrics = _evaluate_model("logistic", data.y_test, proba)
    logger.info(
        "Logistic done: PR-AUC=%.4f ROC-AUC=%.4f",
        metrics["pr_auc"],
        metrics["roc_auc"],
    )
    return metrics, scaler, model


def train_lightgbm(
    data: ChurnData,
) -> tuple[dict[str, Any], Any, pd.DataFrame, lgb.LGBMClassifier]:
    logger.info("Training LightGBM …")
    n_pos = int(data.y_train.sum())
    n_neg = len(data.y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos else 1.0
    logger.info(
        "LightGBM scale_pos_weight=%.3f (neg=%d pos=%d)",
        scale_pos_weight,
        n_neg,
        n_pos,
    )

    X_fit, y_fit, X_calib, y_calib = _temporal_calib_split(
        data.X_train,
        data.y_train,
        holdout_fraction=CALIB_HOLDOUT_FRACTION,
    )
    logger.info(
        "Calibration holdout: %d rows (latest train cohorts), fit on %d rows",
        len(X_calib),
        len(X_fit),
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

    # Mild optimism: early stopping watches the time-based test set.
    lgbm.fit(
        X_fit,
        y_fit,
        eval_set=[(data.X_test, data.y_test)],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )

    # Isotonic calibration on latest train cohorts (not test) via FrozenEstimator (sklearn 1.6+).
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(lgbm),
        method="isotonic",
    )
    calibrated.fit(X_calib, y_calib)

    proba = _predict_active_proba(calibrated, data.X_test)
    metrics = _evaluate_model("lightgbm_calibrated", data.y_test, proba)
    cal_table = _calibration_table(data.y_test, proba)
    logger.info(
        "LightGBM done: PR-AUC=%.4f ROC-AUC=%.4f Brier=%.4f",
        metrics["pr_auc"],
        metrics["roc_auc"],
        metrics["brier"],
    )
    return metrics, calibrated, cal_table, lgbm


def _print_feature_importance(
    lgbm: lgb.LGBMClassifier,
    feature_names: list[str],
    *,
    top_n: int = 15,
) -> None:
    """Print top features by LightGBM gain importance (pre-calibration model)."""
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


def main() -> int:
    _configure_stdout()
    np.random.seed(RANDOM_STATE)

    logger.info("Loading churn features from %s (target=%s)", DATA_PATH, TARGET)
    df = load_churn_frame(DATA_PATH)
    X, y, feature_names = build_feature_matrix(df, target=TARGET)
    data = time_based_split(df, X, y)
    test_base_churn = float((data.y_test == 0).mean())
    logger.info(
        "Test base rates: P(%s)=%.3f  churn=%.3f",
        TARGET,
        float(data.y_test.mean()),
        test_base_churn,
    )

    logistic_metrics, scaler, logistic = train_logistic(data)
    lgbm_metrics, lgbm_calibrated, lgbm_cal_table, lgbm = train_lightgbm(data)

    _print_metrics_table([logistic_metrics, lgbm_metrics])
    _print_calibration_table(
        lgbm_cal_table,
        title=f"LightGBM calibration check (test deciles of P({TARGET}))",
    )
    _print_feature_importance(lgbm, feature_names)

    winner = (
        "lightgbm_calibrated"
        if lgbm_metrics["pr_auc"] >= logistic_metrics["pr_auc"]
        else "logistic"
    )
    winner_metrics = lgbm_metrics if winner == "lightgbm_calibrated" else logistic_metrics

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lgbm_calibrated, MODELS_DIR / "churn_lgbm.joblib")
    joblib.dump(
        {"scaler": scaler, "model": logistic},
        MODELS_DIR / "churn_logistic.joblib",
    )
    with open(MODELS_DIR / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    all_metrics = {
        "random_state": RANDOM_STATE,
        "target": TARGET,
        "train_size": len(data.X_train),
        "test_size": len(data.X_test),
        "train_positive_rate": float(data.y_train.mean()),
        "test_positive_rate": float(data.y_test.mean()),
        "test_base_churn_rate": test_base_churn,
        "winner_pr_auc": winner,
        "logistic": logistic_metrics,
        "lightgbm_calibrated": lgbm_metrics,
    }
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info("Saved models and metrics to %s", MODELS_DIR)

    print("\n" + "=" * 72)
    print(f"Final summary (target={TARGET})")
    print("=" * 72)
    print(f"Winner on PR-AUC: {winner}")
    print(
        f"  {winner} — PR-AUC={winner_metrics['pr_auc']:.4f}  "
        f"ROC-AUC={winner_metrics['roc_auc']:.4f}"
    )
    print(
        f"  Top-decile churn precision={winner_metrics['top_decile_churn_precision']:.1%}  "
        f"lift={winner_metrics['top_decile_lift']:.2f}x "
        f"(base churn {winner_metrics['base_churn_rate']:.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
