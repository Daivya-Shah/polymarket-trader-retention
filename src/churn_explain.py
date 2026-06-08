"""
SHAP explainability and wallet scoring for churn (active_m3) models.

Uses calibrated probabilities from the saved bundle; TreeExplainer runs on the
underlying LightGBM inside CalibratedClassifierCV(FrozenEstimator(...)).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV

from churn_features import HRS_TO_2ND_NULL_SENTINEL

logger = logging.getLogger(__name__)

Variant = Literal["behavioral", "full"]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / "models"

_VARIANT_FILES: Final[dict[str, dict[str, str]]] = {
    "behavioral": {
        "model": "churn_lgbm_behavioral.joblib",
        "features": "feature_names_behavioral.json",
        "metrics": "metrics_behavioral.json",
    },
    "full": {
        "model": "churn_lgbm.joblib",
        "features": "feature_names.json",
        "metrics": "metrics.json",
    },
}

RAW_INPUT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "n_trades_w1",
        "active_days_w1",
        "n_markets_w1",
        "n_categories_w1",
        "multi_category_flag",
        "extreme_price_share",
        "median_hour",
        "hrs_to_2nd",
        "has_second_trade",
        "first_category",
    }
)

CATEGORY_LEVELS: Final[list[str]] = ["crypto", "other", "politics", "sports"]

# p_churn cutoffs (base churn ~61% on test; bands target growth triage).
RISK_HIGH_P_CHURN: Final[float] = 0.70
RISK_MEDIUM_P_CHURN: Final[float] = 0.40


@dataclass(frozen=True)
class ModelBundle:
    """Loaded calibrated model, metadata, and SHAP explainer."""

    variant: str
    model: CalibratedClassifierCV
    tree_model: lgb.LGBMClassifier
    feature_names: list[str]
    metrics: dict[str, Any]
    explainer: shap.TreeExplainer


def extract_tree_model(calibrated: CalibratedClassifierCV) -> lgb.LGBMClassifier:
    """
    Extract the fitted LGBMClassifier from CalibratedClassifierCV + FrozenEstimator.

    Path: calibrated_classifiers_[0].estimator.estimator (sklearn 1.9 layout).
    """
    try:
        frozen = calibrated.calibrated_classifiers_[0].estimator
        tree = frozen.estimator
    except (AttributeError, IndexError) as exc:
        raise ValueError(
            "Could not extract LightGBM from calibrated model; "
            "expected FrozenEstimator wrapping LGBMClassifier"
        ) from exc
    if not isinstance(tree, lgb.LGBMClassifier):
        raise TypeError(f"Expected LGBMClassifier, got {type(tree).__name__}")
    return tree


def build_explainer(tree_model: lgb.LGBMClassifier) -> shap.TreeExplainer:
    """Build a SHAP TreeExplainer on the raw LightGBM booster."""
    return shap.TreeExplainer(tree_model)


def load_model_bundle(
    variant: Variant = "behavioral",
    *,
    models_dir: Path | str = _MODELS_DIR,
) -> ModelBundle:
    """Load calibrated model, feature names, metrics, and SHAP explainer."""
    if variant not in _VARIANT_FILES:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(_VARIANT_FILES)}")

    root = Path(models_dir)
    paths = _VARIANT_FILES[variant]
    model_path = root / paths["model"]
    features_path = root / paths["features"]
    metrics_path = root / paths["metrics"]

    for p in (model_path, features_path, metrics_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing artifact for variant={variant}: {p}")

    calibrated = joblib.load(model_path)
    with open(features_path, encoding="utf-8") as f:
        feature_names = json.load(f)
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    tree_model = extract_tree_model(calibrated)
    explainer = build_explainer(tree_model)
    logger.info(
        "Loaded %s bundle: %d features, target=%s",
        variant,
        len(feature_names),
        metrics.get("target", "?"),
    )
    return ModelBundle(
        variant=variant,
        model=calibrated,
        tree_model=tree_model,
        feature_names=feature_names,
        metrics=metrics,
        explainer=explainer,
    )


def _normalize_wallet_inputs(features: dict[str, Any]) -> dict[str, Any]:
    """Apply hrs_to_2nd / has_second_trade rules matching churn_features.load_churn_frame."""
    out = dict(features)
    hrs = out.get("hrs_to_2nd")
    if hrs is None or (isinstance(hrs, float) and np.isnan(hrs)):
        out["hrs_to_2nd"] = HRS_TO_2ND_NULL_SENTINEL
        if "has_second_trade" not in out:
            out["has_second_trade"] = 0
    elif "has_second_trade" not in out:
        out["has_second_trade"] = 1
    return out


def _encode_row(features: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    """One-hot first_category and align columns to the trained feature order."""
    norm = _normalize_wallet_inputs(features)
    if "first_category" not in norm:
        raise ValueError("features must include first_category")

    row: dict[str, float] = {}
    for name in feature_names:
        if name.startswith("first_category_"):
            row[name] = 0.0

    scalar_fields = [
        "n_trades_w1",
        "active_days_w1",
        "n_markets_w1",
        "n_categories_w1",
        "extreme_price_share",
        "median_hour",
        "hrs_to_2nd",
        "multi_category_flag",
        "has_second_trade",
    ]
    for field in scalar_fields:
        if field in feature_names and field in norm:
            row[field] = float(norm[field])

    cat = str(norm["first_category"])
    cat_col = f"first_category_{cat}"
    if cat_col in feature_names:
        row[cat_col] = 1.0
    elif f"first_category_{cat}" not in [n for n in feature_names if n.startswith("first_category_")]:
        raise ValueError(f"Unknown first_category={cat!r} for model features")

    for name in feature_names:
        row.setdefault(name, 0.0)

    return pd.DataFrame([row])[feature_names]


def encode_dataframe(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Batch-encode raw churn feature rows for vectorized scoring."""
    work = df.copy()
    work["has_second_trade"] = work["hrs_to_2nd"].notna().astype(int)
    work["hrs_to_2nd"] = work["hrs_to_2nd"].fillna(HRS_TO_2ND_NULL_SENTINEL)

    numeric_cols = [
        c
        for c in feature_names
        if not c.startswith("first_category_")
    ]
    X = work[numeric_cols].astype(float).copy()

    cats = pd.get_dummies(
        work["first_category"].astype(str),
        prefix="first_category",
        prefix_sep="_",
        dtype=float,
    )
    for name in feature_names:
        if name.startswith("first_category_") and name not in cats.columns:
            cats[name] = 0.0

    for name in feature_names:
        if name.startswith("first_category_"):
            X[name] = cats.get(name, 0.0)

    return X[feature_names]


def _risk_band(p_churn: float) -> str:
    if p_churn >= RISK_HIGH_P_CHURN:
        return "high"
    if p_churn >= RISK_MEDIUM_P_CHURN:
        return "medium"
    return "low"


def _plain_reason(feature: str, value: float, shap_val: float) -> dict[str, Any]:
    """Translate one SHAP contribution into a short growth-facing sentence."""
    increases_churn = shap_val < 0
    direction = "increases" if increases_churn else "decreases"

    if feature == "n_trades_w1":
        plain = (
            f"Only {int(value)} trade(s) in week one"
            if value <= 2
            else f"{int(value)} trades in week one"
        )
    elif feature == "active_days_w1":
        plain = (
            f"Traded on only {int(value)} day(s) in week one"
            if value <= 1
            else f"Active on {int(value)} distinct days in week one"
        )
    elif feature == "n_markets_w1":
        plain = (
            f"Explored only {int(value)} market(s) in week one"
            if value <= 1
            else f"Explored {int(value)} markets in week one"
        )
    elif feature == "n_categories_w1":
        plain = (
            f"Stayed in {int(value)} category in week one"
            if value <= 1
            else f"Explored {int(value)} categories in week one"
        )
    elif feature == "multi_category_flag":
        plain = (
            "Single-category week-one behavior"
            if int(value) == 0
            else "Multi-category exploration in week one"
        )
    elif feature == "has_second_trade":
        plain = (
            "No second trade after first"
            if int(value) == 0
            else "Returned quickly for a second trade"
        )
    elif feature == "hrs_to_2nd":
        if value >= HRS_TO_2ND_NULL_SENTINEL - 1:
            plain = "Never made a second trade"
        elif value <= 24:
            plain = f"Second trade within {int(value)} hour(s)"
        else:
            plain = f"Second trade after {int(value)} hours"
    elif feature == "extreme_price_share":
        pct = int(round(100 * value))
        plain = f"{pct}% of week-one trades at extreme prices (<10c or >90c)"
    elif feature == "median_hour":
        plain = f"Typical trading hour around {int(value)}:00 UTC"
    elif feature.startswith("first_category_"):
        label = feature.removeprefix("first_category_")
        plain = (
            f"First category: {label}"
            if value >= 0.5
            else f"Not a {label} first-trade profile"
        )
    else:
        plain = f"{feature} = {value:.3g}"

    plain = f"{plain} -> {direction} churn risk"
    return {
        "feature": feature,
        "shap": float(shap_val),
        "direction": f"{direction} churn risk",
        "plain": plain,
    }


def score_wallet(features: dict[str, Any], bundle: ModelBundle) -> dict[str, Any]:
    """
    Score one wallet and return calibrated probabilities plus SHAP reasons.

    ``features`` uses pre-encoding field names (see RAW_INPUT_FIELDS).
    """
    X = _encode_row(features, bundle.feature_names)
    p_active = float(bundle.model.predict_proba(X)[0, 1])
    p_churn = 1.0 - p_active

    shap_values = bundle.explainer.shap_values(X)
    if isinstance(shap_values, list):
        # Binary classifiers may return a list per class; use positive (active) class.
        shap_row = np.asarray(shap_values[1])[0]
    else:
        shap_row = np.asarray(shap_values)[0]

    contributors = sorted(
        [
            _plain_reason(name, float(X[name].iloc[0]), float(shap_row[i]))
            for i, name in enumerate(bundle.feature_names)
        ],
        key=lambda r: abs(r["shap"]),
        reverse=True,
    )
    top_reasons = [r for r in contributors if abs(r["shap"]) > 1e-9][:4]

    return {
        "p_active_m3": p_active,
        "p_churn": p_churn,
        "risk_band": _risk_band(p_churn),
        "top_reasons": top_reasons,
    }


def score_dataframe(df: pd.DataFrame, bundle: ModelBundle) -> pd.DataFrame:
    """Vectorized churn scores (no per-row SHAP)."""
    X = encode_dataframe(df, bundle.feature_names)
    p_active = bundle.model.predict_proba(X)[:, 1]
    out = df.copy()
    out["p_active_m3"] = p_active
    out["p_churn"] = 1.0 - p_active
    out["risk_band"] = [_risk_band(p) for p in out["p_churn"]]
    return out
