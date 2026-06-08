#!/usr/bin/env python3
"""Streamlit demo: Polymarket churn radar (month-3 retention from week-1 behavior)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from churn_explain import load_model_bundle, score_dataframe, score_wallet  # noqa: E402

DATA_PATH = _ROOT / "data" / "raw" / "churn_features.csv"
SHAP_FIGURE_PATH = _ROOT / "outputs" / "figures" / "churn_shap_summary.png"
COHORT_SAMPLE_SIZE = 5_000
RANDOM_STATE = 42

RISK_BADGE_CSS = """
<style>
.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 0.5rem;
    font-weight: 700;
    font-size: 1.1rem;
    color: #fff;
}
.risk-high { background-color: #D1495B; }
.risk-medium { background-color: #E8A33D; color: #1a1a1a; }
.risk-low { background-color: #2E7D32; }
</style>
"""


@st.cache_resource
def get_bundle():
    return load_model_bundle("behavioral")


@st.cache_data
def load_churn_csv() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def _risk_badge_html(band: str) -> str:
    cls = {"high": "risk-high", "medium": "risk-medium", "low": "risk-low"}.get(
        band, "risk-medium"
    )
    return f'<span class="risk-badge {cls}">{band.upper()} RISK</span>'


def _behavioral_metrics(bundle) -> dict:
    return bundle.metrics.get("lightgbm_calibrated_behavioral", {})


def tab_score_wallet(bundle) -> None:
    st.subheader("Score a wallet from week-1 behavior")
    st.caption(
        "Adjust first-7-day signals below, then score predicted month-3 retention "
        "(any on-chain trade in days 60-120 after first trade)."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        n_trades = st.number_input("Trades in week 1", 1, 500, 5)
        active_days = st.number_input("Active days in week 1", 1, 7, 2)
        n_markets = st.number_input("Distinct markets in week 1", 1, 50, 2)
    with col_b:
        n_categories = st.number_input("Distinct categories in week 1", 1, 4, 1)
        multi_cat = st.checkbox("Multi-category week 1 (2+ categories)", value=False)
        extreme_share = st.slider("Extreme-price trade share", 0.0, 1.0, 0.15, 0.05)
    with col_c:
        median_hour = st.slider("Median trade hour (UTC)", 0, 23, 14)
        first_category = st.selectbox(
            "First-trade category",
            ["politics", "sports", "crypto", "other"],
            index=0,
        )
        made_second = st.checkbox("Made a second trade", value=True)
        hrs_to_2nd = None
        if made_second:
            hrs_to_2nd = st.slider("Hours to second trade", 0, 720, 36)

    if st.button("Score", type="primary"):
        features = {
            "n_trades_w1": int(n_trades),
            "active_days_w1": int(active_days),
            "n_markets_w1": int(n_markets),
            "n_categories_w1": int(n_categories),
            "multi_category_flag": int(multi_cat),
            "extreme_price_share": float(extreme_share),
            "median_hour": float(median_hour),
            "hrs_to_2nd": hrs_to_2nd,
            "has_second_trade": 1 if made_second else 0,
            "first_category": first_category,
        }
        result = score_wallet(features, bundle)

        st.markdown(RISK_BADGE_CSS, unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("P(churn)", f"{result['p_churn']:.1%}")
        with m2:
            st.metric("P(still active @ M3)", f"{result['p_active_m3']:.1%}")
        with m3:
            st.markdown(_risk_badge_html(result["risk_band"]), unsafe_allow_html=True)

        st.markdown("#### Top drivers")
        for reason in result["top_reasons"]:
            arrow = "↓" if "decreases" in reason["direction"] else "↑"
            st.markdown(f"- {arrow} **{reason['plain']}**")


def tab_cohort_view(bundle, df: pd.DataFrame) -> None:
    st.subheader("Cohort view: do risk bands separate real churn?")
    metrics = _behavioral_metrics(bundle)
    st.caption(
        f"Model on held-out test wallets: ROC-AUC **{metrics.get('roc_auc', 0):.3f}**, "
        f"PR-AUC **{metrics.get('pr_auc', 0):.3f}**, "
        f"top-decile lift **{metrics.get('top_decile_lift', 0):.2f}×** "
        f"(from metrics_behavioral.json)."
    )

    sample = df.sample(
        n=min(COHORT_SAMPLE_SIZE, len(df)),
        random_state=RANDOM_STATE,
    )
    scored = score_dataframe(sample, bundle)

    fig = px.histogram(
        scored,
        x="p_churn",
        nbins=40,
        title=f"Predicted P(churn), {len(scored):,} wallet sample",
        labels={"p_churn": "Predicted P(churn)"},
        color_discrete_sequence=["#3A6EA5"],
    )
    fig.update_layout(bargap=0.05, height=360)
    st.plotly_chart(fig, use_container_width=True)

    band_stats = (
        scored.assign(actual_churn=(scored["active_m3"] == 0).astype(int))
        .groupby("risk_band", as_index=False)
        .agg(
            wallets=("wallet", "count"),
            actual_churn_rate=("actual_churn", "mean"),
            mean_p_churn=("p_churn", "mean"),
        )
        .sort_values(
            "risk_band",
            key=lambda s: s.map({"low": 0, "medium": 1, "high": 2}),
        )
    )
    band_stats["actual_churn_rate"] = (100 * band_stats["actual_churn_rate"]).round(1)
    band_stats["mean_p_churn"] = (100 * band_stats["mean_p_churn"]).round(1)
    band_stats = band_stats.rename(
        columns={
            "actual_churn_rate": "actual_churn_%",
            "mean_p_churn": "mean_pred_churn_%",
        }
    )
    st.dataframe(band_stats, hide_index=True, use_container_width=True)

    base = metrics.get("base_churn_rate", scored["p_churn"].mean())
    high_row = band_stats[band_stats["risk_band"] == "high"]
    if not high_row.empty:
        high_actual = high_row["actual_churn_%"].iloc[0] / 100.0
        st.info(
            f"High-risk band actual churn **{high_actual:.1%}** vs "
            f"sample base **{base:.1%}**. Bands carry lift if high > base."
        )


def tab_how_it_works(bundle) -> None:
    st.subheader("How it works")
    m = bundle.metrics
    bm = _behavioral_metrics(bundle)
    n_wallets = m.get("train_size", 0) + m.get("test_size", 0)

    st.markdown(
        f"""
The model predicts **active_m3**: whether a taker wallet makes **any on-chain trade
between days 60 and 120** after its first trade, using **first-7-day behavior only**.

- **Training data:** ~{n_wallets:,} mature taker wallets from
  [Dune](https://dune.com) `polymarket_polygon.market_trades` (12.5% wallet sample).
- **Variant:** behavioral-only (no USD/volume features) so we are not simply flagging whales.
- **Probabilities:** isotonic-calibrated LightGBM; SHAP explains the underlying tree model.

**Held-out test metrics** (from `metrics_behavioral.json`):

| Metric | Value |
|--------|-------|
| ROC-AUC | {bm.get('roc_auc', 0):.3f} |
| PR-AUC | {bm.get('pr_auc', 0):.3f} |
| Top-decile churn precision | {bm.get('top_decile_churn_precision', 0):.1%} |
| Top-decile lift | {bm.get('top_decile_lift', 0):.2f}× |
| Test base churn rate | {m.get('test_base_churn_rate', 0):.1%} |

**Caveats**

- Wallets are not humans (one person may use several addresses).
- Scores are **probabilities**, not verdicts. Use for triage, not punishment.
- Features show **correlation**, not proven causation.
- A growth win requires an **A/B test** (e.g. nudge high-risk week-1 traders) before scaling spend.
        """
    )

    st.markdown("#### Global drivers (SHAP summary)")
    if SHAP_FIGURE_PATH.exists():
        st.image(
            str(SHAP_FIGURE_PATH),
            caption="SHAP beeswarm on 2,000-wallet sample. Behavioral model, no dollar features.",
            use_container_width=True,
        )
    else:
        st.warning(
            f"SHAP plot not found at `{SHAP_FIGURE_PATH}`. "
            "Run `python scripts/10_explain_demo.py` first."
        )


def main() -> None:
    st.set_page_config(
        page_title="Polymarket Churn Radar",
        page_icon="📡",
        layout="wide",
    )
    st.title("Polymarket Churn Radar: predicting month-3 retention from a trader's first week")
    st.caption(
        "Built on on-chain Polymarket taker data via Dune Analytics · behavioral-only churn model"
    )

    bundle = get_bundle()
    df = load_churn_csv()

    tab1, tab2, tab3 = st.tabs(["Score a wallet", "Cohort view", "How it works"])
    with tab1:
        tab_score_wallet(bundle)
    with tab2:
        tab_cohort_view(bundle, df)
    with tab3:
        tab_how_it_works(bundle)


if __name__ == "__main__":
    main()
