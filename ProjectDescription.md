# Polymarket Trader Retention: Project Description

This project studies how well Polymarket keeps its traders over time using on-chain Polygon trade data via Dune Analytics. It answers two questions: how sticky is growth (do first-time traders come back, and does that vary by entry category?), and can we predict churn early from a wallet's first seven days of behavior?

It has two parts: a **retention teardown** (cohort curves, category breakdowns, and a fact-check of Polymarket's public "~75% return" claim) and a **churn model** with SHAP explainability and a public demo called Churn Radar.

## Context

Polymarket is a prediction market where trades settle on-chain. Each trade has a taker wallet. For growth teams, acquisition is only half the story; retention determines whether marketing spend builds lasting value. Polymarket has cited high return rates (~75%), but "coming back" can mean different things. This project measures retention from chain data with explicit definitions so claims can be compared fairly.

Every analysis treats a taker wallet as a user. That works for on-chain data but has limits: one person may use multiple wallets, and protocol addresses are filtered out. Metrics are wallet-level, useful at platform scale, not for treating individual addresses as verified humans.

## Retention Teardown

This is descriptive analytics: what happened, not what will happen next.

**Cohorts** group wallets by the calendar month of their first trade. All retention analysis is organized around these acquisition months.

**Sequential retention** is the primary metric. Month 0 is 100% by definition. Month N is the share of a cohort still trading N calendar months later. This is the standard SaaS-style cohort curve adapted for on-chain monthly activity.

**Ever-returned** is a looser metric: the wallet was active in at least two distinct calendar months (including the first). It approximates "came back at least once" and is used to benchmark Polymarket's headline return-rate claim, which usually does not match strict sequential month-3 retention.

**First-trade category** (politics, sports, crypto, other) is assigned from market metadata tags, with keyword fallback on market text when tags are missing. Category retention analysis uses both; the churn pipeline uses tags only for query performance.

The analysis focuses on **mature cohorts** (roughly mid-2024 through mid-2025) where enough follow-up months exist. Recent cohorts are excluded because their future retention is not yet observable.

Key patterns the teardown is built to surface: politics-driven acquisition spikes around elections (especially October 2024), different long-run retention by entry category, and a consistent gap between sequential and ever-returned rates (ever-returned is almost always higher).

## Churn Model

This is predictive analytics: estimating month-3 retention from week-one behavior alone.

**Target:** `active_m3`, whether a wallet trades at all between days 60 and 120 after its first trade. Churn means failing that window, not never trading again.

**Features** (first seven days only): trade count, active days, markets and categories explored, multi-category flag, share of extreme-price trades, median trading hour (UTC), time to second trade, and first category. A **behavioral** variant excludes dollar volume so predictions reflect engagement patterns, not whale size. That variant is the primary public model.

**Training:** ~200,000 mature wallets (12.5% sample of the population). Chronological train/test split: earliest cohorts train, latest ~25% test. LightGBM with isotonic calibration; logistic regression as a baseline. Test base churn is ~61%. Behavioral model ROC-AUC ~0.67, PR-AUC ~0.53. Top-decile churn lift ~1.5x base rate, useful for triage not certainty.

**Explainability:** SHAP values on the tree model, translated into plain-language drivers (e.g. few week-one trades, no second trade). Risk bands (low/medium/high) support prioritization, not punishment.

## Churn Radar Demo

A Streamlit app running on precomputed artifacts (no live data pulls). Users can score hypothetical week-one profiles, compare predicted risk bands to actual outcomes on a wallet sample, and review methodology, metrics, and caveats.

## Limitations

Wallets are not humans. SHAP shows correlation, not causation; interventions need A/B tests. Retention numbers shift by definition (sequential vs ever-returned vs the model's day-60-120 window). The churn sample and category heuristics are approximations of full-population truth.

## How It Fits Together

The retention teardown establishes baseline stickiness by cohort and category and tests whether public return claims hold on-chain. The churn model layers a forward-looking signal on the same wallet data: who looks likely to leave by month three, and why. Neither replaces product judgment; both inform it.

## Technical Stack

Python 3.11+, Dune Analytics (SQL on Polygon trade tables), pandas, matplotlib, Plotly, scikit-learn, LightGBM, SHAP, joblib, Streamlit, python-dotenv.
