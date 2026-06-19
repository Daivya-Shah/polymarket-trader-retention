# Polymarket Trader Retention

## Overview

This project analyzes trader retention on Polymarket, a prediction market where users bet on real-world outcomes. Using on-chain trade data from Polygon, it asks three questions: how sticky are new traders, does retention differ by what someone trades first, and can a trader's first-week behavior predict whether they will still be active around month three?

Each unique taker wallet counts as one trader. That is a simplification (wallets are not people), but it is the most reproducible unit available from public data.

The work splits into two parts. The first is descriptive cohort retention analysis, segmented by market category and careful about how "retention" is defined. The second is a machine learning churn model that scores month-three risk from week-one behavior alone, paired with an interactive demo that turns scores into plain-English explanations.

## Why This Matters

Polymarket grows in bursts, not steadily. Major events (the October 2024 election, March 2026 spikes) can pull hundreds of thousands of first-time traders into a single month. A headline-driven politics bettor may behave nothing like a quiet-month sports explorer.

Growth teams need to know not just how many users arrive, but who stays, through which categories, and which new traders are already drifting away while intervention is still possible. Public claims that most users "return" also mean little until you define what "return" means. This project replaces intuition with reproducible numbers.

## Retention Analysis

Traders are grouped into monthly cohorts by first trade date. For each cohort, we count how many original members made at least one trade in each subsequent calendar month, producing the standard retention curve.

Analysis focuses on mature cohorts with enough observable future months. Recent cohorts are excluded from long-horizon averages to avoid right-censoring bias.

**Sequential vs. ever-returned.** These two metrics must not be conflated.

Sequential retention is strict: of traders who joined in month M, what fraction were active in exactly month M+N? Month-three sequential retention of 20% means only one in five traded again during their third month, regardless of activity in between.

Ever-returned is lenient: what fraction were active in at least two different calendar months at any point? A trader who trades once in month one and once in month seven counts as returned.

Other sources have cited roughly 75% user return for Polymarket (not an official Polymarket figure). Ever-returned rates on mature cohorts land closer to that number. Sequential month-three and month-six retention are substantially lower. Both are valid; they answer different questions.

**Category segmentation.** Each wallet is labeled by its first trade: politics, sports, crypto, or other (from market metadata tags, with keyword fallback). Politics drives the largest acquisition volume, especially around elections, but consistently trails sports and crypto on retention at months one, three, and six. That holds both in mature cohort averages and within the October 2024 election wave. Acquisition volume and acquisition quality are not the same thing.

**Event-driven growth.** New trader counts spike around major events. Retention strategy cannot treat all new users as interchangeable; cohort vintage and entry category both matter.

## Churn Prediction

The model predicts **active_m3**: whether a wallet trades at least once between days 60 and 120 after its first trade. No trade in that window counts as churn. All inputs come from the first seven days: trade count, active days, markets and categories explored, multi-category flag, extreme-price trade share, typical trading hour, time to second trade (or none), and first-trade category.

Every feature is observable within a week, so scores work as early warnings for onboarding or outreach rather than post-mortems.

Two variants are trained. The full model includes week-one dollar volume. The **behavioral-only** model drops volume features and relies on engagement patterns alone. It performs nearly as well (ROC-AUC ~0.67, PR-AUC ~0.53), confirming the signal is not just "whale vs. small trader." The behavioral variant powers the demo.

LightGBM is the primary classifier, with isotonic calibration and chronological train/test splits (earliest cohorts train, latest test) to prevent temporal leakage. Base churn on the test set is ~61%. The highest-risk decile reaches ~91% actual churn, roughly 1.5x lift over random. Useful for triage, not as automatic verdicts.

Higher churn risk correlates with: one trade in week one, one active day, no second trade, single-category browsing, high extreme-price share. Lower risk correlates with: multiple active days, quick second trade, cross-market exploration. SHAP values explain individual predictions in plain language. Scores map to high, medium, and low risk bands (70% and 40% P(churn) cutoffs).

## Interactive Demo

**Polymarket Churn Radar** is a Streamlit app with three views: score hypothetical week-one profiles with risk badges and top reasons, validate that risk bands separate real churners from retainers on held-out wallets, and review model methodology plus a global SHAP summary. It runs on pre-trained artifacts with no live database queries.

## Limitations

Wallets are not humans. Only taker-side on-chain activity is captured, not browsing or maker flow. Feature-outcome links are correlational; interventions need A/B tests before scaling. Training uses a 12.5% wallet sample due to query engine limits. Models will drift as acquisition mix shifts with future events.

## Technical Stack

Python 3.11+, Dune Analytics (SQL + dune-client API), pandas, matplotlib, plotly, LightGBM, scikit-learn, SHAP, joblib, Streamlit, python-dotenv. Polymarket Gamma API via requests is used only for connectivity checks.
