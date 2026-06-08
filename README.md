# Polymarket Trader Retention

On-chain growth analysis for Polymarket takers: cohort retention by acquisition month and category (Dune), plus a **week-1 churn model** that predicts month-3 retention (`active_m3`) from first-7-day behavior.

## What this repo does

1. **Retention teardown** (Parts 1-4): pull cohort matrices from Dune, segment by first-trade category, chart mature retention, and benchmark Polymarket's "~75% return" claim against ever-returned vs sequential metrics.
2. **Churn model** (Part 5): train a behavioral LightGBM on ~200k wallets (`active_m3` label), with SHAP explainability and a Streamlit demo.
3. **Live demo**: Streamlit Community Cloud app backed by precomputed model artifacts (no Dune API at runtime).

## Requirements

- Python **3.11+**
- [Dune](https://dune.com) API key (for data pulls only; not needed to run the Streamlit app)
- Optional: [uv](https://docs.astral.sh/uv/) for fast installs

## Quick start

```bash
cd polymarket-growth
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e .
```

Copy `.env.example` to `.env` and fill in your Dune API key and query IDs (see below).

Verify connectivity:

```bash
python scripts/00_check_access.py
```

## Project layout

```
polymarket-growth/
  README.md
  requirements.txt          # Streamlit Cloud / minimal app runtime
  pyproject.toml
  .env.example
  app/
    streamlit_app.py          # Churn Radar demo
  src/
    config.py                 # env loading
    data_access.py            # Dune + Gamma clients
    analysis.py               # retention analytics
    churn_features.py         # feature matrix + time split
    churn_explain.py          # scoring + SHAP
  sql/
    01_cohort_retention.sql
    02_cohort_retention_by_category.sql
    03_ever_returned.sql
    04_churn_features.sql
  scripts/
    00_check_access.py
    01_pull_cohort_retention.py
    02_pull_cohort_by_category.py
    03_build_charts.py
    05_pull_ever_returned.py
    07_pull_churn_features.py
    08_train_churn_model.py
    09_train_churn_behavioral.py
    10_explain_demo.py
  models/                     # trained churn artifacts (committed for deploy)
  data/raw/
    churn_features.csv        # committed (~28 MB) for Streamlit deploy
  outputs/figures/
    churn_shap_summary.png    # committed for Streamlit deploy
```

Most other `data/raw/`, `data/processed/`, and `outputs/` paths are gitignored.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DUNE_API_KEY` | Dune Analytics API key |
| `DUNE_CHECK_QUERY_ID` | Connectivity check query |
| `DUNE_COHORT_QUERY_ID` | `sql/01_cohort_retention.sql` |
| `DUNE_COHORT_CAT_QUERY_ID` | `sql/02_cohort_retention_by_category.sql` |
| `DUNE_EVER_RETURNED_QUERY_ID` | `sql/03_ever_returned.sql` |
| `DUNE_CHURN_FEATURES_QUERY_ID` | `sql/04_churn_features.sql` |

Never commit `.env` (it is gitignored).

---

## Part 1: Dune setup and connectivity

1. Create a Dune account and API key: [dune.com/settings/api](https://dune.com/settings/api).
2. Save a small check query in Dune and set `DUNE_CHECK_QUERY_ID` in `.env`.
3. Run `python scripts/00_check_access.py`.

Check query SQL (paste into Dune):

```sql
SELECT event_market_name, question,
       SUM(amount) AS total_volume_usd,
       COUNT(*)    AS num_trades,
       COUNT(DISTINCT taker) AS unique_takers
FROM polymarket_polygon.market_trades
WHERE block_time >= NOW() - INTERVAL '7' DAY
GROUP BY 1, 2
ORDER BY 3 DESC
LIMIT 20;
```

Always filter on `block_time` in production queries against `polymarket_polygon.market_trades`.

---

## Part 2: Cohort retention matrix

1. Paste `sql/01_cohort_retention.sql` into Dune, run on the **small** engine, **Save** (public for API).
2. Add `DUNE_COHORT_QUERY_ID` to `.env`.
3. Pull:

```bash
python scripts/01_pull_cohort_retention.py
```

Writes `data/raw/cohort_retention.csv`.

| Error | Fix |
|-------|-----|
| `403 Forbidden` | Query is unsaved. Open in Dune, click **Save**, use that query_id. |
| Wrong directory | Run from `polymarket-growth/`, not a parent folder. |

---

## Part 3: Retention by first-trade category

Category logic in `sql/02_cohort_retention_by_category.sql`: tags from `market_details`, then keyword fallback on market text.

1. Save `sql/02` as a public Dune query; set `DUNE_COHORT_CAT_QUERY_ID`.
2. Pull:

```bash
python scripts/02_pull_cohort_by_category.py
```

Writes `data/raw/cohort_retention_by_category.csv`.

---

## Part 4: Charts and the ~75% claim

```bash
python scripts/03_build_charts.py
```

Requires Part 2-3 CSVs. Writes `data/processed/*.csv`, `outputs/figures/*.png`, and `outputs/retention_dashboard.html`.

**Ever-returned** (apples-to-apples with "~75% return"):

1. Save `sql/03_ever_returned.sql` in Dune; set `DUNE_EVER_RETURNED_QUERY_ID`.
2. Pull and rebuild charts:

```bash
python scripts/05_pull_ever_returned.py
python scripts/03_build_charts.py
```

---

## Part 5: Churn model (month-3 retention from week 1)

Predicts **active_m3**: any on-chain trade in **days 60-120** after first trade, from week-1 behavior only.

### 1. Pull features from Dune

1. Save `sql/04_churn_features.sql` in Dune (small engine); set `DUNE_CHURN_FEATURES_QUERY_ID`.
2. Pull:

```bash
python scripts/07_pull_churn_features.py
```

Writes `data/raw/churn_features.csv` (~200k wallets, 12.5% sample).

### 2. Train models

Full model (with volume features):

```bash
python scripts/08_train_churn_model.py
```

Behavioral-only robustness variant (no USD features):

```bash
python scripts/09_train_churn_behavioral.py
```

Artifacts land in `models/` (`churn_lgbm_behavioral.joblib` is the primary demo model).

### 3. SHAP demo

```bash
python scripts/10_explain_demo.py
```

Writes `outputs/figures/churn_shap_summary.png` and prints example wallet explanations.

**Held-out test metrics (behavioral model):** ROC-AUC ~0.67, PR-AUC ~0.53, top-decile churn lift ~1.5x (see `models/metrics_behavioral.json`).

---

## Live demo: Churn Radar (Streamlit)

Runs on **precomputed artifacts only** (no Dune API at runtime).

**Local:**

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

**Deploy to [Streamlit Community Cloud](https://share.streamlit.io):**

| Setting | Value |
|---------|--------|
| Main file | `app/streamlit_app.py` |
| Requirements | `requirements.txt` (repo root) |
| Secrets | None required for the demo |

Committed deploy assets: `models/churn_lgbm_behavioral.joblib`, `models/*_behavioral.json`, `data/raw/churn_features.csv`, `outputs/figures/churn_shap_summary.png`.

---

## Troubleshooting

- **Dune 403 on pull:** saved query must be **public** and match the SQL file in `sql/`.
- **Streamlit missing modules:** use root `requirements.txt`, not only `pyproject.toml`.
- **Large CSV push:** `churn_features.csv` is ~28 MB (under GitHub's 100 MB file limit).
