# Polymarket Trader Retention

On-chain growth analysis for Polymarket takers. We pull cohort retention from Dune, break it down by first-trade category, and train a week-1 churn model that predicts month-3 retention.

**Links**

- [Presentation](https://drive.google.com/file/d/18Q7KuzhSN7wT4w2mJkLEkvF6vc8qwqTt/view?usp=sharing)
- [Final Report](https://drive.google.com/file/d/1Ho1HvbCvC2JqmMP27qa4qp_xZgL6YPVG/view?usp=sharing)

---

## What this is

Polymarket has talked publicly about high trader return rates (around 75%). This repo measures retention from chain data with explicit definitions, so you can compare apples to apples.

There are two main pieces:

1. **Retention teardown (Parts 1–4)**  
   Cohort matrices, category splits, charts, and a look at how "ever returned" compares to stricter sequential retention.

2. **Churn model (Part 5)**  
   A behavioral LightGBM trained on ~200k wallets. It predicts `active_m3` (any trade between days 60 and 120 after first trade) using only the first seven days of behavior. SHAP explains the scores, and there is a Streamlit demo called **Churn Radar**.

Everything treats a taker wallet as a user. That works well at platform scale, but one person can have multiple wallets, so take individual scores as triage signals, not identity.

---

## Requirements

- Python **3.11+**
- A [Dune](https://dune.com) API key (only for pulling fresh data; the Streamlit app does not need it)
- Optional: [uv](https://docs.astral.sh/uv/) if you prefer it over pip

---

## Quick start

```bash
git clone <your-repo-url>
cd polymarket-trader-retention

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e .
```

Copy `.env.example` to `.env` and fill in your Dune API key and query IDs (see below).

Check that everything connects:

```bash
python scripts/00_check_access.py
```

To run the churn demo without touching Dune:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

---

## Project layout

```
polymarket-trader-retention/
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
  models/                     # trained artifacts (committed for deploy)
  data/raw/
    churn_features.csv        # committed (~28 MB) for Streamlit deploy
  outputs/figures/
    churn_shap_summary.png    # committed for Streamlit deploy
```

Most other paths under `data/raw/`, `data/processed/`, and `outputs/` are gitignored. You generate them by running the scripts.

---

## Environment variables

| Variable | Purpose |
|----------|---------|
| `DUNE_API_KEY` | Dune Analytics API key |
| `DUNE_CHECK_QUERY_ID` | Small connectivity check query |
| `DUNE_COHORT_QUERY_ID` | `sql/01_cohort_retention.sql` |
| `DUNE_COHORT_CAT_QUERY_ID` | `sql/02_cohort_retention_by_category.sql` |
| `DUNE_EVER_RETURNED_QUERY_ID` | `sql/03_ever_returned.sql` |
| `DUNE_CHURN_FEATURES_QUERY_ID` | `sql/04_churn_features.sql` |

Never commit `.env`. It is gitignored.

---

## Part 1: Dune setup

1. Create a Dune account and grab an API key from [dune.com/settings/api](https://dune.com/settings/api).
2. Save a small check query in Dune and put its ID in `DUNE_CHECK_QUERY_ID`.
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

Always filter on `block_time` when querying `polymarket_polygon.market_trades`.

---

## Part 2: Cohort retention matrix

1. Paste `sql/01_cohort_retention.sql` into Dune, run on the **small** engine, and **Save** the query (public if you want API access).
2. Add `DUNE_COHORT_QUERY_ID` to `.env`.
3. Pull:

```bash
python scripts/01_pull_cohort_retention.py
```

Writes `data/raw/cohort_retention.csv`.

**Common issues**

| Error | Fix |
|-------|-----|
| `403 Forbidden` | Query is unsaved. Open it in Dune, click **Save**, use that query ID. |
| Wrong output path | Run from the repo root, not a parent folder. |

---

## Part 3: Retention by first-trade category

Category logic lives in `sql/02_cohort_retention_by_category.sql`: tags from `market_details`, then keyword fallback on market text.

1. Save `sql/02` as a public Dune query. Set `DUNE_COHORT_CAT_QUERY_ID`.
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

Needs the Part 2 and Part 3 CSVs. Writes `data/processed/*.csv`, `outputs/figures/*.png`, and `outputs/retention_dashboard.html`.

**Ever-returned** (closer to what headline "return" claims usually mean):

1. Save `sql/03_ever_returned.sql` in Dune. Set `DUNE_EVER_RETURNED_QUERY_ID`.
2. Pull and rebuild:

```bash
python scripts/05_pull_ever_returned.py
python scripts/03_build_charts.py
```

---

## Part 5: Churn model

The model predicts **active_m3**: did this wallet trade at all between days 60 and 120 after its first trade? Features come from week 1 only.

### Pull features

1. Save `sql/04_churn_features.sql` in Dune (small engine). Set `DUNE_CHURN_FEATURES_QUERY_ID`.
2. Pull:

```bash
python scripts/07_pull_churn_features.py
```

Writes `data/raw/churn_features.csv` (~200k wallets, 12.5% sample).

### Train

Full model (includes USD volume features):

```bash
python scripts/08_train_churn_model.py
```

Behavioral-only variant (no dollar features, used in the demo):

```bash
python scripts/09_train_churn_behavioral.py
```

Artifacts land in `models/`. The demo uses `churn_lgbm_behavioral.joblib`.

### SHAP demo

```bash
python scripts/10_explain_demo.py
```

Writes `outputs/figures/churn_shap_summary.png` and prints example wallet explanations.

**Held-out test metrics (behavioral model):** ROC-AUC ~0.67, PR-AUC ~0.53, top-decile churn lift ~1.5×. See `models/metrics_behavioral.json` for the full numbers.

---

## Churn Radar (Streamlit)

The app runs on precomputed artifacts. No Dune API key at runtime.

**Local**

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

**Deploy to [Streamlit Community Cloud](https://share.streamlit.io)**

| Setting | Value |
|---------|--------|
| Main file | `app/streamlit_app.py` |
| Requirements | `requirements.txt` (repo root) |
| Secrets | None required |

Committed deploy assets: `models/churn_lgbm_behavioral.joblib`, `models/*_behavioral.json`, `data/raw/churn_features.csv`, `outputs/figures/churn_shap_summary.png`.

---

## Troubleshooting

- **Dune 403 on pull:** the saved query must be **public** and match the SQL in `sql/`.
- **Streamlit import errors:** install from root `requirements.txt`, not only `pyproject.toml`.
- **Large CSV in git:** `churn_features.csv` is ~28 MB, under GitHub's 100 MB file limit.

---

## A note on metrics

"Retention" means different things depending on how you count:

- **Sequential:** active in exactly month N after first trade (stricter, cohort-curve style).
- **Ever returned:** active in at least two calendar months (looser, closer to headline return claims).
- **active_m3:** any trade in days 60–120 (what the churn model predicts).

Pick the definition that matches the question you are asking.
