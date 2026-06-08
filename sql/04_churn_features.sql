/*
 * Churn training features: one row per taker wallet (mature cohorts only).
 * SINGLE-SCAN build for the free small engine (2-min limit):
 *   - computes first_ts with one window pass (no repeated scans),
 *   - reads NO text columns from market_trades (category comes from market tags),
 *   - 12.5% wallet sample (last hex char in 0-1).
 *
 * Retention labels (any trade in window after first_ts):
 *   active_m2  = [30d, 60d]   — came back ~month 2
 *   active_m3  = [60d, 120d]  — broad month-3 window
 *   active_m6b = [120d, 180d] — broadened back-half retention
 *   active_m6  = [150d, 180d] — narrow month-6 (original, noisy)
 * Maturity guard: first_ts <= 2025-11-30 so month-6 is fully observable.
 *
 * Note: category is derived from market_details.tags only (~99.99% coverage);
 * the text-regex fallback used in sql/02 is dropped here purely for engine speed.
 */

WITH md_dedup AS (
    -- One row per market so the trade join can't fan out and inflate counts.
    SELECT condition_id, arbitrary(tags) AS tags
    FROM polymarket_polygon.market_details
    GROUP BY condition_id
),
base AS (
    SELECT
        taker AS wallet,
        block_time,
        condition_id,
        amount,
        price,
        MIN(block_time) OVER (PARTITION BY taker) AS first_ts
    FROM polymarket_polygon.market_trades
    WHERE block_time >= TIMESTAMP '2024-01-01'
      AND taker NOT IN (
          from_hex('4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e'),  -- CTF Exchange
          from_hex('c5d563a36ae78145c45a50134d48a1215220f80a'),  -- NegRisk Exchange
          from_hex('d91e80cf2e7be2e162c6513ced06f1dd0da35296'),  -- NegRisk Adapter
          from_hex('4d97dcd97ec945f40cf65f87097ace5ea0476045')   -- CTF
      )
      AND substr(to_hex(taker), -1) IN ('0', '1')  -- 12.5% wallet sample; widen for more data
),
tagged AS (
    SELECT
        b.wallet,
        b.block_time,
        b.first_ts,
        date_trunc('month', b.first_ts) AS cohort_month,
        b.condition_id,
        b.amount,
        b.price,
        (b.block_time BETWEEN b.first_ts AND b.first_ts + INTERVAL '7' DAY) AS is_w1,
        (b.block_time > b.first_ts) AS is_after,
        (b.block_time BETWEEN b.first_ts + INTERVAL '150' DAY
                          AND b.first_ts + INTERVAL '180' DAY) AS is_m6,
        CASE
            WHEN b.block_time BETWEEN b.first_ts AND b.first_ts + INTERVAL '7' DAY THEN
                CASE
                    WHEN lower(coalesce(md.tags, '')) LIKE '%politics%'
                      OR lower(coalesce(md.tags, '')) LIKE '%election%'
                      OR lower(coalesce(md.tags, '')) LIKE '%geopolitics%'
                      OR lower(coalesce(md.tags, '')) LIKE '%trump%'
                      OR lower(coalesce(md.tags, '')) LIKE '%impeach%'
                      OR lower(coalesce(md.tags, '')) LIKE '%referendum%'
                      OR lower(coalesce(md.tags, '')) LIKE '%parliament%'
                      OR lower(coalesce(md.tags, '')) LIKE '%senate%'
                      OR lower(coalesce(md.tags, '')) LIKE '%governor%'
                      OR lower(coalesce(md.tags, '')) LIKE '%mayoral%'
                      OR lower(coalesce(md.tags, '')) LIKE '%sotu%'
                        THEN 'politics'
                    WHEN lower(coalesce(md.tags, '')) LIKE '%crypto%' THEN 'crypto'
                    WHEN lower(coalesce(md.tags, '')) LIKE '%sports%' THEN 'sports'
                    ELSE 'other'
                END
        END AS trade_category
    FROM base b
    LEFT JOIN md_dedup md
        ON cast(b.condition_id AS varchar) = md.condition_id
    WHERE b.first_ts <= TIMESTAMP '2025-11-30'   -- right-censoring guard
),
agg AS (
    SELECT
        wallet,
        cohort_month,
        SUM(CASE WHEN is_w1 THEN 1 ELSE 0 END) AS n_trades_w1,
        COUNT(DISTINCT CASE WHEN is_w1 THEN date(block_time) END) AS active_days_w1,
        COUNT(DISTINCT CASE WHEN is_w1 THEN condition_id END) AS n_markets_w1,
        COUNT(DISTINCT trade_category) AS n_categories_w1,
        AVG(CASE WHEN is_w1 THEN amount END) AS avg_usd,
        MAX(CASE WHEN is_w1 THEN amount END) AS max_usd,
        SUM(CASE WHEN is_w1 THEN amount ELSE 0 END) AS total_usd,
        AVG(CASE WHEN is_w1 THEN (CASE WHEN price < 0.10 OR price > 0.90 THEN 1.0 ELSE 0.0 END) END) AS extreme_price_share,
        approx_percentile(CASE WHEN is_w1 THEN hour(block_time) END, 0.5) AS median_hour,
        min_by(trade_category, block_time) AS first_category,
        MIN(CASE WHEN is_after THEN date_diff('hour', first_ts, block_time) END) AS hrs_to_2nd,
        MAX(CASE WHEN block_time BETWEEN first_ts + INTERVAL '30' DAY
                                 AND first_ts + INTERVAL '60' DAY THEN 1 ELSE 0 END) AS active_m2,
        MAX(CASE WHEN block_time BETWEEN first_ts + INTERVAL '60' DAY
                                 AND first_ts + INTERVAL '120' DAY THEN 1 ELSE 0 END) AS active_m3,
        MAX(CASE WHEN block_time BETWEEN first_ts + INTERVAL '120' DAY
                                 AND first_ts + INTERVAL '180' DAY THEN 1 ELSE 0 END) AS active_m6b,
        MAX(CASE WHEN is_m6 THEN 1 ELSE 0 END) AS active_m6
    FROM tagged
    GROUP BY wallet, cohort_month
)
SELECT
    cast(wallet AS varchar) AS wallet,
    cohort_month,
    COALESCE(first_category, 'other') AS first_category,
    n_trades_w1,
    active_days_w1,
    n_markets_w1,
    n_categories_w1,
    IF(n_categories_w1 >= 2, 1, 0) AS multi_category_flag,
    COALESCE(avg_usd, 0.0) AS avg_usd,
    COALESCE(max_usd, 0.0) AS max_usd,
    total_usd,
    COALESCE(extreme_price_share, 0.0) AS extreme_price_share,
    median_hour,
    hrs_to_2nd,
    active_m2,
    active_m3,
    active_m6b,
    active_m6
FROM agg