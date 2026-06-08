WITH monthly AS (
    SELECT taker AS usr, date_trunc('month', block_time) AS m
    FROM polymarket_polygon.market_trades
    WHERE block_time >= TIMESTAMP '2024-01-01'
    GROUP BY 1, 2
),
firsts AS (
    SELECT usr, MIN(m) AS cohort_month, COUNT(*) AS active_months
    FROM monthly
    GROUP BY 1
)
SELECT cohort_month,
       COUNT(*) AS cohort_size,
       COUNT_IF(active_months >= 2) AS ever_returned,
       ROUND(COUNT_IF(active_months >= 2) * 100.0 / COUNT(*), 1) AS ever_returned_pct
FROM firsts
GROUP BY 1
ORDER BY 1
