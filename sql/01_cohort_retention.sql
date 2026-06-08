WITH monthly AS (
    SELECT taker AS usr,
           date_trunc('month', block_time) AS active_month
    FROM polymarket_polygon.market_trades
    WHERE block_time >= TIMESTAMP '2024-01-01'
    GROUP BY 1, 2
),
firsts AS (
    SELECT usr, MIN(active_month) AS cohort_month
    FROM monthly
    GROUP BY 1
)
SELECT f.cohort_month,
       date_diff('month', f.cohort_month, m.active_month) AS months_since,
       COUNT(*) AS active_users
FROM firsts f
JOIN monthly m USING (usr)
GROUP BY 1, 2
ORDER BY 1, 2
