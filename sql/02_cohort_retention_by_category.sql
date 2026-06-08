WITH trades AS (
    SELECT
        taker,
        block_time,
        tx_hash,
        condition_id,
        question,
        event_market_name
    FROM polymarket_polygon.market_trades
    WHERE block_time >= TIMESTAMP '2024-01-01'
),
first_times AS (
    SELECT taker, MIN(block_time) AS first_time
    FROM trades
    GROUP BY 1
),
first_trade AS (
    SELECT
        t.taker AS usr,
        t.block_time,
        t.tx_hash,
        t.condition_id,
        t.question,
        t.event_market_name
    FROM trades t
    INNER JOIN first_times ft
        ON t.taker = ft.taker AND t.block_time = ft.first_time
),
first_trade_one AS (
    SELECT *
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY usr
                ORDER BY tx_hash
            ) AS tie_rn
        FROM first_trade
    ) x
    WHERE tie_rn = 1
),
first_with_meta AS (
    SELECT
        f.usr,
        f.block_time,
        f.question,
        f.event_market_name,
        md.tags,
        lower(
            coalesce(f.question, '') || ' ' || coalesce(f.event_market_name, '')
        ) AS market_text
    FROM first_trade_one f
    LEFT JOIN polymarket_polygon.market_details md
        ON cast(f.condition_id AS varchar) = md.condition_id
),
first_cat AS (
    SELECT
        usr,
        date_trunc('month', block_time) AS cohort_month,
        CASE
            WHEN lower(coalesce(tags, '')) LIKE '%politics%'
              OR lower(coalesce(tags, '')) LIKE '%election%'
              OR lower(coalesce(tags, '')) LIKE '%geopolitics%'
              OR lower(coalesce(tags, '')) LIKE '%trump%'
              OR lower(coalesce(tags, '')) LIKE '%impeach%'
              OR lower(coalesce(tags, '')) LIKE '%referendum%'
              OR lower(coalesce(tags, '')) LIKE '%parliament%'
              OR lower(coalesce(tags, '')) LIKE '%senate%'
              OR lower(coalesce(tags, '')) LIKE '%governor%'
              OR lower(coalesce(tags, '')) LIKE '%mayoral%'
              OR lower(coalesce(tags, '')) LIKE '%sotu%'
                THEN 'politics'
            WHEN lower(coalesce(tags, '')) LIKE '%crypto%'
                THEN 'crypto'
            WHEN lower(coalesce(tags, '')) LIKE '%sports%'
                THEN 'sports'
            WHEN regexp_like(
                market_text,
                'president|presidential|election|\belect|trump|harris|biden|kamala|senate|senator|governor|gubernatorial|mayoral|\bmayor|parliament|prime minister|nominee|nomination|primary|caucus|electoral|popular vote|congress|impeach|referendum|chancellor'
            )
                THEN 'politics'
            WHEN regexp_like(
                market_text,
                'bitcoin|\bbtc\b|ethereum|\beth\b|solana|\bsol\b|microstrategy|dogecoin|\bdoge\b|\bxrp\b|all-time high|\bath\b|coinbase|binance|crypto'
            )
                THEN 'crypto'
            WHEN regexp_like(
                market_text,
                '\bvs\.?\b|\bnba\b|\bnfl\b|\bmlb\b|\bnhl\b|\bufc\b|champions league|premier league|la liga|serie a|bundesliga|world cup|super bowl|roland garros|wimbledon|\batp\b|\bwta\b|grand prix|\bfc\b|playoffs|\bfinals?\b|win on \d'
            )
                THEN 'sports'
            ELSE 'other'
        END AS first_category
    FROM first_with_meta
),
monthly AS (
    SELECT DISTINCT
        taker AS usr,
        date_trunc('month', block_time) AS active_month
    FROM polymarket_polygon.market_trades
    WHERE block_time >= TIMESTAMP '2024-01-01'
)
SELECT
    fc.first_category,
    fc.cohort_month,
    date_diff('month', fc.cohort_month, m.active_month) AS months_since,
    COUNT(DISTINCT fc.usr) AS active_users
FROM first_cat fc
JOIN monthly m ON fc.usr = m.usr
GROUP BY 1, 2, 3
ORDER BY 1, 2, 3
