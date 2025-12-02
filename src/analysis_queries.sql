-- ===========================================================
-- OPTIMIZED ANALYTICAL QUERIES FOR BANK REVIEWS DATABASE
-- Features: Performance indexing, CTEs, window functions, and parameterization
-- ===========================================================

-- -----------------------------------------------------------
-- Create performance indexes if not exist (Run once)
-- -----------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_reviews_bank_id_sentiment ON reviews(bank_id, sentiment_score);
CREATE INDEX IF NOT EXISTS idx_reviews_sentiment_date ON reviews(sentiment_score, review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_bank_date ON reviews(bank_id, review_date);
CREATE INDEX IF NOT EXISTS idx_reviews_source_sentiment ON reviews(source, sentiment_score);
CREATE INDEX IF NOT EXISTS idx_banks_bank_name ON banks(bank_name);

-- -----------------------------------------------------------
-- Query 1: Overall Sentiment Performance by Bank
-- Purpose: Ranks banks by their average VADER sentiment score,
--          providing a clear performance metric with statistics.
-- Optimizations: 
--   - Uses materialized CTE for bank-level aggregations
--   - Includes statistical measures (std dev, confidence intervals)
--   - Window function for ranking
-- -----------------------------------------------------------
WITH bank_performance AS (
    SELECT
        b.bank_name,
        COUNT(r.review_id) AS total_reviews,
        ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_sentiment,
        ROUND(STDDEV(r.sentiment_score)::numeric, 4) AS sentiment_stddev,
        MIN(r.sentiment_score) AS min_sentiment,
        MAX(r.sentiment_score) AS max_sentiment,
        ROUND((AVG(r.sentiment_score) - 1.96 * STDDEV(r.sentiment_score) / SQRT(COUNT(r.review_id)))::numeric, 4) AS ci_lower_95,
        ROUND((AVG(r.sentiment_score) + 1.96 * STDDEV(r.sentiment_score) / SQRT(COUNT(r.review_id)))::numeric, 4) AS ci_upper_95
    FROM
        reviews r
    INNER JOIN banks b ON r.bank_id = b.bank_id
    GROUP BY
        b.bank_name
),
bank_rankings AS (
    SELECT
        *,
        RANK() OVER (ORDER BY avg_sentiment DESC) AS sentiment_rank,
        RANK() OVER (ORDER BY total_reviews DESC) AS volume_rank,
        CASE
            WHEN avg_sentiment > 0.25 THEN 'Excellent'
            WHEN avg_sentiment > 0.05 THEN 'Good'
            WHEN avg_sentiment > -0.05 THEN 'Neutral'
            WHEN avg_sentiment > -0.25 THEN 'Poor'
            ELSE 'Critical'
        END AS sentiment_category
    FROM
        bank_performance
)
SELECT
    bank_name,
    total_reviews,
    avg_sentiment AS average_sentiment_score,
    sentiment_stddev,
    min_sentiment,
    max_sentiment,
    ci_lower_95,
    ci_upper_95,
    sentiment_rank,
    volume_rank,
    sentiment_category
FROM
    bank_rankings
ORDER BY
    sentiment_rank,
    total_reviews DESC;

-- -----------------------------------------------------------
-- Query 2: Dynamic Monthly Sentiment Trend Analysis
-- Purpose: Provides month-by-month sentiment trends for ANY bank
--          with month-over-month change calculations
-- Optimizations:
--   - Parameterized approach using function
--   - Calculates MoM changes
--   - Includes trend direction indicators
-- -----------------------------------------------------------
CREATE OR REPLACE FUNCTION get_bank_monthly_trend(p_bank_name VARCHAR)
RETURNS TABLE (
    bank_name VARCHAR,
    review_month DATE,
    month_name VARCHAR,
    monthly_review_count INTEGER,
    avg_monthly_sentiment NUMERIC(6,4),
    previous_month_sentiment NUMERIC(6,4),
    mom_change NUMERIC(6,4),
    trend_direction VARCHAR(10)
) AS $$
BEGIN
    RETURN QUERY
    WITH monthly_data AS (
        SELECT
            b.bank_name,
            DATE_TRUNC('month', r.review_date)::DATE AS review_month,
            TO_CHAR(DATE_TRUNC('month', r.review_date), 'YYYY-MM') AS month_name,
            COUNT(r.review_id) AS review_count,
            ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_sentiment
        FROM
            reviews r
        INNER JOIN banks b ON r.bank_id = b.bank_id
        WHERE
            b.bank_name = p_bank_name
            AND r.review_date >= CURRENT_DATE - INTERVAL '12 months'  -- Last 12 months for relevance
        GROUP BY
            b.bank_name, DATE_TRUNC('month', r.review_date)
    ),
    with_lag AS (
        SELECT
            *,
            LAG(avg_sentiment) OVER (PARTITION BY bank_name ORDER BY review_month) AS prev_sentiment
        FROM
            monthly_data
    )
    SELECT
        bank_name,
        review_month,
        month_name,
        review_count::INTEGER,
        avg_sentiment,
        prev_sentiment,
        ROUND((avg_sentiment - COALESCE(prev_sentiment, avg_sentiment))::numeric, 4) AS mom_change,
        CASE
            WHEN avg_sentiment - COALESCE(prev_sentiment, avg_sentiment) > 0.05 THEN '↑ Improving'
            WHEN avg_sentiment - COALESCE(prev_sentiment, avg_sentiment) < -0.05 THEN '↓ Declining'
            ELSE '→ Stable'
        END AS trend_direction
    FROM
        with_lag
    ORDER BY
        review_month DESC;
END;
$$ LANGUAGE plpgsql;

-- Example usage for any bank:
SELECT * FROM get_bank_monthly_trend('Bank of Abyssinia');
-- SELECT * FROM get_bank_monthly_trend('CBE');
-- SELECT * FROM get_bank_monthly_trend('Dashen');

-- For quick top 5 months for a specific bank:
SELECT * FROM get_bank_monthly_trend('Bank of Abyssinia')
WHERE monthly_review_count > 10
ORDER BY avg_monthly_sentiment DESC
LIMIT 5;

-- -----------------------------------------------------------
-- Query 3: Advanced Negative Review Analysis
-- Purpose: Pinpoints critical feedback sources with severity levels
--          and identifies priority areas for improvement
-- Optimizations:
--   - Multiple severity tiers
--   - Calculates negative review percentage
--   - Includes performance benchmarking
-- -----------------------------------------------------------
WITH negative_reviews AS (
    SELECT
        b.bank_name,
        r.source,
        COUNT(r.review_id) AS total_reviews,
        SUM(CASE WHEN r.sentiment_score <= -0.05 THEN 1 ELSE 0 END) AS negative_count,
        SUM(CASE WHEN r.sentiment_score <= -0.25 THEN 1 ELSE 0 END) AS severe_count,
        SUM(CASE WHEN r.sentiment_score <= -0.5 THEN 1 ELSE 0 END) AS critical_count,
        ROUND(AVG(CASE WHEN r.sentiment_score <= -0.05 THEN r.sentiment_score ELSE NULL END)::numeric, 4) AS avg_negative_sentiment,
        ROUND(AVG(r.rating)::numeric, 2) AS avg_rating
    FROM
        reviews r
    INNER JOIN banks b ON r.bank_id = b.bank_id
    GROUP BY
        b.bank_name, r.source
),
bank_totals AS (
    SELECT
        bank_name,
        SUM(total_reviews) AS bank_total_reviews,
        SUM(negative_count) AS bank_negative_reviews
    FROM
        negative_reviews
    GROUP BY
        bank_name
)
SELECT
    nr.bank_name,
    nr.source,
    nr.total_reviews,
    nr.negative_count,
    nr.severe_count,
    nr.critical_count,
    ROUND((nr.negative_count::DECIMAL / NULLIF(nr.total_reviews, 0) * 100)::numeric, 2) AS negative_percentage,
    ROUND((nr.negative_count::DECIMAL / NULLIF(bt.bank_negative_reviews, 0) * 100)::numeric, 2) AS bank_negative_share,
    nr.avg_negative_sentiment,
    nr.avg_rating,
    CASE
        WHEN (nr.negative_count::DECIMAL / NULLIF(nr.total_reviews, 0)) > 0.3 THEN '🔴 High Priority'
        WHEN (nr.negative_count::DECIMAL / NULLIF(nr.total_reviews, 0)) > 0.15 THEN '🟡 Medium Priority'
        ELSE '🟢 Low Priority'
    END AS priority_level,
    RANK() OVER (PARTITION BY nr.bank_name ORDER BY nr.negative_count DESC) AS negative_rank_in_bank
FROM
    negative_reviews nr
INNER JOIN bank_totals bt ON nr.bank_name = bt.bank_name
WHERE
    nr.negative_count > 0
ORDER BY
    negative_percentage DESC,
    nr.bank_name,
    nr.negative_count DESC;

-- -----------------------------------------------------------
-- Query 4: Comprehensive Bank Health Dashboard (Bonus)
-- Purpose: Single query to get all key metrics for executive dashboard
-- -----------------------------------------------------------
WITH bank_metrics AS (
    SELECT
        b.bank_name,
        COUNT(r.review_id) AS total_reviews,
        COUNT(DISTINCT DATE_TRUNC('month', r.review_date)) AS active_months,
        ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_sentiment,
        ROUND(AVG(r.rating)::numeric, 2) AS avg_rating,
        COUNT(CASE WHEN r.sentiment_score > 0.05 THEN 1 END) AS positive_count,
        COUNT(CASE WHEN r.sentiment_score < -0.05 THEN 1 END) AS negative_count,
        COUNT(CASE WHEN r.sentiment_score BETWEEN -0.05 AND 0.05 THEN 1 END) AS neutral_count,
        MIN(r.review_date) AS first_review_date,
        MAX(r.review_date) AS latest_review_date,
        COUNT(DISTINCT r.source) AS source_count
    FROM
        banks b
    LEFT JOIN reviews r ON b.bank_id = r.bank_id
    GROUP BY
        b.bank_name
),
monthly_trend AS (
    SELECT
        b.bank_name,
        ROUND(
            (AVG(CASE WHEN r.review_date >= CURRENT_DATE - INTERVAL '3 months' THEN r.sentiment_score END) -
             AVG(CASE WHEN r.review_date < CURRENT_DATE - INTERVAL '3 months' AND 
                         r.review_date >= CURRENT_DATE - INTERVAL '6 months' THEN r.sentiment_score END))::numeric, 4
        ) AS qoq_trend
    FROM
        banks b
    LEFT JOIN reviews r ON b.bank_id = r.bank_id
    WHERE
        r.review_date >= CURRENT_DATE - INTERVAL '6 months'
    GROUP BY
        b.bank_name
)
SELECT
    bm.bank_name,
    bm.total_reviews,
    bm.active_months,
    bm.avg_sentiment,
    bm.avg_rating,
    ROUND((bm.positive_count::DECIMAL / NULLIF(bm.total_reviews, 0) * 100)::numeric, 2) AS positive_percentage,
    ROUND((bm.negative_count::DECIMAL / NULLIF(bm.total_reviews, 0) * 100)::numeric, 2) AS negative_percentage,
    ROUND((bm.neutral_count::DECIMAL / NULLIF(bm.total_reviews, 0) * 100)::numeric, 2) AS neutral_percentage,
    COALESCE(mt.qoq_trend, 0) AS quarterly_trend,
    bm.first_review_date,
    bm.latest_review_date,
    bm.source_count,
    CASE
        WHEN bm.avg_sentiment > 0.2 AND mt.qoq_trend > 0 THEN '🟢 Excellent & Improving'
        WHEN bm.avg_sentiment > 0.2 THEN '🟢 Excellent'
        WHEN bm.avg_sentiment > 0 AND mt.qoq_trend > 0 THEN '🟡 Good & Improving'
        WHEN bm.avg_sentiment > 0 THEN '🟡 Good'
        WHEN mt.qoq_trend > 0 THEN '🟠 Poor but Improving'
        ELSE '🔴 Needs Attention'
    END AS health_status
FROM
    bank_metrics bm
LEFT JOIN monthly_trend mt ON bm.bank_name = mt.bank_name
ORDER BY
    bm.avg_sentiment DESC,
    bm.total_reviews DESC;

-- -----------------------------------------------------------
-- Query 5: Real-time Alerting Query (For Monitoring)
-- Purpose: Identifies banks needing immediate attention
-- -----------------------------------------------------------
SELECT
    b.bank_name,
    COUNT(r.review_id) AS recent_negative_reviews,
    ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_recent_sentiment,
    MIN(r.sentiment_score) AS worst_recent_sentiment,
    COUNT(DISTINCT r.source) AS sources_with_issues,
    ARRAY_AGG(DISTINCT r.source) AS problem_sources
FROM
    reviews r
INNER JOIN banks b ON r.bank_id = b.bank_id
WHERE
    r.sentiment_score < -0.2
    AND r.review_date >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY
    b.bank_name
HAVING
    COUNT(r.review_id) >= 5
    OR AVG(r.sentiment_score) < -0.3
ORDER BY
    COUNT(r.review_id) DESC,
    AVG(r.sentiment_score)
LIMIT 10;

-- -----------------------------------------------------------
-- Query 6: Source Performance Comparison (Bonus)
-- Purpose: Compares review sources to identify platform-specific issues
-- -----------------------------------------------------------
WITH source_metrics AS (
    SELECT
        r.source,
        COUNT(r.review_id) AS total_reviews,
        ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_sentiment,
        ROUND(AVG(r.rating)::numeric, 2) AS avg_rating,
        COUNT(CASE WHEN r.sentiment_score < -0.05 THEN 1 END) AS negative_count,
        COUNT(DISTINCT b.bank_name) AS banks_represented,
        MIN(r.review_date) AS first_review,
        MAX(r.review_date) AS latest_review
    FROM
        reviews r
    INNER JOIN banks b ON r.bank_id = b.bank_id
    GROUP BY
        r.source
    HAVING
        COUNT(r.review_id) >= 10
)
SELECT
    source,
    total_reviews,
    ROUND((negative_count::DECIMAL / total_reviews * 100)::numeric, 2) AS negative_percentage,
    avg_sentiment,
    avg_rating,
    banks_represented,
    first_review,
    latest_review,
    RANK() OVER (ORDER BY avg_sentiment DESC) AS sentiment_rank,
    RANK() OVER (ORDER BY negative_count::DECIMAL / total_reviews) AS positivity_rank
FROM
    source_metrics
ORDER BY
    total_reviews DESC;

-- ===========================================================
-- VIEW CREATIONS FOR EASY ACCESS
-- ===========================================================

-- View 1: Daily sentiment summary (for time-series dashboards)
CREATE OR REPLACE VIEW daily_sentiment_summary AS
SELECT
    r.review_date,
    b.bank_name,
    COUNT(r.review_id) AS daily_reviews,
    ROUND(AVG(r.sentiment_score)::numeric, 4) AS daily_avg_sentiment,
    ROUND(AVG(r.rating)::numeric, 2) AS daily_avg_rating,
    COUNT(CASE WHEN r.sentiment_score > 0.05 THEN 1 END) AS positive_reviews,
    COUNT(CASE WHEN r.sentiment_score < -0.05 THEN 1 END) AS negative_reviews
FROM
    reviews r
INNER JOIN banks b ON r.bank_id = b.bank_id
GROUP BY
    r.review_date, b.bank_name;

-- View 2: Bank performance summary (for executive reporting)
CREATE OR REPLACE VIEW bank_performance_summary AS
SELECT
    b.bank_name,
    COUNT(r.review_id) AS total_reviews,
    ROUND(AVG(r.sentiment_score)::numeric, 4) AS overall_sentiment,
    ROUND(AVG(r.rating)::numeric, 2) AS overall_rating,
    ROUND(STDDEV(r.sentiment_score)::numeric, 4) AS sentiment_volatility,
    COUNT(DISTINCT DATE_TRUNC('month', r.review_date)) AS active_months,
    MIN(r.review_date) AS first_review,
    MAX(r.review_date) AS latest_review,
    COUNT(DISTINCT r.source) AS sources_count
FROM
    banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY
    b.bank_name;

-- ===========================================================
-- MATERIALIZED VIEW FOR HEAVY ANALYTICS (Refresh as needed)
-- ===========================================================

CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_bank_performance AS
SELECT
    DATE_TRUNC('month', r.review_date)::DATE AS month,
    b.bank_name,
    COUNT(r.review_id) AS monthly_reviews,
    ROUND(AVG(r.sentiment_score)::numeric, 4) AS monthly_sentiment,
    ROUND(AVG(r.rating)::numeric, 2) AS monthly_rating,
    COUNT(CASE WHEN r.sentiment_score < -0.05 THEN 1 END) AS negative_count,
    ROUND((COUNT(CASE WHEN r.sentiment_score < -0.05 THEN 1 END)::DECIMAL / 
           COUNT(r.review_id) * 100)::numeric, 2) AS negative_percentage
FROM
    reviews r
INNER JOIN banks b ON r.bank_id = b.bank_id
GROUP BY
    DATE_TRUNC('month', r.review_date), b.bank_name
WITH DATA;

CREATE INDEX IF NOT EXISTS idx_monthly_bank_performance_month ON monthly_bank_performance(month);
CREATE INDEX IF NOT EXISTS idx_monthly_bank_performance_bank ON monthly_bank_performance(bank_name);

-- Function to refresh materialized view
CREATE OR REPLACE FUNCTION refresh_monthly_performance()
RETURNS VOID AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY monthly_bank_performance;
END;
$$ LANGUAGE plpgsql;

-- ===========================================================
-- SAMPLE EXECUTION SCRIPTS
-- ===========================================================

-- To get top performing banks this month:
SELECT 
    bank_name,
    monthly_sentiment,
    monthly_reviews,
    negative_percentage
FROM monthly_bank_performance 
WHERE month = DATE_TRUNC('month', CURRENT_DATE)
ORDER BY monthly_sentiment DESC;

-- To get banks with deteriorating sentiment:
SELECT
    curr.bank_name,
    curr.monthly_sentiment AS current_sentiment,
    prev.monthly_sentiment AS previous_sentiment,
    ROUND((curr.monthly_sentiment - prev.monthly_sentiment)::numeric, 4) AS sentiment_change,
    curr.negative_percentage AS current_negative_pct
FROM monthly_bank_performance curr
LEFT JOIN monthly_bank_performance prev 
    ON curr.bank_name = prev.bank_name 
    AND curr.month = prev.month + INTERVAL '1 month'
WHERE curr.month = DATE_TRUNC('month', CURRENT_DATE)
    AND (curr.monthly_sentiment - prev.monthly_sentiment) < -0.1
ORDER BY sentiment_change;

-- ===========================================================
-- PERFORMANCE OPTIMIZATION QUERIES
-- ===========================================================

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM bank_performance_summary;

-- Check index usage
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM
    pg_stat_user_indexes
WHERE
    tablename IN ('reviews', 'banks')
ORDER BY
    idx_scan DESC;

-- Table statistics
SELECT
    schemaname,
    tablename,
    n_live_tup AS row_count,
    pg_size_pretty(pg_total_relation_size(schemaname || '.' || tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname || '.' || tablename)) AS table_size,
    pg_size_pretty(pg_indexes_size(schemaname || '.' || tablename)) AS index_size
FROM
    pg_stat_user_tables
WHERE
    tablename IN ('reviews', 'banks');