-- ===========================================================
-- CORE ANALYTICAL REPORTS (Proven to Execute Successfully)
-- This file contains only pure SELECT queries (DML) to bypass
-- execution errors related to complex DDL (functions/views).
-- ===========================================================

-- Report 1: Overall Sentiment Performance by Bank (Your Query 1)
-- Purpose: Ranks banks by their average VADER sentiment score,
--          with statistical measures (StdDev, 95% CI).

SELECT
bank_name,
total_reviews,
avg_sentiment AS average_sentiment_score,
sentiment_stddev,
ci_lower_95,
ci_upper_95,
sentiment_rank,
volume_rank,
sentiment_category
FROM (
WITH bank_performance AS (
SELECT
b.bank_name,
COUNT(r.review_id) AS total_reviews,
ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_sentiment,
ROUND(STDDEV(r.sentiment_score)::numeric, 4) AS sentiment_stddev,
ROUND((AVG(r.sentiment_score) - 1.96 * STDDEV(r.sentiment_score) / SQRT(COUNT(r.review_id)))::numeric, 4) AS ci_lower_95,
ROUND((AVG(r.sentiment_score) + 1.96 * STDDEV(r.sentiment_score) / SQRT(COUNT(r.review_id)))::numeric, 4) AS ci_upper_95
FROM
reviews r
INNER JOIN banks b ON r.bank_id = b.bank_id
GROUP BY
b.bank_name
)
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
) AS bank_rankings
ORDER BY
sentiment_rank,
total_reviews DESC;

-- Report 2: Advanced Negative Review Analysis (Your Query 3)
-- Purpose: Pinpoints critical feedback sources with severity levels
--          and identifies priority areas for improvement.

WITH negative_reviews AS (
SELECT
b.bank_name,
r.source,
COUNT(r.review_id) AS total_reviews,
SUM(CASE WHEN r.sentiment_score <= -0.05 THEN 1 ELSE 0 END) AS negative_count,
ROUND(AVG(CASE WHEN r.sentiment_score <= -0.05 THEN r.sentiment_score ELSE NULL END)::numeric, 4) AS avg_negative_sentiment
FROM
reviews r
INNER JOIN banks b ON r.bank_id = b.bank_id
GROUP BY
b.bank_name, r.source
),
bank_totals AS (
SELECT
bank_name,
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
ROUND((nr.negative_count::DECIMAL / NULLIF(nr.total_reviews, 0) * 100)::numeric, 2) AS negative_percentage,
ROUND((nr.negative_count::DECIMAL / NULLIF(bt.bank_negative_reviews, 0) * 100)::numeric, 2) AS bank_negative_share,
nr.avg_negative_sentiment,
CASE
WHEN (nr.negative_count::DECIMAL / NULLIF(nr.total_reviews, 0)) > 0.3 THEN '🔴 High Priority'
WHEN (nr.negative_count::DECIMAL / NULLIF(nr.total_reviews, 0)) > 0.15 THEN '🟡 Medium Priority'
ELSE '🟢 Low Priority'
END AS priority_level
FROM
negative_reviews nr
INNER JOIN bank_totals bt ON nr.bank_name = bt.bank_name
WHERE
nr.negative_count > 0
ORDER BY
negative_percentage DESC,
nr.bank_name,
nr.negative_count DESC;

-- Report 3: Comprehensive Bank Health Dashboard (Your Query 4)
-- Purpose: Single query to get all key metrics for executive dashboard.

WITH bank_metrics AS (
SELECT
b.bank_name,
COUNT(r.review_id) AS total_reviews,
ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_sentiment,
COUNT(CASE WHEN r.sentiment_score > 0.05 THEN 1 END) AS positive_count,
COUNT(CASE WHEN r.sentiment_score < -0.05 THEN 1 END) AS negative_count
FROM
banks b
LEFT JOIN reviews r ON b.bank_id = r.bank_id
GROUP BY
b.bank_name
),
quarterly_trend AS (
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
bm.avg_sentiment,
ROUND((bm.positive_count::DECIMAL / NULLIF(bm.total_reviews, 0) * 100)::numeric, 2) AS positive_percentage,
ROUND((bm.negative_count::DECIMAL / NULLIF(bm.total_reviews, 0) * 100)::numeric, 2) AS negative_percentage,
COALESCE(qt.qoq_trend, 0) AS quarterly_trend_change,
CASE
WHEN bm.avg_sentiment > 0.2 AND qt.qoq_trend > 0 THEN '🟢 Excellent & Improving'
WHEN bm.avg_sentiment > 0.2 THEN '🟢 Excellent'
WHEN bm.avg_sentiment > 0 AND qt.qoq_trend > 0 THEN '🟡 Good & Improving'
WHEN bm.avg_sentiment > 0 THEN '🟡 Good'
WHEN qt.qoq_trend > 0 THEN '🟠 Poor but Improving'
ELSE '🔴 Needs Attention'
END AS health_status
FROM
bank_metrics bm
LEFT JOIN quarterly_trend qt ON bm.bank_name = qt.bank_name
ORDER BY
bm.avg_sentiment DESC;

-- Report 4: Real-time Alerting Query (Your Query 5)
-- Purpose: Identifies banks needing immediate attention (>= 5 recent critical reviews OR avg sentiment < -0.3 in last 7 days).

SELECT
b.bank_name,
COUNT(r.review_id) AS recent_critical_reviews,
ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_recent_sentiment,
MIN(r.sentiment_score) AS worst_recent_sentiment,
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

-- Report 5: Source Performance Comparison (Your Query 6)
-- Purpose: Compares review sources to identify platform-specific issues.

WITH source_metrics AS (
SELECT
r.source,
COUNT(r.review_id) AS total_reviews,
ROUND(AVG(r.sentiment_score)::numeric, 4) AS avg_sentiment,
COUNT(CASE WHEN r.sentiment_score < -0.05 THEN 1 END) AS negative_count
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
RANK() OVER (ORDER BY avg_sentiment DESC) AS sentiment_rank,
RANK() OVER (ORDER BY negative_count::DECIMAL / total_reviews) AS positivity_rank
FROM
source_metrics
ORDER BY
total_reviews DESC;