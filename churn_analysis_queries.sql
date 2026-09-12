-- ============================================================
-- ChurnGuard AI - SQL Business Analytics
-- Target table: churn_customers (load data/raw_churn.csv into this)
-- Works on PostgreSQL / MySQL / SQLite with minor syntax tweaks
-- ============================================================

-- 1. Overall churn rate
SELECT
    COUNT(*)                                   AS total_customers,
    SUM(Exited)                                AS churned_customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)   AS churn_rate_pct
FROM churn_customers;

-- 2. Churn by country
SELECT
    Geography,
    COUNT(*)                                   AS customers,
    SUM(Exited)                                AS churned,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)   AS churn_rate_pct
FROM churn_customers
GROUP BY Geography
ORDER BY churn_rate_pct DESC;

-- 3. Churn by gender
SELECT
    Gender,
    COUNT(*)                                   AS customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)   AS churn_rate_pct
FROM churn_customers
GROUP BY Gender;

-- 4. Age-group segmentation
SELECT
    CASE
        WHEN Age < 30 THEN '18-29'
        WHEN Age < 40 THEN '30-39'
        WHEN Age < 50 THEN '40-49'
        WHEN Age < 60 THEN '50-59'
        ELSE '60+'
    END                                         AS age_group,
    COUNT(*)                                    AS customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)    AS churn_rate_pct
FROM churn_customers
GROUP BY age_group
ORDER BY age_group;

-- 5. Product usage vs churn
SELECT
    NumOfProducts,
    COUNT(*)                                    AS customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)    AS churn_rate_pct
FROM churn_customers
GROUP BY NumOfProducts
ORDER BY NumOfProducts;

-- 6. Active vs inactive members
SELECT
    IsActiveMember,
    COUNT(*)                                    AS customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)    AS churn_rate_pct
FROM churn_customers
GROUP BY IsActiveMember;

-- 7. Credit score band vs churn
SELECT
    CASE
        WHEN CreditScore < 580 THEN 'Poor (<580)'
        WHEN CreditScore < 670 THEN 'Fair (580-669)'
        WHEN CreditScore < 740 THEN 'Good (670-739)'
        WHEN CreditScore < 800 THEN 'Very Good (740-799)'
        ELSE 'Excellent (800+)'
    END                                          AS credit_band,
    COUNT(*)                                     AS customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)     AS churn_rate_pct
FROM churn_customers
GROUP BY credit_band
ORDER BY churn_rate_pct DESC;

-- 8. Balance distribution vs churn
SELECT
    CASE
        WHEN Balance = 0 THEN 'Zero Balance'
        WHEN Balance < 50000 THEN 'Low (<50k)'
        WHEN Balance < 100000 THEN 'Mid (50k-100k)'
        WHEN Balance < 150000 THEN 'High (100k-150k)'
        ELSE 'Very High (150k+)'
    END                                          AS balance_band,
    COUNT(*)                                     AS customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)     AS churn_rate_pct
FROM churn_customers
GROUP BY balance_band
ORDER BY churn_rate_pct DESC;

-- 9. Tenure vs churn
SELECT
    Tenure,
    COUNT(*)                                     AS customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)     AS churn_rate_pct
FROM churn_customers
GROUP BY Tenure
ORDER BY Tenure;

-- 10. High-value customers at risk (balance > 100k AND inactive)
SELECT
    CustomerId,
    Geography,
    Age,
    Balance,
    NumOfProducts,
    IsActiveMember,
    Exited
FROM churn_customers
WHERE Balance > 100000
  AND IsActiveMember = 0
ORDER BY Balance DESC
LIMIT 50;

-- 11. Country x Gender churn matrix
SELECT
    Geography,
    Gender,
    COUNT(*)                                     AS customers,
    ROUND(100.0 * SUM(Exited) / COUNT(*), 2)     AS churn_rate_pct
FROM churn_customers
GROUP BY Geography, Gender
ORDER BY Geography, Gender;
