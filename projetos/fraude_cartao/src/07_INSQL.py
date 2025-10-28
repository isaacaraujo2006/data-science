-- 1. Total de transações
SELECT COUNT(*) AS total_transacoes FROM transacoes;

-- 2. Total de fraudes detectadas
SELECT COUNT(*) AS total_fraudes FROM transacoes WHERE class = 1;

-- 3. Percentual de transações fraudulentas
SELECT 
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 4) AS perc_fraude
FROM transacoes;

-- 4. Transações com maior valor médio (fraude vs. não fraude)
SELECT 
    class,
    ROUND(AVG(amount), 2) AS valor_medio
FROM transacoes
GROUP BY class;

-- 5. Distribuição de fraudes por hora do dia
SELECT 
    hour_of_day,
    COUNT(*) AS total,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraudes,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS perc_fraudes
FROM transacoes
GROUP BY hour_of_day
ORDER BY perc_fraudes DESC;

-- 6. Transações com valor considerado outlier e taxa de fraude
SELECT 
    outlier_amount,
    COUNT(*) AS total_transacoes,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraudes,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS perc_fraudes
FROM transacoes
GROUP BY outlier_amount;

-- 7. Top 10 transações com maior valor suspeito
SELECT * FROM transacoes 
WHERE class = 1
ORDER BY amount DESC
LIMIT 10;

-- 8. Distribuição de fraudes por faixa de valor (buckets)
SELECT 
    CASE 
        WHEN amount < 0 THEN 'Negativo'
        WHEN amount < 10 THEN '0-10'
        WHEN amount < 50 THEN '10-50'
        WHEN amount < 100 THEN '50-100'
        WHEN amount < 500 THEN '100-500'
        ELSE '500+' 
    END AS faixa_valor,
    COUNT(*) AS total,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraudes,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END)/ COUNT(*), 2) AS perc_fraudes
FROM transacoes
GROUP BY faixa_valor
ORDER BY perc_fraudes DESC;

-- 9. Média e desvio padrão dos componentes principais (V1–V28) para fraudes
SELECT 
    AVG(v1) AS media_v1,
    STDDEV(v1) AS desvio_v1,
    AVG(v2) AS media_v2,
    STDDEV(v2) AS desvio_v2
    -- Adicione mais v3, v4... se quiser
FROM transacoes
WHERE class = 1;

-- 10. Análise de transações com maior densidade de fraudes (por minuto)
SELECT 
    FLOOR(time / 60) AS minuto,
    COUNT(*) AS total_transacoes,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS total_fraudes,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END)/COUNT(*), 2) AS perc_fraudes
FROM transacoes
GROUP BY FLOOR(time / 60)
ORDER BY perc_fraudes DESC
LIMIT 10;

-- 11. Transações com alta anomalia (fraudes com outlier_amount = 1)
SELECT * FROM transacoes 
WHERE class = 1 AND outlier_amount = 1
ORDER BY amount DESC;

-- 12. Quantidade de transações por hora em dias mais críticos
SELECT 
    hour_of_day,
    COUNT(*) AS total_transacoes,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS total_fraudes
FROM transacoes
GROUP BY hour_of_day
ORDER BY total_fraudes DESC;

-- 13. Faixa horária com maior concentração de fraudes
SELECT 
    CASE 
        WHEN hour_of_day BETWEEN 0 AND 5 THEN 'Madrugada'
        WHEN hour_of_day BETWEEN 6 AND 11 THEN 'Manhã'
        WHEN hour_of_day BETWEEN 12 AND 17 THEN 'Tarde'
        ELSE 'Noite'
    END AS periodo,
    COUNT(*) AS total_transacoes,
    SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END) AS fraudes,
    ROUND(100.0 * SUM(CASE WHEN class = 1 THEN 1 ELSE 0 END)/COUNT(*), 2) AS perc_fraudes
FROM transacoes
GROUP BY periodo
ORDER BY perc_fraudes DESC;

-- 14. Taxa de fraude entre valores abaixo e acima da mediana
WITH mediana AS (
    SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY amount) AS valor_median
    FROM transacoes
)
SELECT 
    CASE WHEN t.amount < m.valor_median THEN 'Abaixo da mediana' ELSE 'Acima da mediana' END AS faixa,
    COUNT(*) AS total,
    SUM(CASE WHEN t.class = 1 THEN 1 ELSE 0 END) AS fraudes,
    ROUND(100.0 * SUM(CASE WHEN t.class = 1 THEN 1 ELSE 0 END)/ COUNT(*), 2) AS perc_fraudes
FROM transacoes t
CROSS JOIN mediana m
GROUP BY faixa;

-- 15. Últimas transações com suspeita alta (probabilidade > 0.95)
-- (Pressupõe que você tenha uma coluna chamada "score_fraude" adicionada após predição)
SELECT * FROM transacoes 
WHERE score_fraude >= 0.95
ORDER BY time DESC
LIMIT 20;
