/* ================================================================
    SCRIPT DE ANÁLISE DE FRAUDE - 15 INSIGHTS
    Autor: Isaac Araújo
    Descrição: Consultas para monitoramento e auditoria de fraudes
================================================================ */

/* 1. Taxa de fraude por mês */
SELECT 
    DATE_TRUNC('month', data_transacao) AS mes,
    COUNT(*) AS total_transacoes,
    SUM(fraude) AS total_fraudes,
    ROUND(100.0 * SUM(fraude) / COUNT(*), 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY 1
ORDER BY mes DESC;

/* 2. Top 10 usuários por valor total de fraudes */
SELECT 
    id_usuario,
    SUM(valor_transacao) AS valor_total_fraudes,
    COUNT(*) AS qtd_fraudes
FROM transacoes
WHERE fraude = 1
GROUP BY id_usuario
ORDER BY valor_total_fraudes DESC
LIMIT 10;

/* 3. Distribuição de valor em transações fraudulentas vs não fraudulentas */
SELECT 
    fraude,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY valor_transacao) AS mediana,
    AVG(valor_transacao) AS media,
    MAX(valor_transacao) AS maximo
FROM transacoes
GROUP BY fraude;

/* 4. Quantidade de transações suspeitas por hora do dia */
SELECT 
    EXTRACT(HOUR FROM data_transacao) AS hora,
    COUNT(*) AS total_transacoes,
    SUM(fraude) AS total_fraudes
FROM transacoes
GROUP BY hora
ORDER BY hora;

/* 5. Estados/Regiões com mais fraudes */
SELECT 
    estado,
    COUNT(*) AS total_transacoes,
    SUM(fraude) AS total_fraudes,
    ROUND(100.0 * SUM(fraude) / COUNT(*), 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY estado
ORDER BY taxa_fraude_pct DESC;

/* 6. Relação entre número de tentativas e fraude */
SELECT 
    tentativas_login,
    COUNT(*) AS total_transacoes,
    SUM(fraude) AS total_fraudes,
    ROUND(100.0 * SUM(fraude) / COUNT(*), 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY tentativas_login
ORDER BY taxa_fraude_pct DESC;

/* 7. Dispositivos mais utilizados em fraudes */
SELECT 
    dispositivo,
    COUNT(*) AS total_transacoes,
    SUM(fraude) AS total_fraudes
FROM transacoes
GROUP BY dispositivo
ORDER BY total_fraudes DESC;

/* 8. Fraudes acima de determinado valor */
SELECT *
FROM transacoes
WHERE fraude = 1
  AND valor_transacao > 10000
ORDER BY valor_transacao DESC;

/* 9. Tempo médio entre transações para usuários fraudulentos */
WITH diffs AS (
    SELECT 
        id_usuario,
        data_transacao - LAG(data_transacao) OVER (PARTITION BY id_usuario ORDER BY data_transacao) AS tempo_diff
    FROM transacoes
    WHERE fraude = 1
)
SELECT 
    id_usuario,
    AVG(tempo_diff) AS tempo_medio
FROM diffs
GROUP BY id_usuario
ORDER BY tempo_medio;

/* 10. Percentual de fraudes por canal de transação */
SELECT 
    canal,
    COUNT(*) AS total_transacoes,
    SUM(fraude) AS total_fraudes,
    ROUND(100.0 * SUM(fraude) / COUNT(*), 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY canal
ORDER BY taxa_fraude_pct DESC;

/* 11. Fraudes recorrentes por usuário */
SELECT 
    id_usuario,
    COUNT(*) AS total_fraudes
FROM transacoes
WHERE fraude = 1
GROUP BY id_usuario
HAVING COUNT(*) > 1
ORDER BY total_fraudes DESC;

/* 12. Comparação do ticket médio */
SELECT 
    fraude,
    ROUND(AVG(valor_transacao), 2) AS ticket_medio
FROM transacoes
GROUP BY fraude;

/* 13. Correlação entre score de risco e fraude */
SELECT 
    ROUND(score_risco, 1) AS faixa_score,
    COUNT(*) AS total_transacoes,
    SUM(fraude) AS total_fraudes,
    ROUND(100.0 * SUM(fraude) / COUNT(*), 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY faixa_score
ORDER BY faixa_score DESC;

/* 14. Evolução semanal da taxa de fraude */
SELECT 
    DATE_TRUNC('week', data_transacao) AS semana,
    ROUND(100.0 * SUM(fraude) / COUNT(*), 2) AS taxa_fraude_pct
FROM transacoes
GROUP BY semana
ORDER BY semana;

/* 15. Detecção de contas novas com fraude */
SELECT 
    id_usuario,
    MIN(data_transacao) AS primeira_transacao,
    SUM(fraude) AS total_fraudes
FROM transacoes
GROUP BY id_usuario
HAVING DATE_PART('day', NOW() - MIN(data_transacao)) < 30
ORDER BY total_fraudes DESC;
