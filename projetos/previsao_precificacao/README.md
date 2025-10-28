# 📈 Projeto: Previsão de Precificação

## 🧩 Visão Geral
Este projeto implementa um pipeline completo e otimizado de **previsão de precificação** em nível sênior, desde o tratamento dos dados brutos até o refinamento final de modelos de Machine Learning. O objetivo é construir um sistema robusto para prever valores de venda com base em variáveis comerciais, sazonais e climáticas.

O fluxo segue a seguinte sequência:
1. **tratamento.py** → Leitura, padronização e otimização dos dados.
2. **EDA.py** → Análise exploratória com geração automática de KPIs, gráficos e insights.
3. **modelagem.py** → Construção de modelos baseline com LightGBM, CatBoost e RandomForest.
4. **refinamentomodelagem.py** → Refinamento avançado do LightGBM com Optuna e TimeSeriesSplit.
5. **teste_mesa.py** → Validação do modelo final com dados simulados e métricas reais.

---

## ⚙️ Estrutura de Diretórios
```
previsao_precificacao/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── predicoes/
├── logs/
├── models/
│   ├── baseline_results/
│   └── refinamento_lightgbm/
├── reports/
│   └── eda/
│       ├── figures/
│       └── kpis/
├── src/
│   ├── tratamento.py
│   ├── EDA.py
│   ├── modelagem.py
│   ├── refinamentomodelagem.py
│   └── teste_mesa.py
```

---

## 🧠 Scripts Principais

### 1. `tratamento.py`
Responsável por:
- Leitura do dataset bruto via caminho configurado em `config.yaml`.
- Padronização de nomes de colunas e conversão de tipos.
- Tratamento de colunas complexas (listas/arrays).
- Cálculo de estatísticas básicas, outliers e duplicações.
- Geração do **dataset otimizado** em formato Parquet (compressão Snappy).

**Saída:** `data/processed/previsao_precificacao.parquet`

---

### 2. `EDA.py`
Executa uma análise exploratória detalhada:
- Cálculo automático de **KPIs** de vendas, descontos, clima e atividade.
- Geração de relatórios em `.csv`, `.json` e `.md`.
- Criação automática de gráficos (`heatmap`, `histogramas`, `boxplots`, `top lojas`, etc.).
- Produz um **relatório Markdown consolidado** com KPIs e insights automáticos.

**Saídas:**
- `reports/eda/kpis/kpis_summary.json`
- `reports/eda/kpis/summary.md`
- `reports/eda/figures/*.png`

---

### 3. `modelagem.py`
Etapa de baseline modelagem:
- Amostragem balanceada por loja e categoria.
- Criação de variáveis derivadas (ano, mês, dia, etc.).
- Treinamento de três modelos:
  - LightGBM
  - CatBoost
  - RandomForest
- Busca aleatória de hiperparâmetros com `RandomizedSearchCV`.
- Avaliação por **RMSE, MAE, R² e MAPE**.
- Geração de ranking e relatório em Markdown com gráfico comparativo.

**Saídas:**
- `models/baseline_results/resultados_baseline.json`
- `models/baseline_results/report_modelagem.md`
- `models/baseline_results/comparativo_rmse.png`

---

### 4. `refinamentomodelagem.py`
Aprimoramento do modelo com foco em desempenho e generalização:
- Utiliza **Optuna (TPESampler)** para busca bayesiana de hiperparâmetros.
- Estrutura de validação temporal (`TimeSeriesSplit`).
- Criação de features temporais e interativas (lags, médias móveis, interações climáticas).
- Geração de métricas no conjunto holdout e gráficos de avaliação.

**Saídas:**
- `models/refinamento_lightgbm/lightgbm_refinado.pkl`
- `models/refinamento_lightgbm/final_metrics.json`
- `models/refinamento_lightgbm/feature_importance.png`
- `models/refinamento_lightgbm/dispersao_preditos.png`

---

### 5. `teste_mesa.py`
Validação final com dados artificiais coerentes:
- Simula 15 registros com variabilidade realista.
- Aplica engenharia de features idêntica ao pipeline.
- Calcula métricas de erro e exporta previsões.

**Saídas:**
- `data/predicoes/teste_mesa_predicoes.parquet`
- `data/predicoes/teste_mesa_predicoes.csv`

---

## 📦 Dependências Principais
```bash
pandas
numpy
yaml
matplotlib
seaborn
scikit-learn
lightgbm
catboost
optuna
psutil
joblib
tqdm
pyarrow
tabulate
```

---

## 🚀 Execução
Ordem recomendada:
```bash
python tratamento.py
python EDA.py
python modelagem.py
python refinamentomodelagem.py
python teste_mesa.py
```

---

## 📈 Métricas e Logs
- Todos os logs são salvos em `logs/`.
- Cada etapa inclui medição de tempo e uso de memória (via `psutil`).
- O pipeline completo exibe desempenho, consumo de memória e resultados em formato legível.

---

## 🧾 Licença
Este projeto é de uso educacional e profissional, desenvolvido com fins de estudo, portfólio e aprimoramento técnico em Data Science aplicada a precificação.

---

**Autor:** Isaac — Cientista de Dados  
**Repositório:** `github/data-science/projetos/previsao_precificacao`

