# 📦 Previsão de Demanda — Machine Learning + Django + Power BI  

> ⚡ **Pipeline completa de Ciência de Dados aplicada ao varejo (Olist Dataset)**  
> Desde o tratamento de dados e feature engineering até a previsão com LightGBM e visualização via Power BI e aplicação web Django.

---

## 🚀 Instalação e Execução

**1️⃣ Clonar o repositório**

```bash
git clone https://github.com/isaacaraujo2006/data-science/tree/main/projetos/previsao_cesta_demanda
cd previsao_cesta_demanda
```

**2️⃣ Criar e ativar ambiente virtual**

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

**3️⃣ Instalar dependências**

```bash
pip install -r requirements.txt
```

---

## ⚙️ Estrutura do Projeto

```bash
previsao_cesta_demanda/
│
├── config/
│   └── config.yaml                  # Caminhos e variáveis globais
│
├── data/
│   ├── raw/                         # Dados originais em parquet
│   └── processed/                   # Dados tratados e enriquecidos
│
├── models/
│   ├── final_model.joblib           # Modelo LightGBM treinado
│   └── versioned/                   # Versões anteriores
│
├── preprocessors/
│   ├── preprocessor.joblib
│   └── scaler.joblib
│
├── reports/
│   ├── figures/                     # Gráficos gerados no EDA
│   ├── shap_values.csv              # Explicabilidade SHAP
│   ├── shap_summary.png
│   └── data_profile.html
│
├── logs/                            # Logs de processamento e modelagem
│
├── src/
│   ├── tratamento.py                # Limpeza, normalização e relatórios de qualidade
│   ├── feature_engineering.py       # Criação de variáveis derivadas
│   ├── EDA.py                       # Análises exploratórias e gráficos
│   ├── conversao.py                 # Conversão Parquet → CSV
│   └── refinamento_lightgbm.py      # Treinamento, tuning e avaliação do modelo
│
├── site_django/                     # Aplicação web com previsão e métricas
│   ├── templates/
│   ├── static/
│   ├── views.py
│   ├── urls.py
│   └── ...
│
└── requirements.txt
```

---

## 🧠 Pipeline do Projeto

1. **Coleta e Leitura dos Dados**  
   Arquivos Olist consolidados em `previsao_cesta_demanda.parquet`.

2. **Tratamento e Limpeza (`tratamento.py`)**
   - Correção de tipos, colunas e valores anômalos.  
   - Detecção de outliers e faltantes.  
   - Geração automática de relatórios de qualidade e drift.  
   - Escalonamento com `RobustScaler` e exportação para parquet limpo.

3. **Análise Exploratória (`EDA.py`)**
   - Distribuições de preço, frete, volume e categorias.  
   - Séries temporais (mensal, semanal, diária).  
   - Review score, SLA por estado e mapa interativo de clientes (Plotly).  
   - Gráficos salvos em `/reports/figures`.

4. **Feature Engineering (`feature_engineering.py`)**
   - Criação de variáveis temporais, logísticas, de produto e cliente.  
   - Cálculo de *recência*, *ticket médio*, *densidade* e *volume*.  
   - Exportação para `processed_features.parquet`.

5. **Modelagem com LightGBM (`refinamento_lightgbm.py`)**
   - Regressão supervisionada para previsão de demanda.  
   - Busca de hiperparâmetros com `Optuna`.  
   - Métricas registradas em `/metrics` e logs completos em `/logs`.

6. **Aplicação Web — Django**
   - Interface moderna para:
     - 🔹 **Previsão Manual:** o cliente insere parâmetros e obtém a previsão.  
     - 🔹 **Previsão Automática:** baseada em dados históricos.  
     - 🔹 **Métricas do Modelo:** exibição de RMSE, MAE, R² e MAPE.  
   - Layout minimalista, com tema verde e cards de métricas.  
   - Integração com Power BI para visualização gerencial.

7. **Dashboard no Power BI**
   - KPIs de previsão vs. vendas reais.  
   - Mapa interativo e análise por categoria, estado e período.  
   - Background personalizado criado em PNG para visualizações limpas e profissionais.

---

## 📊 Resultados e Métricas

| Métrica | Valor |
|----------|--------|
| RMSE | 312.45 |
| MAE  | 198.21 |
| R²   | 0.91 |
| MAPE | 8.4% |

> As métricas indicam alta precisão preditiva para séries temporais de demanda.

**📈 Interpretação com SHAP:**  
Os gráficos `shap_summary.png` e `shap_values.csv` mostram a importância relativa das variáveis mais influentes (ex.: `price`, `freight_value`, `feat_category_demand_month`, `feat_customer_avg_ticket`).

---

## 🌐 Execução do Site Django

**1️⃣ Iniciar servidor**

```bash
cd site_django
python manage.py runserver
```

**→ http://127.0.0.1:8000**

**2️⃣ Rotas principais:**

| Rota | Função |
|------|---------|
| `/` | Página inicial com resumo |
| `/manual/` | Previsão manual |
| `/auto/` | Previsão automática |
| `/metricas/` | Desempenho do modelo |

---

## 🧰 Tecnologias Utilizadas

| Categoria | Ferramentas |
|------------|-------------|
| Linguagem | Python 3.8 |
| Framework Web | Django |
| Visualização | Power BI, Plotly, Seaborn, Matplotlib |
| Machine Learning | LightGBM, Scikit-learn, Optuna |
| Processamento | Pandas, NumPy, PyArrow, Fastparquet |
| Validação & Perfil | ydata-profiling, SHAP |
| Escalonamento | RobustScaler |
| Logging | YAML, joblib, log files |
| Dashboard | Power BI Desktop |

---

## 📦 Reprodutibilidade

**Executar todo o pipeline:**

```bash
python src/tratamento.py
python src/feature_engineering.py
python src/refinamento_lightgbm.py
python src/EDA.py
python src/conversao.py
```

> Ao final, o modelo e os gráficos são atualizados automaticamente nos diretórios configurados em `config.yaml`.

---

## 🧪 Teste Rápido

**Simular previsão manual via interface Django:**

1. Inserir valores de preço, categoria, peso e volume.  
2. Clicar em **“Gerar Previsão”**.  
3. A aplicação retorna o valor previsto de demanda e exibe métricas globais do modelo.

---

## 🧾 Créditos e Autoria

**Desenvolvido por:** *Isaac Araújo*  
📧 `isaacaraujo2006@gmail.com`  
💼 *Cientista de Dados | Machine Learning | Power BI | Django*
