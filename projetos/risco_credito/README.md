# 💳 Previsão de Risco de Crédito — CatBoost + Streamlit

> 🚀 **Projeto de Machine Learning para previsão de inadimplência de clientes.**
> Inclui tratamento de dados, EDA automatizada, modelagem com CatBoost e interface interativa em Streamlit.

---

## 📘 Visão Geral

O projeto tem como objetivo **prever o risco de crédito** de clientes com base em variáveis comportamentais e financeiras.  
A pipeline foi desenvolvida para ser **modular, explicável e integrada ao Power BI** e ao **Streamlit**.

---

## ⚙️ Pipeline de Desenvolvimento

1. **Tratamento de Dados (`01_tratamento.py`)**
   - Tradução e padronização de colunas.
   - Remoção de duplicatas e outliers.
   - Imputação de valores faltantes e conversão de tipos.
   - Geração do dataset tratado em `.parquet` e `.csv` (para Power BI).

2. **Análise Exploratória (`02_EDA.py`)**
   - Estatísticas descritivas e gráficos automatizados (histogramas, violin plots, heatmaps).
   - Testes estatísticos *Mann-Whitney U* para verificar diferenças entre classes.
   - Exportação automática das figuras para `reports/figures`.

3. **Modelagem (`03_modelagem.py`)**
   - Comparação entre **LightGBM**, **XGBoost**, **CatBoost** e **Random Forest**.
   - Balanceamento via **SMOTE**.
   - **Busca de hiperparâmetros** com `RandomizedSearchCV`.
   - **Calibração de probabilidades** e busca de *threshold ótimo*.
   - Métricas calculadas: *F1, Precision, Recall, Accuracy, AUC-PR*.
   - **SHAP** e **curvas de aprendizado** salvas automaticamente.

4. **Otimização CatBoost (`cat.py`)**
   - Ajuste fino do modelo CatBoost com `SMOTETomek`.
   - Calibração isotônica e seleção de threshold ótimo com base em F1-score.
   - Geração de relatórios e gráficos: importância das variáveis, matriz de confusão, curvas F1/accuracy.

5. **Interface Web (`site.py`)**
   - Aplicação interativa em **Streamlit** com duas abas:
     - 📂 **Previsão em Lote:** upload de CSV/Parquet e download dos resultados.
     - 📝 **Previsão Manual:** formulário para preenchimento individual.
   - Estilo moderno e responsivo em **CSS customizado**.
   - Retorno visual do resultado com probabilidade, classificação e barra de progresso.

6. **Testes e Validação (`teste.py`)**
   - Avaliação do modelo em *zona cinzenta* (probabilidades entre 0.20 e 0.30).
   - Cálculo de acurácia e taxa de acertos interpretável.

7. **Consultas SQL (`queryssql`)**
   - Conjunto de **15 queries analíticas** para extrair insights sobre inadimplência:
     - Distribuição por faixa etária, sexo, educação, estado civil e limite de crédito.
     - Correlação entre atraso de pagamento e inadimplência.

---

## 🗂 Estrutura do Projeto

```
risco_credito/
|➜ config/
|   └── config.yaml                # Caminhos e parâmetros globais
|➜ data/
|   |➜ raw/                        # Dados originais
|   |➜ processed/                  # Dados tratados (Parquet e CSV)
|➜ models/                         # Modelos e thresholds salvos
|➜ preprocessors/                  # Scalers e preprocessadores
|➜ logs/                           # Registros de execução
|➜ reports/
|   |➜ figures/                    # Gráficos e relatórios visuais
|   └── shap_values.csv            # Explicabilidade
|➜ src/
|   |➜ 01_tratamento.py
|   |➜ 02_EDA.py
|   |➜ 03_modelagem.py
|   |➜ cat.py
|   |➜ teste.py
|➜ site.py                         # Aplicação Streamlit
|➜ queryssql                       # Consultas SQL
|➜ README.md                       # Este arquivo
```

---

## 🧠 Tecnologias Utilizadas

| Categoria | Ferramentas |
|------------|-------------|
| Linguagem | Python 3.8 |
| Bibliotecas Principais | pandas, numpy, seaborn, matplotlib, plotly, joblib |
| Modelagem | LightGBM, XGBoost, CatBoost, RandomForest, imbalanced-learn |
| Otimização | RandomizedSearchCV, SMOTE, SMOTETomek |
| Avaliação | sklearn.metrics, SHAP, calibration, learning_curve |
| Visualização | Streamlit, Plotly, Seaborn |
| Persistência | Parquet, CSV, Joblib |
| Relatórios | YAML, logs automáticos, figuras e métricas |

---

## ▶️ Como Executar

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/isaacaraujo2006/data-science/tree/main/projetos/risco_credito
cd risco_credito
```

### 2️⃣ Criar e ativar ambiente virtual
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 4️⃣ Executar os scripts principais
```bash
python src/01_tratamento.py
python src/02_EDA.py
python src/03_modelagem.py
```

### 5️⃣ Rodar o aplicativo Streamlit
```bash
streamlit run site.py
```
➡️ Acesse em: **http://localhost:8501**

---

## 📊 Resultados e Métricas

| Modelo | F1-Score | AUC-PR | Observações |
|--------|-----------|---------|--------------|
| LightGBM | ~0.82 | ~0.85 | Melhor balanceamento geral |
| XGBoost | ~0.80 | ~0.84 | Performance estável |
| CatBoost | **~0.84** | **~0.88** | 🏆 Modelo final escolhido |
| Random Forest | ~0.79 | ~0.82 | Menor eficiência em classes desbalanceadas |

**Threshold ótimo (CatBoost):** ~0.25  
**Custo ponderado:** FP × 1 + FN × 10  
**Métricas e gráficos** disponíveis em `/reports/` e `/logs/`.

---

## 💻 Dashboard e Visualização

O arquivo `previsao_risco_credito.csv` pode ser importado diretamente no **Power BI**, permitindo análises interativas:

- Distribuição de risco por faixa etária e limite.
- Tendência de inadimplência por perfil sociodemográfico.
- Comparação entre previsões e casos reais.

---

## 🧩 Próximas Melhorias

- Implementar API REST com **FastAPI** para consumo externo.
- Criar automação de re-treinamento semanal via **Airflow**.
- Adicionar monitoramento de *data drift* e métricas online.
- Exportar explicações SHAP no Streamlit.

---

## 👨‍💻 Autor

**Isaac Araújo**  
Cientista de Dados | Python | Machine Learning | Power BI  
📫 [GitHub](https://github.com/isaacaraujo2006) | 💼 [LinkedIn](https://www.linkedin.com/in/isaacaraujo2006)
