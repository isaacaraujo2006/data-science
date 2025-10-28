# 👥 Projeto de Segmentação de Clientes — Machine Learning + Análise Exploratória

> ⚡ **Clusterização inteligente para identificar perfis de clientes e padrões de comportamento.**  
> Projeto completo com pipeline de dados, visualização e modelagem não supervisionada (K-Means).

---

## 🚀 Visão Geral

O objetivo deste projeto é **segmentar clientes** com base em seus atributos de compra, comportamento e perfil demográfico.  
Com isso, é possível **personalizar campanhas de marketing, otimizar estratégias comerciais** e **aumentar a retenção de clientes**.

**Pipeline completo:**
1. 🧹 **Tratamento de Dados** — Limpeza, transformação e padronização.
2. 📊 **Análise Exploratória (EDA)** — Estatísticas, visualizações e correlações.
3. 🧠 **Modelagem (K-Means)** — Definição automática de clusters e análise de perfil.

---

## ⚙️ Estrutura do Projeto

```bash
segmentacao_clientes/
│
├── data/
│   ├── raw/                # Dados brutos
│   └── processed/           # Dados tratados
│
├── logs/                    # Registros de execução
│
├── reports/
│   └── figures/             # Visualizações geradas na EDA
│
├── models/
│   └── kmeans_model.joblib  # Modelo final de clusterização
│
├── src/
│   ├── 1_tratamento.py      # Tratamento e limpeza de dados
│   ├── 2_eda.py             # Análise exploratória e estatísticas
│   └── 3_modelagem.py       # Modelagem e definição dos clusters
│
└── README.md
```

---

## 🧹 1. Tratamento de Dados (`1_tratamento.py`)

- Carrega o dataset bruto.  
- Renomeia colunas e padroniza nomes.  
- Trata valores nulos e remove outliers com base em IQR.  
- Valida consistência e exporta dataset tratado em múltiplos formatos.

**Entrada:** `data/raw/customer_segmentation.csv`  
**Saída:**  
- `data/processed/processed.csv`  
- `data/processed/processed.parquet`

---

## 📊 2. Análise Exploratória (`2_eda.py`)

- Calcula estatísticas descritivas e distribuições.  
- Gera histogramas, boxplots, scatter plots e heatmaps de correlação.  
- Aplica **PCA (Análise de Componentes Principais)** para reduzir dimensionalidade.  
- Exporta gráficos e relatórios automáticos.

**Entrada:** `data/processed/processed.csv`  
**Saída:** `reports/figures/` (gráficos e relatórios)

---

## 🧠 3. Modelagem (`3_modelagem.py`)

- Divide os dados em treino e teste (70/30).  
- Aplica normalização (`StandardScaler`) e codificação de variáveis categóricas.  
- Treina modelo **K-Means** variando o número de clusters (k = 2–10).  
- Seleciona o melhor número de clusters via:
  - 📈 **Método do Cotovelo**
  - 📉 **Silhouette Score**
- Salva o modelo e rótulos preditos.

**Entrada:** `data/processed/processed.csv`  
**Saída:** `models/kmeans_model.joblib`  

---

## 🧩 Como Executar

```bash
python src/1_tratamento.py
python src/2_eda.py
python src/3_modelagem.py
```

---

## 🧰 Requisitos

Instalar dependências com:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn tqdm joblib
```

---

## 📊 Exemplo de Resultados

| Cluster | Qtd. Clientes | Ticket Médio | Recência (dias) | Região Predominante |
|----------|----------------|---------------|------------------|----------------------|
| 0 | 845 | R$ 230,50 | 28 | SP |
| 1 | 1.220 | R$ 480,10 | 14 | RJ |
| 2 | 670 | R$ 135,70 | 41 | MG |

> Os grupos indicam perfis distintos de comportamento, permitindo ações direcionadas de marketing e fidelização.

---

## 🔮 Próximos Passos

- Implementar **DBSCAN** e **Gaussian Mixture Models** para comparação.  
- Integrar **Power BI** para dashboards interativos de clusters.  
- Criar **API Flask** para disponibilizar segmentações em tempo real.

---

## 🧾 Créditos e Autoria

**Desenvolvido por:** *Isaac Araújo*  
📧 `isaacaraujo2006@gmail.com`  
💼 *Cientista de Dados | Machine Learning | Power BI*
