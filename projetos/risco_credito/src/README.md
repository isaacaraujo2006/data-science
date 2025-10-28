# 💳 Projeto de Previsão de Risco de Crédito

Este projeto entrega uma solução **end-to-end** para previsão de inadimplência em crédito, desde a engenharia de dados, análise exploratória, modelagem avançada, avaliação, geração de relatórios e deploy em web app interativo via **Streamlit**.

## ✨ Visão Geral

- **Tratamento e padronização de dados**: pipeline robusto para garantir dados limpos, tipos corretos, tratamento de outliers, conversão de variáveis e encoding.
- **Análise Exploratória (EDA)**: geração de gráficos automáticos, testes estatísticos, heatmaps de correlação e insights de negócio.
- **Modelagem avançada**: pipelines completos com _CatBoost_, _LightGBM_, _XGBoost_ e _Random Forest_, balanceamento via SMOTE/SMOTETomek, _calibração_ de probabilidades, seleção automática de threshold ótimo para F1-score.
- **Teste de mesa e zona cinzenta**: análise dos exemplos mais difíceis para o modelo.
- **SQL Insights**: 15 queries analíticas para uso direto em Data Warehouse ou dashboards.
- **Web App Profissional**: Deploy em Streamlit com previsão manual e por arquivo, layout escuro, responsivo e moderno.

---

## 🚀 Como rodar o projeto

### 1. Pré-requisitos

- Python 3.8+
- Recomendado: criar um ambiente virtual
- Instale as dependências principais:

```bash
pip install -r requirements.txt
# ou, manualmente:
pip install pandas numpy scikit-learn matplotlib seaborn plotly imblearn catboost xgboost lightgbm joblib streamlit pyyaml tqdm shap
2. Estrutura do Projeto
arduino
Copiar
Editar
risco_credito/
├── config/
│   └── config.yaml
├── data/
│   └── raw/
│   └── processed/
├── logs/
├── models/
├── notebook/
├── preprocessors/
├── reports/
│   └── figures/
│   └── shap_values.csv
├── src/
│   ├── 01_tratamento.py
│   ├── 02_EDA.py
│   ├── 03_modelagem.py
│   ├── cat.py
│   ├── teste.py
│   └── site.py
├── site.py 
├── insights.sql
└── README.md
3. Pipelines Principais
01_tratamento.py: Processa, limpa, trata e salva os dados prontos para modelagem.

02_EDA.py: Gera todos os gráficos e relatórios exploratórios automaticamente.

03_modelagem.py: Compara modelos de classificação, seleciona threshold ótimo, salva artefatos.

cat.py: Pipeline CatBoost otimizado (SMOTETomek, calibração, exportação de modelo).

teste.py: Teste de mesa avançado, inclusive na zona cinzenta (avaliação de casos críticos).

site.py: Aplicação web Streamlit, interface intuitiva para upload e previsão manual.

📂 Dados e configuração
Os caminhos dos dados, modelos, artefatos, relatórios e logs estão todos centralizados em config/config.yaml.
Basta ajustar para o seu ambiente de desenvolvimento.

🧠 Observações técnicas
Pipelines projetados com padrão profissional: uso de pipeline do scikit-learn, SMOTE/SMOTETomek, tuning automatizado, early stopping, calibração de probabilidades, tuning de threshold.

Pronto para experimentação e deploy.

Código organizado por scripts independentes para facilitar manutenção, testes e experimentos.

👤 Autor
Projeto desenvolvido por Isaac Araújo, com código limpo, reproducível, orientado a portfólio e entrevistas.

📢 Contato
LinkedIn: https://www.linkedin.com/in/isaacdearaujo/

GitHub: https://github.com/isaacaraujo2006

Dica: Para adaptar para outros problemas de risco, basta ajustar o pré-processamento, variáveis de entrada e config!