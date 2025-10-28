# 💳 Detecção de Fraude em Cartões com Machine Learning #

Este projeto tem como objetivo detectar transações fraudulentas em cartões de crédito utilizando técnicas avançadas de **Machine Learning**, com apoio de explicabilidade via **SHAP**, visualização interativa no **Streamlit** e geração de insights estratégicos via **SQL** e **Power BI**.


## Estrutura do Projeto ##

A estrutura do projeto é a seguinte:

fraude_cartao/
│
├── config/ # Arquivos de configuração
│ └── config.yaml
│
├── data/
│ ├── raw/ # Dados brutos
│ └── processed/ # Dados tratados
│
├── logs/ # Arquivos de log de execução
│
├── models/ # Modelos treinados e calibrados
│ └── histgb_model_calibrated.pkl
│
├── reports/
│ └── figures/ # Curvas de aprendizado, SHAP, etc.
│
├── notebook/ # Análises exploratórias e experimentações
│
├── src/ # Scripts principais
│ ├── 04_avanc_modelos.py # Treinamento com HistGradientBoosting + SMOTE
│ ├── 05_teste.py # Teste de mesa na zona cinzenta
│ ├── STREAMLIT.py # App de inferência web
│ ├── pipeline.py # Pré-processamento inicial
│ └── 07_INSQL_py # Consultas SQL para análise no Power BI
│
├── requirements.txt # Lista de bibliotecas necessárias
└── README.md # Documentação do projeto

## 🔍 Objetivo ##

Detectar fraudes em transações de cartão de crédito com alto desempenho, utilizando um modelo calibrado com threshold otimizado, explicável e validado com técnicas de:

- Aprendizado supervisionado (HistGradientBoosting)
- Balanceamento de dados com SMOTE
- Calibração com `CalibratedClassifierCV`
- Otimização de threshold por F1-Score
- Interpretação de variáveis com SHAP
- Visualização e inferência com Streamlit
- Dashboard Interativo em Power BI
- Insights feitos com base em SQL

## 🛠️ Tecnologias Utilizadas ##

- **Python 3.8**
- Scikit-learn, Imbalanced-learn, XGBoost
- SHAP, Matplotlib, Pandas, NumPy
- Streamlit (interface web para predição)
- SQL (consultas para Power BI)
- Power BI (visualização de insights)
- YAML (configuração externa)
- Joblib (persistência de modelos)

---

## 🧪 Modelagem Avançada (HistGradientBoosting) ##

- **SMOTE** para balanceamento da classe minoritária
- **RandomizedSearchCV** para hiperparâmetros
- **Calibração de probabilidades** para decisões mais confiáveis
- **Threshold ótimo** ajustado para maximizar F1-Score
- **Cross-validation** para avaliar robustez
- **Curva de aprendizado**
- **SHAP values** para explicação das variáveis

---

## ⚙️ Execução dos Scripts ##

1. **pipeline.py**  
   Carrega e limpa os dados brutos.

2. **04_avanc_modelos.py**  
   Treina e calibra o modelo `HistGradientBoostingClassifier`, salva o modelo e gera gráficos de aprendizado e SHAP.

3. **05_teste.py**  
   Realiza um teste de mesa com 15 transações da “zona cinzenta” (probabilidades entre 0.20 e 0.80) para simular decisões desafiadoras.

4. **STREAMLIT.py**  
   Interface web com duas opções de uso:
   - Previsão individual (formulário manual)
   - Previsão em lote (upload de CSV)

---

## 🧠 Teste de Mesa (zona cinzenta) ##

O script `05_teste.py` foca nas transações com **probabilidades intermediárias**, onde há maior incerteza. Ele:

- Seleciona amostras balanceadas entre fraudes e não fraudes
- Apresenta cada caso com suas features
- Informa acertos e erros finais

---

## 📊 Power BI + SQL Insights ##

O script `07_INSQL_py` fornece **15 consultas SQL poderosas**, incluindo:

- Distribuição de fraudes por hora
- Faixa de valor com maior percentual de fraude
- Outliers e seus comportamentos
- Diferença entre fraudes acima/abaixo da mediana
- Transações suspeitas com score alto

Esses insights alimentam relatórios visuais no **Power BI** com indicadores decisivos.

---

## 🌐 Interface Web com Streamlit ##

## bash ##

streamlit run src/STREAMLIT.py

## Requisitos:
As dependências necessárias para rodar o projeto estão listadas no arquivo requirements.txt:

pandas
scikit-learn
xgboost
lightgbm
imbalanced-learn
joblib
streamlit
matplotlib
seaborn
numpy
Contribuição
Se você quiser contribuir para o projeto, sinta-se à vontade para abrir issues e pull requests.

## Licença ##
Este projeto está licenciado sob a licença MIT - consulte o arquivo LICENSE para mais detalhes.

javascript
Copiar
Editar

Esse `README.md` descreve claramente o projeto, como instalá-lo e rodá-lo, além de explicar a estrutura do código e os modelos de Machine Learning usados.

## Autor:
Desenvolvido por Isaac Araújo
📧 isaac.eudes2006@gmail.com

## 🔗 Projeto educacional focado em aprendizado profundo de Ciência de Dados.