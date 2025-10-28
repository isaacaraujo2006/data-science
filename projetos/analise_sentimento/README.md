# Análise de Sentimentos

Este projeto tem como objetivo realizar uma análise de sentimentos em textos, classificando-os como positivos, negativos ou neutros. Para isso, utilizamos técnicas de Processamento de Linguagem Natural (NLP) e modelos de Machine Learning.

## Estrutura do Projeto

A estrutura do projeto é a seguinte:

```
analise_sentimentos/
├── data/
│   ├── raw/  # Dados brutos
│   ├── processed/  # Dados processados
│   └── predictions/  # Arquivos de previsões
├── models/
│   └── final_model.joblib  # Modelo final treinado
├── preprocessors/
│   └── vectorizer.joblib  # Vetorizador utilizado para transformação dos textos
├── src/
│   ├── app.py  # Código da aplicação Streamlit para análise de sentimentos
│   ├── train.py  # Script para treinamento dos modelos
│   ├── preprocess.py  # Módulo de pré-processamento dos dados
│   └── predict.py  # Script de previsão de sentimentos
├── logs/
│   └── classification_reports.json  # Relatórios de classificação de cada modelo
├── README.md  # Documentação do projeto
├── requirements.txt  # Dependências do projeto
└── config/
    └── config.yaml  # Arquivo de configuração do projeto
```

## Instalação

### Pré-requisitos:

- Python 3.8 ou superior
- Pip instalado

### Instalar Dependências:

Para instalar as dependências do projeto, crie um ambiente virtual e execute o comando:

```
pip install -r requirements.txt
```

## Rodar a Aplicação

Para rodar a aplicação Streamlit e realizar a análise de sentimentos, execute o comando abaixo:

```
streamlit run src/app.py
```

## Descrição do Código

1. **Pré-processamento:**
   - O arquivo de dados brutos é carregado, e os textos passam por limpeza (remoção de stopwords, stemming, lematização, etc.).
   - A vetorizacao é realizada utilizando TF-IDF ou CountVectorizer.

2. **Treinamento de Modelos:**
   - Diferentes modelos de Machine Learning, como Random Forest, Logistic Regression, XGBoost e LightGBM, são treinados.
   - O modelo final é escolhido com base no desempenho nos dados de teste e salvo como `final_model.joblib`.

3. **Previsão de Sentimentos:**
   - A aplicação Streamlit permite que o usuário insira textos e visualize a classificação de sentimentos em tempo real.
   - Relatórios de classificação detalhados (precisão, recall, f1-score) são salvos em `logs/classification_reports.json`.

## Requisitos

As dependências necessárias para rodar o projeto estão listadas no arquivo `requirements.txt`:

```
pandas
scikit-learn
xgboost
lightgbm
nltk
joblib
streamlit
matplotlib
seaborn
numpy
```

## Contribuição

Se você quiser contribuir para o projeto, sinta-se à vontade para abrir issues e pull requests.

## Licença
Este projeto está licenciado sob a licença MIT - consulte o arquivo LICENSE para mais detalhes.

javascript
Copiar
Editar

Esse `README.md` descreve claramente o projeto, como instalá-lo e rodá-lo, além de explicar a estrutura do código e os modelos de Machine Learning usados.