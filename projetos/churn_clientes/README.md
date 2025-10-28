# Previsão de Churn de Clientes

Este projeto tem como objetivo realizar previsões sobre o churn de clientes, ou seja, prever se um cliente irá cancelar o serviço ou permanecer. A previsão é feita utilizando modelos de Machine Learning, como XGBoost, LightGBM, Random Forest e Regressão Logística.

## Estrutura do Projeto

A estrutura do projeto é a seguinte:

churn_clientes/ ├── data/ │ ├── raw/ │ │ └── rclientes.csv # Dados brutos dos clientes │ ├── processed/ # Dados processados │ └── predictions/ # Arquivos de previsões ├── models/ │ └── final_model.joblib # Modelo final treinado ├── preprocessors/ │ └── scaler.joblib # Scaler utilizado para normalização dos dados ├── src/ │ └── app.py # Código da aplicação Streamlit para previsão ├── logs/ │ └── classification_reports.json # Relatórios de classificação de cada modelo ├── README.md # Documentação do projeto └── requirements.txt # Dependências do projeto


## Instalação

### Pré-requisitos:

- Python 3.8 ou superior
- Pip instalado

### Instalar Dependências:

Para instalar as dependências do projeto, crie um ambiente virtual e execute o comando:
pip install -r requirements.txt

## Rodar a Aplicação
Para rodar a aplicação Streamlit, execute o comando abaixo:
streamlit run src/app.py

## Descrição do Código

1. Pré-processamento:
- O arquivo rclientes.csv é carregado, e as colunas são tratadas.
- Variáveis categóricas são transformadas usando One-Hot Encoding.
- A normalização é feita com o uso de StandardScaler.

2. Treinamento de Modelos:
- Modelos como XGBoost, Random Forest, LightGBM e Logistic Regression são treinados para prever o churn de clientes.
- O modelo final é selecionado com base no desempenho nos dados de teste e é salvo em final_model.joblib.
- O scaler utilizado no pré-processamento é salvo como scaler.joblib.

3. Previsão de Churn:

- A aplicação Streamlit permite que o usuário insira dados de clientes, como idade, saldo, tempo de relacionamento, salário estimado e se o cliente possui cartão de crédito.
O modelo final é utilizado para realizar a previsão de churn (se o cliente irá sair ou permanecer).
Resultados
- O melhor modelo treinado é selecionado automaticamente, e o modelo final é salvo como final_model.joblib.

Relatórios de classificação detalhados para cada modelo (incluindo métricas como precisão, recall e f1-score) são salvos em logs/classification_reports.json.

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

## Licença
Este projeto está licenciado sob a licença MIT - consulte o arquivo LICENSE para mais detalhes.

javascript
Copiar
Editar

Esse `README.md` descreve claramente o projeto, como instalá-lo e rodá-lo, além de explicar a estrutura do código e os modelos de Machine Learning usados.
