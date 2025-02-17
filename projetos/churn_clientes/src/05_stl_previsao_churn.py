import streamlit as st
import pandas as pd
import joblib
import yaml
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Carregar o arquivo de configuração
config_path = r'C:/Github/data-science/projetos/churn_clientes/config/config.yaml'
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Carregar o modelo e o preprocessador
model_path = config['models']['directory'] + '/logistic_regression_model.joblib'
scaler_path = config['preprocessors']['scaler_path']
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Função para realizar a previsão de churn
def predict_churn(input_data):
    # Escalonar os dados
    input_scaled = scaler.transform([input_data])
    # Fazer a previsão
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[:, 1]
    return prediction[0], prediction_proba[0]

# Página inicial
st.title('Previsão de Churn de Clientes')
st.write("Preencha os campos abaixo para prever se o cliente irá sair ou ficar.")

# Campos de entrada (alterar conforme suas colunas e requisitos)
input_data = []
input_data.append(st.number_input("Idade", min_value=18, max_value=100, value=30))
input_data.append(st.selectbox("Status de Assinatura", ['Ativo', 'Inativo']))
input_data.append(st.number_input("Gasto Mensal", min_value=0, value=50))
input_data.append(st.selectbox("Tipo de Contrato", ['Mensal', 'Anual']))
# Adicione mais campos conforme necessário com base nas suas features

# Quando o botão for pressionado, predizimos o churn
if st.button('Prever Churn'):
    # Preparar os dados de entrada
    input_data_processed = process_input_data(input_data)  # Função para processar os dados de entrada, como One-Hot Encoding
    prediction, prediction_proba = predict_churn(input_data_processed)

    # Exibir o resultado
    if prediction == 1:
        st.write("O cliente **irá sair**. Probabilidade de Churn: {:.2f}%".format(prediction_proba * 100))
    else:
        st.write("O cliente **irá ficar**. Probabilidade de Churn: {:.2f}%".format((1 - prediction_proba) * 100))

# Função de processamento de dados de entrada (exemplo de como processar)
def process_input_data(input_data):
    # Exemplo simples de codificação e tratamento, altere conforme a necessidade
    # Aqui podemos usar técnicas como One-Hot Encoding ou quaisquer transformações necessárias.
    processed_data = []
    # Adapte o processamento para os dados específicos
    if input_data[1] == 'Ativo':
        processed_data.append(1)  # Exemplo: 1 para ativo
    else:
        processed_data.append(0)  # 0 para inativo

    if input_data[3] == 'Anual':
        processed_data.append(1)  # Exemplo: 1 para contrato anual
    else:
        processed_data.append(0)  # 0 para mensal

    # Adicionar outros campos conforme sua lógica de pré-processamento

    processed_data.extend(input_data[:1])  # Adiciona os outros dados diretamente (como idade)
    
    return processed_data
