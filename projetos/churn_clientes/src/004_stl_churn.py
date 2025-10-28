import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Carregar o modelo e o scaler salvos
model_path = 'C:/Github/data-science/projetos/churn_clientes/models/final_model.joblib'
scaler_path = 'C:/Github/data-science/projetos/churn_clientes/preprocessors/scaler.joblib'

# Carregar o modelo e o scaler salvos
model_path = 'C:/Github/data-science/projetos/churn_clientes/models/final_model.joblib'
scaler_path = 'C:/Github/data-science/projetos/churn_clientes/preprocessors/scaler.joblib'

# Carregar os arquivos
modelo = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Função para realizar a previsão
def realizar_previsao(idade, saldo, tempo_relacionamento, salario_estimado, tem_cartao_credito):
    # Criar o DataFrame com os dados de entrada
    dados = pd.DataFrame({
        'idade': [idade],
        'saldo': [saldo],
        'tempo_relacionamento': [tempo_relacionamento],
        'salario_estimado': [salario_estimado],
        'tem_cartao_credito': [tem_cartao_credito]
    })
    
    # Normalizar os dados de entrada
    dados_normalizados = scaler.transform(dados)

    # Realizar a previsão
    previsao = modelo.predict(dados_normalizados)
    
    # Retornar o resultado mais claro
    if previsao[0] == 1:
        return "O cliente provavelmente sairá (Churn)."
    else:
        return "O cliente provavelmente ficará."

# Interface do usuário com Streamlit
st.title("Previsão de Churn de Clientes")

st.write("""
    Insira as informações do cliente abaixo para prever se ele vai desistir (Churn) ou não.
""")

# Entrada de dados pelo usuário
idade = st.number_input("Idade", min_value=18, max_value=100, value=30)
saldo = st.number_input("Saldo", min_value=0, value=50000)
tempo_relacionamento = st.number_input("Tempo de Relacionamento (anos)", min_value=0, value=5)
salario_estimado = st.number_input("Salário Estimado", min_value=0, value=55000)
tem_cartao_credito = st.selectbox("O cliente tem cartão de crédito?", [0, 1])

# Botão para realizar a previsão
if st.button("Realizar Previsão"):
    resultado = realizar_previsao(idade, saldo, tempo_relacionamento, salario_estimado, tem_cartao_credito)
    st.write(f"{resultado}")
