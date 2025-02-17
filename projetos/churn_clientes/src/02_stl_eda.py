import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Carregar o arquivo de configuração
with open(r'C:/Github/data-science/projetos/churn_clientes/config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Função para exibir os gráficos com base na escolha
def exibir_graficos(df, grafico_selecionado, coluna=None):
    if grafico_selecionado == "Histograma":
        st.write(f"### Histograma de {coluna}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[coluna], kde=True, bins=30, ax=ax)
        st.pyplot(fig)

    elif grafico_selecionado == "Boxplot":
        st.write(f"### Boxplot de {coluna}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=df[coluna], ax=ax)
        st.pyplot(fig)

    elif grafico_selecionado == "Heatmap de Correlação":
        st.write("### Heatmap de Correlação")
        correlation_matrix = df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)

    elif grafico_selecionado == "Gráfico de Dispersão":
        st.write(f"### Gráfico de Dispersão de {coluna} vs Churn")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x='churn', y=coluna, data=df, ax=ax)
        st.pyplot(fig)

# Função principal da aplicação Streamlit
def main():
    st.set_page_config(page_title="Análise de Churn de Clientes", layout="wide")
    
    st.title("Análise de Churn de Clientes")
    st.markdown("""
    Este dashboard permite carregar dados e visualizar diferentes gráficos sobre os clientes que cancelaram ou não seus serviços.
    Você pode escolher até 4 tipos de gráficos para visualização.
    """)

    # Divisão da página em duas colunas: uma para upload e outra para gráficos
    col1, col2 = st.columns([1, 3])

    with col1:
        st.sidebar.header("Importar Dados")
        uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)

            # Traduzir as colunas
            traducao_colunas = {
                'CustomerID': 'id_cliente',
                'Age': 'idade',
                'Gender': 'genero',
                'Tenure': 'tempo_de_assinatura',
                'Usage Frequency': 'frequencia_uso',
                'Support Calls': 'chamadas_suporte',
                'Payment Delay': 'atraso_pagamento',
                'Subscription Type': 'tipo_assinatura',
                'Contract Length': 'duracao_contrato',
                'Total Spend': 'gasto_total',
                'Last Interaction': 'ultima_interacao',
                'Churn': 'churn'
            }
            df.rename(columns=traducao_colunas, inplace=True)

    with col2:
        if uploaded_file is not None:
            st.sidebar.header("Escolha os Gráficos a Exibir")

            # Seleção de gráficos em 4 slots
            grafico_1 = st.sidebar.selectbox("Escolha o gráfico 1", ["Nenhum", "Histograma", "Boxplot", "Heatmap de Correlação", "Gráfico de Dispersão"])
            grafico_2 = st.sidebar.selectbox("Escolha o gráfico 2", ["Nenhum", "Histograma", "Boxplot", "Heatmap de Correlação", "Gráfico de Dispersão"])
            grafico_3 = st.sidebar.selectbox("Escolha o gráfico 3", ["Nenhum", "Histograma", "Boxplot", "Heatmap de Correlação", "Gráfico de Dispersão"])
            grafico_4 = st.sidebar.selectbox("Escolha o gráfico 4", ["Nenhum", "Histograma", "Boxplot", "Heatmap de Correlação", "Gráfico de Dispersão"])

            # Exibição de gráficos selecionados
            if grafico_1 != "Nenhum":
                coluna_1 = st.selectbox("Escolha a coluna para o gráfico 1", df.select_dtypes(include=[np.number]).columns)
                exibir_graficos(df, grafico_1, coluna_1)

            if grafico_2 != "Nenhum":
                coluna_2 = st.selectbox("Escolha a coluna para o gráfico 2", df.select_dtypes(include=[np.number]).columns)
                exibir_graficos(df, grafico_2, coluna_2)

            if grafico_3 != "Nenhum":
                coluna_3 = st.selectbox("Escolha a coluna para o gráfico 3", df.select_dtypes(include=[np.number]).columns)
                exibir_graficos(df, grafico_3, coluna_3)

            if grafico_4 != "Nenhum":
                coluna_4 = st.selectbox("Escolha a coluna para o gráfico 4", df.select_dtypes(include=[np.number]).columns)
                exibir_graficos(df, grafico_4, coluna_4)
        else:
            st.write("Por favor, faça o upload de um arquivo CSV para continuar.")

# Rodar a aplicação
if __name__ == "__main__":
    main()
