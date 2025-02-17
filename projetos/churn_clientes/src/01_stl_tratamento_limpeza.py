import streamlit as st
import pandas as pd
import numpy as np
import os
import yaml
import time
from io import BytesIO

# Função para carregar o arquivo de configuração
def load_config():
    config_path = r'C:/Github/data-science/projetos/churn_clientes/config/config.yaml'
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

# Função para tratar e limpar os dados
def clean_data(df):
    # Dicionário de tradução dos nomes das colunas para português
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

    # Traduzindo os nomes das colunas
    df.rename(columns=traducao_colunas, inplace=True)

    # Número total de linhas
    num_linhas = df.shape[0]

    # 1. Verificar dados duplicados
    num_duplicados = df.duplicated().sum()

    # 2. Verificar dados faltantes
    num_faltantes = df.isnull().sum().sum()

    # 3. Definir intervalos esperados
    intervalos_esperados = {
        'idade': (0, 120),
        'tempo_de_assinatura': (0, 240),
        'frequencia_uso': (0, 100),
        'gasto_total': (0, 50000),
        'ultima_interacao': (0, 100),
        'churn': (0, 1)
    }

    # 4. Ajustar valores fora do intervalo
    for coluna, (min_val, max_val) in intervalos_esperados.items():
        df[coluna] = df[coluna].clip(lower=min_val, upper=max_val)

    # 5. Remover dados duplicados
    df.drop_duplicates(inplace=True)

    # 6. Preencher dados faltantes com a mediana
    df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)

    return df, num_linhas, num_duplicados, num_faltantes

# Função para salvar os dados processados em .csv e .parquet
def save_processed_files(df, config):
    # Caminho correto dos arquivos de saída
    processed_dir = os.path.dirname(config['data']['processed'])
    processed_csv_path = os.path.join(processed_dir, "processed.csv")
    processed_parquet_path = os.path.join(processed_dir, "processed.parquet")

    # Criar diretório se não existir
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    # Salvar arquivos CSV e Parquet
    df.to_csv(processed_csv_path, index=False)
    df.to_parquet(processed_parquet_path, index=False)

    return processed_csv_path, processed_parquet_path

# Função principal da interface do Streamlit
def main():
    st.set_page_config(page_title="Tratamento e Limpeza de Dados", layout="wide")

    # Título da aplicação
    st.title("Tratamento e Limpeza de Dados")

    # Carregar configurações
    config = load_config()

    # Carregar o arquivo não tratado
    uploaded_file = st.file_uploader("Escolha um arquivo CSV para importar", type="csv")
    if uploaded_file is not None:
        # Carregar o arquivo CSV
        df = pd.read_csv(uploaded_file)
        st.write("Dados originais carregados:")
        st.dataframe(df.head())

        # Realizar o tratamento e limpeza
        st.write("Processando e limpando os dados...")
        start_time = time.time()
        df_cleaned, num_linhas, num_duplicados, num_faltantes = clean_data(df)
        processing_time = time.time() - start_time

        # Exibir relatório
        st.subheader("Relatório de Tratamento de Dados")
        st.write(f"Número total de linhas: {num_linhas}")
        st.write(f"Percentual de Dados Duplicados: {num_duplicados / num_linhas * 100:.2f}%")
        st.write(f"Percentual de Dados Faltantes: {num_faltantes / (num_linhas * df.shape[1]) * 100:.2f}%")
        st.write(f"Tempo de Processamento: {processing_time:.2f} segundos")

        # Exibir dados limpos
        st.write("Dados após o tratamento:")
        st.dataframe(df_cleaned.head())

        # Botão para salvar os dados processados
        if st.button("Salvar Arquivos Processados"):
            processed_csv_path, processed_parquet_path = save_processed_files(df_cleaned, config)
            st.success(f"Arquivos salvos com sucesso!")

            # Preparar os arquivos para download
            with open(processed_csv_path, "rb") as file:
                st.download_button(
                    label="Baixar Arquivo CSV",
                    data=file,
                    file_name="processed.csv",
                    mime="application/octet-stream"
                )

            with open(processed_parquet_path, "rb") as file:
                st.download_button(
                    label="Baixar Arquivo Parquet",
                    data=file,
                    file_name="processed.parquet",
                    mime="application/octet-stream"
                )

if __name__ == "__main__":
    main()
