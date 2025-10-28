import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def carregar_arquivo(uploaded_file):
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)
    return df

def gerar_grafico(df, grafico_tipo, class_column, key):
    plt.clf()  # Limpar figura antes de gerar um novo gráfico
    plt.figure(figsize=(10, 6))
    
    if grafico_tipo == "Histograma":
        coluna = st.selectbox("Selecione a coluna para o histograma", df.select_dtypes(include=['float', 'int']).columns, key=f"{key}_hist")
        sns.histplot(df[coluna], kde=True)
        plt.title(f'Distribuição de {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
    
    elif grafico_tipo == "Boxplot":
        coluna = st.selectbox("Selecione a coluna para o boxplot", df.select_dtypes(include=['float', 'int']).columns, key=f"{key}_box")
        sns.boxplot(x=class_column, y=coluna, data=df)
        plt.title(f'Boxplot de {coluna} por Classe')
        plt.xlabel('Classe')
        plt.ylabel(coluna)
    
    elif grafico_tipo == "Scatter Plot":
        x_coluna = st.selectbox("Selecione a coluna X para o scatter plot", df.select_dtypes(include=['float', 'int']).columns, key=f"{key}_scatter_x")
        y_coluna = st.selectbox("Selecione a coluna Y para o scatter plot", df.select_dtypes(include=['float', 'int']).columns, key=f"{key}_scatter_y")
        sns.scatterplot(x=x_coluna, y=y_coluna, data=df)
        plt.title(f'Scatter Plot de {x_coluna} vs {y_coluna}')
        plt.xlabel(x_coluna)
        plt.ylabel(y_coluna)
    
    elif grafico_tipo == "Gráfico de Barras":
        coluna = st.selectbox("Selecione a coluna para o gráfico de barras", df.select_dtypes(include=['object']).columns, key=f"{key}_bar")
        sns.countplot(x=coluna, data=df)
        plt.title(f'Gráfico de Barras de {coluna}')
        plt.xlabel(coluna)
        plt.ylabel('Frequência')
    
    st.pyplot(plt)

st.title("Visualização de Dados Tratados")
st.write("Faça o upload de um arquivo .csv ou .parquet para visualizar gráficos.")

# Upload do arquivo
uploaded_file = st.file_uploader("Escolha um arquivo", type=["csv", "parquet"])

if uploaded_file is not None:
    df = carregar_arquivo(uploaded_file)

    st.write("Número de linhas: ", df.shape[0])
    st.write("Número de colunas: ", df.shape[1])
    st.write("Tipos de dados das colunas:")
    st.write(df.dtypes)

    # Tentar localizar a coluna "Classe" ou algo semelhante
    possible_columns = [col for col in df.columns if 'classe' in col.lower()]
    if possible_columns:
        class_column = possible_columns[0]
        st.write(f"Coluna de classe encontrada: {class_column}")
    else:
        st.write("Coluna de classe não encontrada!")

    st.write("Escolha os gráficos para visualização:")

    col1, col2 = st.columns(2)

    with col1:
        grafico_tipo1 = st.selectbox("Tipo de Gráfico 1", ["Histograma", "Boxplot", "Scatter Plot", "Gráfico de Barras"], key="grafico1")
        gerar_grafico(df, grafico_tipo1, class_column, key="grafico1")
    
    with col2:
        grafico_tipo2 = st.selectbox("Tipo de Gráfico 2", ["Histograma", "Boxplot", "Scatter Plot", "Gráfico de Barras"], key="grafico2")
        gerar_grafico(df, grafico_tipo2, class_column, key="grafico2")

    col3, col4 = st.columns(2)

    with col3:
        grafico_tipo3 = st.selectbox("Tipo de Gráfico 3", ["Histograma", "Boxplot", "Scatter Plot", "Gráfico de Barras"], key="grafico3")
        gerar_grafico(df, grafico_tipo3, class_column, key="grafico3")
    
    with col4:
        grafico_tipo4 = st.selectbox("Tipo de Gráfico 4", ["Histograma", "Boxplot", "Scatter Plot", "Gráfico de Barras"], key="grafico4")
        gerar_grafico(df, grafico_tipo4, class_column, key="grafico4")
