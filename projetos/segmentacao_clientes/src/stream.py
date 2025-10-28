import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Configurações iniciais
st.set_page_config(page_title="Segmentação de Clientes", layout="wide")
st.title("🔍 Análise de Segmentação de Clientes")
st.sidebar.title("Opções de Análise")

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    file_path = r"C:\Github\data-science\projetos\segmentacao_clientes\data\processed\processed.csv"
    clustered_path = r"C:\Github\data-science\projetos\segmentacao_clientes\data\processed\clustered_data.csv"
    df = pd.read_csv(file_path)
    clustered_data = pd.read_csv(clustered_path)
    return df, clustered_data

# Carregar os dados
df, clustered_data = carregar_dados()

# Controle de visualização dos clusters
exibir_clusters = st.sidebar.checkbox("Ativar Visualização de Clusters", value=False)

if not exibir_clusters:
    st.subheader("📋 Dados Originais")
    st.dataframe(df)
    
    # Estatísticas descritivas do dataset original
    if st.sidebar.checkbox("Mostrar Estatísticas Descritivas"):
        st.subheader("📊 Estatísticas Descritivas dos Dados Originais")
        estatisticas_originais = df.describe().rename(index={
            "count": "Contagem",
            "mean": "Média",
            "std": "Desvio Padrão",
            "min": "Mínimo",
            "25%": "1º Quartil",
            "50%": "Mediana",
            "75%": "3º Quartil",
            "max": "Máximo",
        })
        st.dataframe(estatisticas_originais)
else:
    # Mostrar dados com clusters atribuídos
    st.subheader("📋 Dados Segmentados (Clusters Ativados)")
    st.dataframe(clustered_data)

    # Estatísticas descritivas traduzidas
    if st.sidebar.checkbox("Mostrar Estatísticas Descritivas dos Clusters"):
        st.subheader("📊 Estatísticas Descritivas dos Clusters")
        estatisticas_traduzidas = clustered_data.describe().rename(index={
            "count": "Contagem",
            "mean": "Média",
            "std": "Desvio Padrão",
            "min": "Mínimo",
            "25%": "1º Quartil",
            "50%": "Mediana",
            "75%": "3º Quartil",
            "max": "Máximo",
        })
        st.dataframe(estatisticas_traduzidas)

    # Filtro para cluster específico
    st.sidebar.subheader("Filtrar por Cluster")
    clusters_unicos = clustered_data["cluster"].unique()
    cluster_escolhido = st.sidebar.selectbox("Selecione o Cluster", clusters_unicos)
    dados_filtrados = clustered_data[clustered_data["cluster"] == cluster_escolhido]

    # Exibir detalhes do cluster selecionado
    st.subheader(f"📊 Detalhes do Cluster {cluster_escolhido}")
    st.dataframe(dados_filtrados.describe().rename(index={
        "count": "Contagem",
        "mean": "Média",
        "std": "Desvio Padrão",
        "min": "Mínimo",
        "25%": "1º Quartil",
        "50%": "Mediana",
        "75%": "3º Quartil",
        "max": "Máximo",
    }))

    # Escolher gráficos
    st.sidebar.subheader("Escolha os Gráficos")
    if st.sidebar.checkbox("Visualizar Distribuições"):
        st.subheader(f"📈 Distribuição das Variáveis - Cluster {cluster_escolhido}")
        colunas_numericas = clustered_data.select_dtypes(include=["float64", "int64"]).columns
        for coluna in colunas_numericas:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(dados_filtrados[coluna], bins=20, kde=True, ax=ax)
            ax.set_title(f"Distribuição de {coluna} - Cluster {cluster_escolhido}")
            st.pyplot(fig)

    if st.sidebar.checkbox("Visualizar Gráfico de Barras"):
        st.subheader("📊 Gráfico de Barras")
        colunas_categoricas = clustered_data.select_dtypes(include=["object"]).columns
        for coluna in colunas_categoricas:
            fig, ax = plt.subplots(figsize=(8, 6))
            dados_filtrados[coluna].value_counts().plot(kind="bar", ax=ax, color="skyblue")
            ax.set_title(f"Distribuição de {coluna} - Cluster {cluster_escolhido}")
            st.pyplot(fig)

    if st.sidebar.checkbox("Visualizar Gráfico PCA"):
        st.subheader("📉 Visualização Reduzida (PCA)")
        variaveis_numericas = clustered_data.select_dtypes(include=["float64", "int64"]).columns
        pca = PCA(n_components=2)
        pca_resultados = pca.fit_transform(clustered_data[variaveis_numericas])
        clustered_data["PCA1"] = pca_resultados[:, 0]
        clustered_data["PCA2"] = pca_resultados[:, 1]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=clustered_data["PCA1"],
            y=clustered_data["PCA2"],
            hue=clustered_data["cluster"],
            palette="tab10",
            alpha=0.7
        )
        ax.set_title("Visualização dos Clusters com PCA")
        st.pyplot(fig)

st.sidebar.markdown("✅ Personalize sua análise com os filtros!")
