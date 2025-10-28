import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import logging

# Configuração de logging
logs_dir = r"C:\Github\data-science\projetos\segmentacao_clientes\logs"
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, "eda.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Iniciando análise exploratória de dados (EDA).")

# Configuração dos diretórios para salvar gráficos
figures_dir = r"C:\Github\data-science\projetos\segmentacao_clientes\reports\figures"
os.makedirs(figures_dir, exist_ok=True)

# Carregar o dataset processado
file_path = r"C:\Github\data-science\projetos\segmentacao_clientes\data\processed\processed.csv"
df = pd.read_csv(file_path)

# Informar o número de linhas, colunas e tipos de dados
num_linhas, num_colunas = df.shape
tipos_dados = df.dtypes
print(f"Número de linhas: {num_linhas}")
print(f"Número de colunas: {num_colunas}")
print("Tipos de dados:")
print(tipos_dados, "\n")
logging.info(f"Número de linhas: {num_linhas} | Número de colunas: {num_colunas}")
logging.info("Tipos de dados registrados.")

# Estatísticas descritivas
logging.info("Gerando estatísticas descritivas das variáveis numéricas.")
estatisticas_numericas = df.describe()
print("Estatísticas descritivas das variáveis numéricas:")
print(estatisticas_numericas, "\n")

logging.info("Gerando estatísticas descritivas das variáveis categóricas.")
estatisticas_categoricas = df.describe(include=["object", "category"])
print("Estatísticas descritivas das variáveis categóricas:")
print(estatisticas_categoricas, "\n")

# Salvar estatísticas descritivas
estatisticas_numericas.to_csv(os.path.join(figures_dir, "estatisticas_numericas.csv"))
estatisticas_categoricas.to_csv(os.path.join(figures_dir, "estatisticas_categoricas.csv"))
logging.info("Estatísticas descritivas salvas no diretório especificado.")

# Análise exploratória e geração de gráficos
logging.info("Iniciando a geração de gráficos.")

# 1. Histograma para variáveis numéricas
for coluna in df.select_dtypes(include=["float64", "int64"]).columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[coluna], bins=30, kde=True)
    plt.title(f"Distribuição de {coluna}")
    plt.xlabel(coluna)
    plt.ylabel("Frequência")
    plt.savefig(os.path.join(figures_dir, f"{coluna}_histograma.png"))
    plt.close()
    logging.info(f"Gráfico de histograma salvo para a coluna: {coluna}")

# 2. Boxplot para variáveis numéricas
for coluna in df.select_dtypes(include=["float64", "int64"]).columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[coluna])
    plt.title(f"Boxplot de {coluna}")
    plt.xlabel(coluna)
    plt.savefig(os.path.join(figures_dir, f"{coluna}_boxplot.png"))
    plt.close()
    logging.info(f"Gráfico de boxplot salvo para a coluna: {coluna}")

# 3. Gráfico de barras para variáveis categóricas
for coluna in df.select_dtypes(include=["object", "category"]).columns:
    plt.figure(figsize=(8, 6))
    df[coluna].value_counts().plot(kind="bar")
    plt.title(f"Distribuição de {coluna}")
    plt.xlabel(coluna)
    plt.ylabel("Frequência")
    plt.savefig(os.path.join(figures_dir, f"{coluna}_barras.png"))
    plt.close()
    logging.info(f"Gráfico de barras salvo para a coluna: {coluna}")

# 4. Matriz de correlação (heatmap)
plt.figure(figsize=(12, 8))
correlacoes = df.corr()
sns.heatmap(correlacoes, annot=True, cmap="coolwarm")
plt.title("Matriz de Correlação")
plt.savefig(os.path.join(figures_dir, "matriz_correlacao.png"))
plt.close()
logging.info("Gráfico de matriz de correlação salvo.")

# 5. Scatter plots para variáveis selecionadas (exemplo)
variaveis_scatter = [("gastos_vinhos", "gastos_carnes"), ("renda", "recencia")]
for var_x, var_y in variaveis_scatter:
    if var_x in df.columns and var_y in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[var_x], y=df[var_y])
        plt.title(f"Relação entre {var_x} e {var_y}")
        plt.xlabel(var_x)
        plt.ylabel(var_y)
        plt.savefig(os.path.join(figures_dir, f"{var_x}_vs_{var_y}_scatter.png"))
        plt.close()
        logging.info(f"Scatter plot salvo para {var_x} vs {var_y}.")

# 6. Visualização com PCA (adicional)
logging.info("Aplicando PCA para visualização reduzida.")
variaveis_numericas = df.select_dtypes(include=["float64", "int64"]).columns
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df[variaveis_numericas])

plt.figure(figsize=(8, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.5, c="blue")
plt.title("Visualização de Variáveis Reduzidas com PCA")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.savefig(os.path.join(figures_dir, "visualizacao_pca.png"))
plt.close()
logging.info("Gráfico de visualização com PCA salvo.")

# Logs de conclusão
logging.info("Análise exploratória de dados (EDA) concluída com sucesso.")
print("\nEDA concluída. Gráficos e relatórios salvos nos diretórios especificados.")
