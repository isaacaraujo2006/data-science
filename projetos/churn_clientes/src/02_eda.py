import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Configuração de logs
log_file = 'C:/Github/data-science/projetos/churn_clientes/logs/eda_analysis.log'
logging.basicConfig(filename=log_file,
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para salvar gráficos
def save_plot(fig, filename):
    fig.savefig(filename)
    plt.close(fig)
    logging.info(f'Gráfico salvo em: {filename}')

# Verifica se o arquivo existe antes de carregar
file_path = 'C:/Github/data-science/projetos/churn_clientes/data/processed/processed.parquet'
if os.path.exists(file_path):
    df = pd.read_parquet(file_path)
    logging.info(f"Dataset carregado: {file_path}")
else:
    logging.error(f"Arquivo não encontrado: {file_path}")
    raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

# Verificar as colunas do DataFrame e registrar
logging.info(f"Colunas do DataFrame: {df.columns.tolist()}")
print(f"Colunas do DataFrame: {df.columns.tolist()}")  # Adicionando print para verificação no terminal

# Tentar localizar a coluna "Classe" ou algo semelhante
possible_columns = [col for col in df.columns if 'classe' in col.lower()]
if not possible_columns:
    logging.error("Coluna 'Classe' não encontrada no dataset!")
    raise KeyError("Coluna 'Classe' não encontrada no dataset!")
else:
    # Se encontrado, utiliza a primeira correspondência
    class_column = possible_columns[0]
    logging.info(f"Coluna de classe encontrada: {class_column}")

    # Calcular a taxa de churn
    churn_rate = df[class_column].mean()
    logging.info(f"Taxa de churn: {churn_rate:.2f}")

# Gerar informações iniciais sobre o dataset
logging.info(f"Número de linhas: {df.shape[0]}, Número de colunas: {df.shape[1]}")
logging.info(f"Tipos de dados: \n{df.dtypes}")
logging.info(f"Estatísticas Descritivas: \n{df.describe(include='all')}")
logging.info(f"Valores nulos por coluna: \n{df.isnull().sum()}")

# Análise de características dos clientes que cancelaram vs. não cancelaram
churned = df[df[class_column] == 1]
not_churned = df[df[class_column] == 0]
logging.info(f"Características dos clientes que cancelaram (Classe=1): {churned.describe()}")
logging.info(f"Características dos clientes que não cancelaram (Classe=0): {not_churned.describe()}")

# Gráficos
def generate_plots():
    # Distribuição de variáveis numéricas
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        fig = plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')
        save_plot(fig, f'C:/Github/data-science/projetos/churn_clientes/reports/figures/distribuicao_{col}.png')

        # Boxplot das variáveis numéricas por Classe
        fig = plt.figure(figsize=(10, 6))
        sns.boxplot(x=class_column, y=col, data=df)
        plt.title(f'Boxplot de {col} por Classe')
        plt.xlabel('Classe')
        plt.ylabel(col)
        save_plot(fig, f'C:/Github/data-science/projetos/churn_clientes/reports/figures/boxplot_{col}_Classe.png')

        # Scatter plot para variáveis numéricas relacionadas ao Classe
        fig = plt.figure(figsize=(10, 6))
        sns.scatterplot(x=col, y=class_column, data=df)
        plt.title(f'Scatter plot de {col} vs Classe')
        plt.xlabel(col)
        plt.ylabel('Classe')
        save_plot(fig, f'C:/Github/data-science/projetos/churn_clientes/reports/figures/scatter_{col}_Classe.png')

    # Correlação entre variáveis numéricas
    corr_matrix = df[num_cols].corr()
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    plt.title('Matriz de Correlação entre Variáveis Numéricas')
    save_plot(fig, f'C:/Github/data-science/projetos/churn_clientes/reports/figures/corr_matrix.png')

    # Comparação entre Classe e variáveis numéricas (Violin plot)
    for col in num_cols:
        fig = plt.figure(figsize=(10, 6))
        sns.violinplot(x=class_column, y=col, data=df)
        plt.title(f'Violin plot de {col} por Classe')
        plt.xlabel('Classe')
        plt.ylabel(col)
        save_plot(fig, f'C:/Github/data-science/projetos/churn_clientes/reports/figures/violin_{col}_Classe.png')

# Gerar e salvar os gráficos
generate_plots()

logging.info("Análise exploratória concluída com sucesso.")
