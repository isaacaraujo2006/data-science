import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging

# Configuração de logging
log_file = 'C:/Github/data-science/projetos/churn_clientes/logs/eda_log.txt'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(message)s')

# Carregar o dataset
df = pd.read_parquet('C:/Github/data-science/projetos/churn_clientes/data/processed/processed.parquet')

# Log: Informações iniciais do dataset
logging.info(f"Dataset carregado com sucesso. Número de linhas: {df.shape[0]}, Número de colunas: {df.shape[1]}")
logging.info(f"Tipos de dados:\n{df.dtypes}")

# Exibir as primeiras linhas
logging.info(f"Primeiras linhas do dataset:\n{df.head()}")

# Estatísticas descritivas para variáveis numéricas
logging.info("Análise estatística descritiva das variáveis numéricas:")
logging.info(f"\n{df.describe()}")

# Estatísticas descritivas para variáveis categóricas
logging.info("Análise estatística descritiva das variáveis categóricas:")
logging.info(f"\n{df.describe(include=['object'])}")

# Cálculo da taxa de churn
churn_rate = df['exited'].mean() * 100
logging.info(f"Taxa de churn: {churn_rate:.2f}%")

# Análise de fatores que podem influenciar o churn
logging.info("Analisando fatores que podem influenciar o churn...")

# Comparar características dos clientes que cancelaram e os que não cancelaram
churn_comparison = df.groupby('exited').mean()
logging.info(f"Características dos clientes que cancelaram e não cancelaram (média das variáveis numéricas):\n{churn_comparison}")

# Geração de gráficos

# 1. Distribuição de churn
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='exited', palette='Set2')
plt.title('Distribuição de Churn')
plt.xlabel('Churn (0 = Não, 1 = Sim)')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig('C:/Github/data-science/projetos/churn_clientes/reports/figures/churn_distribution.png')
plt.close()

# 2. Histograma para variáveis numéricas
numerical_cols = ['creditscore', 'age', 'balance', 'numofproducts', 'estimatedsalary']
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True, color='skyblue', bins=20)
    plt.title(f'Distribuição de {col}')
    plt.xlabel(col)
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.savefig(f'C:/Github/data-science/projetos/churn_clientes/reports/figures/{col}_distribution.png')
    plt.close()

# 3. Boxplot para variáveis numéricas por churn
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='exited', y=col, data=df, palette='Set2')
    plt.title(f'{col} por Churn')
    plt.xlabel('Churn (0 = Não, 1 = Sim)')
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(f'C:/Github/data-science/projetos/churn_clientes/reports/figures/{col}_by_churn.png')
    plt.close()

# 4. Scatter plot entre 'age' e 'balance' com churn
plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='balance', hue='exited', data=df, palette='Set2', alpha=0.6)
plt.title('Relação entre Idade e Saldo por Churn')
plt.xlabel('Idade')
plt.ylabel('Saldo')
plt.tight_layout()
plt.savefig('C:/Github/data-science/projetos/churn_clientes/reports/figures/age_balance_by_churn.png')
plt.close()

# 5. Gráfico de barras para 'geography' por churn
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='geography', hue='exited', palette='Set2')
plt.title('Distribuição de Churn por Localização')
plt.xlabel('Localização')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig('C:/Github/data-science/projetos/churn_clientes/reports/figures/geography_by_churn.png')
plt.close()

# 6. Gráfico de barras para 'gender' por churn
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='gender', hue='exited', palette='Set2')
plt.title('Distribuição de Churn por Gênero')
plt.xlabel('Gênero')
plt.ylabel('Contagem')
plt.tight_layout()
plt.savefig('C:/Github/data-science/projetos/churn_clientes/reports/figures/gender_by_churn.png')
plt.close()

# Final de execução
logging.info("Análise exploratória concluída e gráficos salvos.")

