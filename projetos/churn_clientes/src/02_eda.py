import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

# Carregar o arquivo de configuração
with open(r'C:/Github/data-science/projetos/churn_clientes/config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Configurar logging
logs_dir = config['paths']['logs_dir']
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logging.basicConfig(filename=os.path.join(logs_dir, 'pipeline.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Iniciar contagem do tempo de processamento
inicio_tempo = time.time()

# Caminho do dataset
dataset_path = config['data']['raw']

# Carregar os dados
logging.info('Carregando dados')
df = pd.read_csv(dataset_path)

# Dicionário de tradução dos nomes das colunas
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

# Análise Estatística Descritiva
logging.info('Gerando análise estatística descritiva')
descricao_numericas = df.describe()
descricao_categoricas = df.describe(include=['object'])

# Visualização da distribuição dos dados
logging.info('Visualizando a distribuição dos dados')
figures_dir = config['paths']['figures_dir']
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Histograma para variáveis numéricas
for coluna in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[coluna], kde=True, bins=30)
    plt.title(f'Distribuição de {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Frequência')
    plt.savefig(os.path.join(figures_dir, f'histograma_{coluna}.png'))
    plt.close()

# Boxplot para variáveis numéricas
for coluna in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[coluna])
    plt.title(f'Boxplot de {coluna}')
    plt.xlabel(coluna)
    plt.savefig(os.path.join(figures_dir, f'boxplot_{coluna}.png'))
    plt.close()

# Análise de Correlação
logging.info('Gerando heatmap de correlação')
correlation_matrix = df.corr(numeric_only=True)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Heatmap de Correlação')
plt.savefig(os.path.join(figures_dir, 'heatmap_correlacao.png'))
plt.close()

# Análise de Churn
logging.info('Calculando taxa de churn')
taxa_churn = df['churn'].mean() * 100

# Comparar as características dos clientes que cancelaram com os que não cancelaram
logging.info('Comparando características dos clientes')
df_churn = df[df['churn'] == 1]
df_nao_churn = df[df['churn'] == 0]

# Estatísticas descritivas para churn e não churn
descricao_churn = df_churn.describe()
descricao_nao_churn = df_nao_churn.describe()

# Geração de Gráficos
logging.info('Gerando gráficos para relação entre variáveis e churn')

# Boxplot para churn e variáveis numéricas
for coluna in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='churn', y=coluna, data=df)
    plt.title(f'Boxplot de {coluna} por Churn')
    plt.xlabel('Churn')
    plt.ylabel(coluna)
    plt.savefig(os.path.join(figures_dir, f'boxplot_churn_{coluna}.png'))
    plt.close()

# Gráfico de barras para variáveis categóricas
for coluna in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=coluna, hue='churn', data=df)
    plt.title(f'Relação de {coluna} com Churn')
    plt.xlabel(coluna)
    plt.ylabel('Contagem')
    plt.savefig(os.path.join(figures_dir, f'relacao_churn_{coluna}.png'))
    plt.close()

# Gráfico de barras para distribuição das variáveis categóricas
for coluna in df.select_dtypes(include=['object']).columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=coluna, data=df)
    plt.title(f'Distribuição de {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Contagem')
    plt.savefig(os.path.join(figures_dir, f'distribuicao_{coluna}.png'))
    plt.close()

# Gráfico de dispersão para churn e variáveis numéricas
for coluna in df.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='churn', y=coluna, data=df)
    plt.title(f'Scatter plot de {coluna} por Churn')
    plt.xlabel('Churn')
    plt.ylabel(coluna)
    plt.savefig(os.path.join(figures_dir, f'scatter_churn_{coluna}.png'))
    plt.close()

# Análise de Frequência
logging.info('Gerando análise de frequência')
frequencia_churn = df['churn'].value_counts(normalize=True) * 100
frequencia_churn.plot(kind='bar', figsize=(10, 6), title='Frequência de Churn')
plt.xlabel('Churn')
plt.ylabel('Percentual')
plt.savefig(os.path.join(figures_dir, 'frequencia_churn.png'))
plt.close()

# Estatísticas Avançadas
logging.info('Gerando estatísticas avançadas')
df_grouped = df.groupby('churn').mean(numeric_only=True)
df_grouped.plot(kind='bar', figsize=(14, 10), title='Médias das Variáveis por Churn')
plt.savefig(os.path.join(figures_dir, 'medias_variaveis_churn.png'))
plt.close()

# Relatório final de qualidade dos dados e tempo de processamento
logging.info('Gerando relatório final')
tempo_processamento = time.time() - inicio_tempo
horas, rem = divmod(tempo_processamento, 3600)
minutos, segundos = divmod(rem, 60)
tempo_processamento_formatado = f"{int(horas):02}:{int(minutos):02}:{int(segundos):02}"

print("\n### RELATÓRIO FINAL ###\n")
print(f"Número total de linhas: {num_linhas}")
print(f"Taxa de Churn: {taxa_churn:.2f}%")
print("\nDescrição Estatística das Variáveis Numéricas:")
print(descricao_numericas)
print("\nDescrição Estatística das Variáveis Categóricas:")
print(descricao_categoricas)
print(f"\nTempo de processamento: {tempo_processamento_formatado}")
logging.info('Processamento concluído')
