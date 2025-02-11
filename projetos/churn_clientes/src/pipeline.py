import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# Caminho do arquivo de configuração
config_path = r'C:/Github/data-science/projetos/churn_clientes/config/config.yaml'

# Carregar o arquivo YAML
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Obter o caminho do dataset
dataset_path = config['data']['raw']

# Carregar os dados
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

# 1. Verificar dados duplicados
num_duplicados = df.duplicated().sum()
percentual_duplicados = (num_duplicados / num_linhas) * 100

# 2. Verificar dados faltantes
num_faltantes = df.isnull().sum().sum()
percentual_faltantes = (num_faltantes / (num_linhas * df.shape[1])) * 100

# 3. Definir intervalos esperados
intervalos_esperados = {
    'idade': (0, 120),
    'tempo_de_assinatura': (0, 240),
    'frequencia_uso': (0, 100),
    'gasto_total': (0, 50000),
    'ultima_interacao': (0, 100),
    'churn': (0, 1)
}

# 4. Identificar valores fora do intervalo esperado
fora_intervalo = {}
for coluna, (min_val, max_val) in intervalos_esperados.items():
    fora_intervalo[coluna] = df[(df[coluna] < min_val) | (df[coluna] > max_val)].shape[0]

# Percentual de dados fora do intervalo
percentual_fora_intervalo = {coluna: (count / num_linhas) * 100 for coluna, count in fora_intervalo.items()}
percentual_total_fora_intervalo = sum(fora_intervalo.values()) / num_linhas * 100

# 5. Ajustar valores fora do intervalo
for coluna, (min_val, max_val) in intervalos_esperados.items():
    df[coluna] = df[coluna].clip(lower=min_val, upper=max_val)

# 6. Remover dados duplicados
df.drop_duplicates(inplace=True)

# 7. Preencher dados faltantes com a mediana
df.fillna(df.select_dtypes(include=[np.number]).median(), inplace=True)

# 8. Identificar e remover outliers (IQR)
outliers = {}
for coluna in df.select_dtypes(include=[np.number]).columns:
    Q1 = df[coluna].quantile(0.25)
    Q3 = df[coluna].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold_min = Q1 - 1.5 * IQR
    outlier_threshold_max = Q3 + 1.5 * IQR
    outliers[coluna] = df[(df[coluna] < outlier_threshold_min) | (df[coluna] > outlier_threshold_max)].shape[0]

# Percentual de outliers
percentual_outliers = {coluna: (count / num_linhas) * 100 for coluna, count in outliers.items()}
percentual_total_outliers = sum(outliers.values()) / num_linhas * 100

# 9. Normalizar e padronizar os dados
scaler = StandardScaler()
df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]).values)

# 10. Caminho correto dos arquivos de saída
processed_dir = os.path.dirname(config['data']['processed'])
csv_path = os.path.join(processed_dir, "processed.csv")
parquet_path = os.path.join(processed_dir, "processed.parquet")

# Criar diretório se não existir
if not os.path.exists(processed_dir):
    os.makedirs(processed_dir)

# 11. Salvar os arquivos
df.to_csv(csv_path, index=False)
df.to_parquet(parquet_path, index=False)

# 12. Relatório final de qualidade dos dados
print("\n### RELATÓRIO FINAL ###\n")
print(f"Número total de linhas: {num_linhas}")
print(f"Percentual de Dados Duplicados: {percentual_duplicados:.2f}%")
print(f"Percentual de Dados Faltantes: {percentual_faltantes:.2f}%")
print(f"Percentual de Dados Fora do Intervalo Esperado: {percentual_total_fora_intervalo:.2f}%")
print(f"Percentual de Outliers: {percentual_total_outliers:.2f}%")

print("\nArquivos salvos com sucesso:")
print(f"  - CSV: {csv_path}")
print(f"  - Parquet: {parquet_path}")
print("\nProcessamento concluído!\n")
