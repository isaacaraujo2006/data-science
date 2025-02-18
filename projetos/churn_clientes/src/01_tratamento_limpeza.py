import pandas as pd
import yaml
import numpy as np

# Carregar o arquivo de configuração (config.yaml)
with open('C:/Github/data-science/projetos/churn_clientes/config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Caminho para o arquivo de dados
data_path = config['data']['raw']

# Importar o dataset
df = pd.read_csv(data_path)

# Traduzir o nome das colunas (sem maiúsculas e sem espaços)
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Exibir as primeiras linhas para visualizar as mudanças
print(df.head())

# Verificar tipos de dados das colunas
current_dtypes = df.dtypes

# Esperados tipos de dados (baseado nas colunas)
expected_dtypes = {
    'rownumber': 'int64',
    'customerid': 'int64',
    'surname': 'object',
    'creditscore': 'int64',
    'geography': 'object',
    'gender': 'object',
    'age': 'int64',
    'tenure': 'int64',
    'balance': 'float64',
    'numofproducts': 'int64',
    'hascrcard': 'int64',
    'isactivemember': 'int64',
    'estimatedsalary': 'float64',
    'exited': 'int64',
}

# Verificando se os tipos de dados estão corretos
for column, expected_type in expected_dtypes.items():
    if column in current_dtypes:
        if current_dtypes[column] != expected_type:
            print(f'A coluna "{column}" tem o tipo de dado {current_dtypes[column]}, mas deveria ser {expected_type}.')
        else:
            print(f'A coluna "{column}" está com o tipo de dado correto: {expected_type}.')
    else:
        print(f'A coluna "{column}" não foi encontrada no dataset.')

# Calcular o IQR para identificar os outliers em colunas contínuas
def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Remover os outliers fora dos limites de IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

# Aplicar a função de remoção de outliers
outlier_columns = ['creditscore', 'age', 'balance', 'numofproducts', 'estimatedsalary']
df_cleaned = remove_outliers_iqr(df, outlier_columns)

# Verificar o número de linhas após o tratamento dos outliers
print(f"Linhas após tratamento de outliers: {len(df_cleaned)}")

# Verificar se o número de linhas diminuiu devido à remoção dos outliers
print(f"Total de linhas removidas: {len(df) - len(df_cleaned)}")

# Verificar a distribuição das colunas após o tratamento
print(df_cleaned.describe())

# Exibir as primeiras linhas do dataset limpo
print(df_cleaned.head())

# Verificar novamente os tipos de dados das colunas
current_dtypes_cleaned = df_cleaned.dtypes
for column, expected_type in expected_dtypes.items():
    if column in current_dtypes_cleaned:
        if current_dtypes_cleaned[column] != expected_type:
            print(f'A coluna "{column}" tem o tipo de dado {current_dtypes_cleaned[column]}, mas deveria ser {expected_type}.')
        else:
            print(f'A coluna "{column}" está com o tipo de dado correto: {expected_type}.')
    else:
        print(f'A coluna "{column}" não foi encontrada no dataset.')

# Definir o caminho para salvar os arquivos processados
processed_csv_path = 'C:/Github/data-science/projetos/churn_clientes/data/processed/processed.csv'
processed_parquet_path = 'C:/Github/data-science/projetos/churn_clientes/data/processed/processed.parquet'

# Salvar o dataset limpo como arquivo CSV
df_cleaned.to_csv(processed_csv_path, index=False)
print(f"Dataset salvo como CSV em: {processed_csv_path}")

# Salvar o dataset limpo como arquivo Parquet
df_cleaned.to_parquet(processed_parquet_path, index=False)
print(f"Dataset salvo como Parquet em: {processed_parquet_path}")
