import pandas as pd
import yaml

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
