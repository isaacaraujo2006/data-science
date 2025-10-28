import pandas as pd

# Caminho do arquivo CSV
data_path = "C:/Github/data-science/projetos/churn_clientes/data/raw/rclientes.csv"

# Carregar o dataset
df = pd.read_csv(data_path)

# Verificar os nomes atuais das colunas
print("Nomes das colunas atuais:")
print(df.columns)

# Traduzir os nomes das colunas (ajustado para 14 colunas)
df.columns = [
    'id_cliente', 'sobrenome', 'pontuacao_credito', 'geografia', 'genero', 'idade',
    'tempo_relacionamento', 'saldo', 'numero_produtos', 'cartao_credito', 'membro_ativo', 'salario_estimado',
    'coluna_extra1', 'coluna_extra2'  # Ajuste de acordo com as colunas reais do seu DataFrame
]

# Verificar os tipos de dados das colunas
tipos_dados = df.dtypes

# Exibir as colunas traduzidas e seus respectivos tipos
print("Nomes das colunas traduzidas:")
print(df.columns)
print("\nTipos de dados das colunas:")
print(tipos_dados)
