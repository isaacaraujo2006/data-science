import pandas as pd
from sklearn.preprocessing import StandardScaler

# Caminho do arquivo CSV com os dados de treinamento
data_path = "C:/Github/data-science/projetos/churn_clientes/data/raw/rclientes.csv"

# Carregar o dataset
df = pd.read_csv(data_path)

# Traduzir colunas
df.rename(columns={
    'Age': 'idade',
    'Balance': 'saldo',
    'NumOfProducts': 'numero_produtos',
    'EstimatedSalary': 'salario_estimado',
    'Tenure': 'tempo_relacionamento',
    'CreditScore': 'pontuacao_credito',
    'Geography': 'geografia',
    'Gender': 'genero'
}, inplace=True)

# Gerar variáveis dummy para as colunas categóricas
df = pd.get_dummies(df, columns=['geografia', 'genero'], drop_first=True)

# Certificar que todas as colunas dummy necessárias estejam presentes
required_columns = ['geografia_Alemanha', 'genero_Masculino']
for col in required_columns:
    if col not in df.columns:
        df[col] = 0

# Selecionar apenas as 8 características esperadas
features = ['idade', 'saldo', 'numero_produtos', 'salario_estimado', 'tempo_relacionamento', 'pontuacao_credito', 'geografia_Alemanha', 'genero_Masculino']

# Reajustar o StandardScaler
scaler = StandardScaler()
scaler.fit(df[features])

# Salvar o novo StandardScaler
import joblib
joblib.dump(scaler, 'C:/Github/data-science/projetos/churn_clientes/preprocessors/scaler_8_features.joblib')
print("Novo StandardScaler com 8 características ajustado e salvo com sucesso!")
