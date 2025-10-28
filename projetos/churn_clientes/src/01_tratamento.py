import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np

# Caminho do arquivo CSV
data_path = "C:/Github/data-science/projetos/churn_clientes/data/raw/rclientes.csv"

# Carregar o dataset
df = pd.read_csv(data_path)

# Verificar o número de colunas e o nome delas
print("Colunas reais no dataset:", df.columns)

# Colunas reais no seu dataset
df.columns = [
    'id_cliente', 'CustomerId', 'sobrenome', 'pontuacao_credito', 'geografia', 'genero', 'idade',
    'tempo_relacionamento', 'saldo', 'numero_produtos', 'cartao_credito', 'membro_ativo', 'salario_estimado',
    'classe'  # Ajuste conforme as colunas reais do seu DataFrame
]



# Remover duplicatas
df = df.drop_duplicates()

# Tratar valores faltantes (usando a média para colunas numéricas como exemplo)
for coluna in df.select_dtypes(include=[float, int]).columns:
    df[coluna] = df[coluna].fillna(df[coluna].mean())

# Verificar consistência dos dados
intervalos = {
    'pontuacao_credito': (300, 850),
    'idade': (18, 120)
}

for coluna, (min_val, max_val) in intervalos.items():
    if pd.api.types.is_numeric_dtype(df[coluna]):
        df.loc[df[coluna] < min_val, coluna] = min_val
        df.loc[df[coluna] > max_val, coluna] = max_val

# Remover colunas de 'geografia' e 'genero' sem valor (todas com 0)
df = df.loc[:, (df != 0).any(axis=0)]

# Manter as variáveis categóricas como strings ou categorias
df['geografia'] = df['geografia'].astype(str)
df['genero'] = df['genero'].astype(str)

# Criar uma coluna de exemplo para 'classe' (substitua isso conforme necessário)
np.random.seed(42)
df['classe'] = np.random.choice([0, 1], size=len(df))

# Criar a coluna 'nome_completo' com id_cliente (formatado como string) e sobrenome
df['nome_completo'] = df['id_cliente'].astype(str) + ' ' + df['sobrenome'].astype(str)

# Excluir a coluna 'nome_completo'
df = df.drop(columns=['nome_completo'])

# Separar características e rótulos
coluna_alvo = 'classe'
X = df.drop(columns=[coluna_alvo, 'id_cliente', 'sobrenome'])  # Não precisamos de 'id_cliente' ou 'classe'
y = df[coluna_alvo]

# Garantir que X e y não contenham valores nulos
print("Valores nulos em X antes de tratamento:", X.isnull().sum())
print("Valores nulos em y antes de tratamento:", y.isnull().sum())

# Preencher valores nulos com 0 em X e com o valor mais comum em y
X = X.fillna(0)
y = y.fillna(y.mode()[0])

# Verificar se X_train e y_train possuem o mesmo número de amostras antes de dividir
print(f"Tamanho de X: {X.shape[0]}")
print(f"Tamanho de y: {y.shape[0]}")

# Garantir que X e y tenham o mesmo número de amostras
if X.shape[0] != y.shape[0]:
    raise ValueError("O número de amostras em X e y não é consistente!")

# Dividir os dados em treino e teste antes de aplicar SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Verificar se X_train e y_train têm o mesmo número de amostras
print(f"Tamanho de X_train: {X_train.shape[0]}")
print(f"Tamanho de y_train: {y_train.shape[0]}")

# Garantir que X_train e y_train tenham o mesmo número de amostras
if X_train.shape[0] != y_train.shape[0]:
    raise ValueError("O número de amostras em X_train e y_train não é consistente!")

# Obter as colunas categóricas restantes
X_categoricas = X.drop(columns=X.select_dtypes(include=[float, int]).columns)

# Normalizar e balancear as colunas numéricas com SMOTE
smote = SMOTE(random_state=42)

# Selecionando apenas as colunas numéricas para o SMOTE
X_train_numerico = X_train.select_dtypes(include=[float, int])

# Aplicando o SMOTE apenas nas colunas numéricas de X_train
X_res_numerico, y_res = smote.fit_resample(X_train_numerico, y_train)

# Concatenar as colunas categóricas com as numéricas balanceadas
X_res = pd.DataFrame(X_res_numerico, columns=X_train_numerico.columns)
X_res_final = pd.concat([X_res, X_categoricas.reset_index(drop=True)], axis=1)

# Unir X_res_final com y_res para o DataFrame final
df_res = pd.concat([X_res_final, pd.DataFrame(y_res, columns=[coluna_alvo])], axis=1)

print("Classes balanceadas com SMOTE.")

# Relatório final
print("Número de linhas duplicadas após tratamento: ", df_res.duplicated().sum())

# Calcular os outliers com base nas colunas numéricas após o SMOTE
outliers = {}
for coluna in X_res_final.select_dtypes(include=[float, int]).columns:
    if coluna in df_res.columns:  # Garantir que a coluna ainda existe no DataFrame
        Q1 = df_res[coluna].quantile(0.25)
        Q3 = df_res[coluna].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        outliers[coluna] = len(df_res[(df_res[coluna] < lim_inf) | (df_res[coluna] > lim_sup)])

print("\nNúmero de outliers por coluna após tratamento:")
print(outliers)

# (Opcional) Remover outliers
for coluna in X_res_final.select_dtypes(include=[float, int]).columns:
    if coluna in df_res.columns:
        Q1 = df_res[coluna].quantile(0.25)
        Q3 = df_res[coluna].quantile(0.75)
        IQR = Q3 - Q1
        lim_inf = Q1 - 1.5 * IQR
        lim_sup = Q3 + 1.5 * IQR
        df_res = df_res[(df_res[coluna] >= lim_inf) & (df_res[coluna] <= lim_sup)]

print("\nNúmero de linhas após remoção de outliers:", df_res.shape[0])

# Caminho para salvar os arquivos
processed_csv_path = "C:/Github/data-science/projetos/churn_clientes/data/processed/processed.csv"
processed_parquet_path = "C:/Github/data-science/projetos/churn_clientes/data/processed/processed.parquet"

# Salvar o DataFrame no formato CSV
df_res.to_csv(processed_csv_path, index=False)

# Salvar o DataFrame no formato Parquet
df_res.to_parquet(processed_parquet_path, index=False)

print(f"Dados salvos com sucesso em:\nCSV: {processed_csv_path}\nParquet: {processed_parquet_path}")
