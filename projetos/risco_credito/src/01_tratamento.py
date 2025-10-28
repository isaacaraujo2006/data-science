import yaml
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import os

def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def salvar_parquet(df, path):
    df.to_parquet(path, index=False)
    print(f"Arquivo salvo em: {path}")

# --- Configurações ---
config_path = "D:/github/data-science/projetos/risco_credito/config/config.yaml"
config = load_config(config_path)

data_path = config['data']['raw']
raw_parquet_path = data_path.rsplit('.', 1)[0] + ".parquet"
processed_parquet_path = config['data']['processed_parquet']

# --- Carregamento flexível ---
if data_path.lower().endswith('.parquet'):
    df = pd.read_parquet(data_path)
elif data_path.lower().endswith('.xlsx') or data_path.lower().endswith('.xls'):
    df = pd.read_excel(data_path, header=1)
else:
    raise ValueError(f"Formato de arquivo não suportado: {data_path}")
print(f"Número de linhas importadas: {df.shape[0]}")

# --- Salvar o raw bruto em parquet (só se não for parquet já) ---
if not data_path.lower().endswith('.parquet'):
    salvar_parquet(df, raw_parquet_path)

# --- Renomear colunas ---
traducao_colunas = {
    'ID': 'id',
    'LIMIT_BAL': 'limite_credito',
    'SEX': 'sexo',
    'EDUCATION': 'educacao',
    'MARRIAGE': 'estado_civil',
    'AGE': 'idade',
    'PAY_0': 'pagamento_mes_0',
    'PAY_2': 'pagamento_mes_2',
    'PAY_3': 'pagamento_mes_3',
    'PAY_4': 'pagamento_mes_4',
    'PAY_5': 'pagamento_mes_5',
    'PAY_6': 'pagamento_mes_6',
    'BILL_AMT1': 'fatura_mes_1',
    'BILL_AMT2': 'fatura_mes_2',
    'BILL_AMT3': 'fatura_mes_3',
    'BILL_AMT4': 'fatura_mes_4',
    'BILL_AMT5': 'fatura_mes_5',
    'BILL_AMT6': 'fatura_mes_6',
    'PAY_AMT1': 'pagamento_mes_1',
    'PAY_AMT2': 'pagamento_mes_2',
    'PAY_AMT3': 'pagamento_mes_3',
    'PAY_AMT4': 'pagamento_mes_4',
    'PAY_AMT5': 'pagamento_mes_5',
    'PAY_AMT6': 'pagamento_mes_6',
    'default payment next month': 'inadimplente_mes_seguinte'
}

df.rename(columns=traducao_colunas, inplace=True)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Remover colunas duplicadas
duplicadas = df.columns[df.columns.duplicated()].tolist()
if duplicadas:
    print(f"Atenção: colunas duplicadas após renomeação: {duplicadas}")
    df = df.loc[:, ~df.columns.duplicated()]
    print("Colunas duplicadas removidas.")

print("\nColunas após tradução e remoção de duplicatas:")
print(df.columns.tolist())

# --- Conversão tipos + imputação faltantes ---
cat_cols = ['sexo', 'educacao', 'estado_civil']
ord_cols = [
    'pagamento_mes_0', 'pagamento_mes_2', 'pagamento_mes_3',
    'pagamento_mes_4', 'pagamento_mes_5', 'pagamento_mes_6'
]
num_cols = [
    'limite_credito', 'idade',
    'fatura_mes_1', 'fatura_mes_2', 'fatura_mes_3',
    'fatura_mes_4', 'fatura_mes_5', 'fatura_mes_6',
    'pagamento_mes_1', 'pagamento_mes_2', 'pagamento_mes_3',
    'pagamento_mes_4', 'pagamento_mes_5', 'pagamento_mes_6'
]

# Tratar NaN: categóricas para 'desconhecido', numéricas preenchem com mediana
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.add_categories('desconhecido').fillna('desconhecido')

for col in ord_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)  # AJUSTADO PARA FLOAT!

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
        df[col] = df[col].fillna(df[col].median())

# ID e target
if 'id' in df.columns:
    df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(-1).astype(int)
if 'inadimplente_mes_seguinte' in df.columns:
    df['inadimplente_mes_seguinte'] = pd.to_numeric(df['inadimplente_mes_seguinte'], errors='coerce').fillna(0).astype(int)
    df['inadimplente_mes_seguinte'] = df['inadimplente_mes_seguinte'].astype('category')

# --- Remoção de duplicados completos ---
n_antes = df.shape[0]
df = df.drop_duplicates()
n_deletados = n_antes - df.shape[0]
print(f"\nRemovidos {n_deletados} registros duplicados completos.")

# --- Tratamento valores fora do intervalo esperado ---
intervalos_esperados = {
    'id': (1, np.inf),
    'sexo': (1, 2),
    'educacao': (1, 4),
    'estado_civil': (1, 3),
    'idade': (18, 100),
    'pagamento_mes_0': (-2, 8),
    'pagamento_mes_2': (-2, 8),
    'pagamento_mes_3': (-2, 8),
    'pagamento_mes_4': (-2, 8),
    'pagamento_mes_5': (-2, 8),
    'pagamento_mes_6': (-2, 8),
    'limite_credito': (0, np.inf),
    'fatura_mes_1': (0, np.inf),
    'fatura_mes_2': (0, np.inf),
    'fatura_mes_3': (0, np.inf),
    'fatura_mes_4': (0, np.inf),
    'fatura_mes_5': (0, np.inf),
    'fatura_mes_6': (0, np.inf),
    'pagamento_mes_1': (-2, 8),
    'inadimplente_mes_seguinte': (0, 1)
}

for col, (min_v, max_v) in intervalos_esperados.items():
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        fora_intervalo = (df[col] < min_v) | (df[col] > max_v)
        n_fora = fora_intervalo.sum()
        if n_fora > 0:
            print(f"Coluna '{col}': {n_fora} valores fora do intervalo [{min_v}, {max_v}], corrigindo para limites.")
            df.loc[df[col] < min_v, col] = min_v
            df.loc[df[col] > max_v, col] = max_v

# --- Tratamento de outliers via IQR ---
def tratar_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers_inferiores = df[col] < limite_inferior
    outliers_superiores = df[col] > limite_superior
    n_out_inf = outliers_inferiores.sum()
    n_out_sup = outliers_superiores.sum()
    if n_out_inf > 0 or n_out_sup > 0:
        print(f"Coluna '{col}': {n_out_inf} outliers inferiores, {n_out_sup} outliers superiores, substituindo por limites IQR.")
        df.loc[outliers_inferiores, col] = limite_inferior
        df.loc[outliers_superiores, col] = limite_superior

for col in num_cols:
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        tratar_outliers_iqr(df, col)

# --- Padronização (opcional, CatBoost NÃO precisa) ---
padronizar = False  # True se quiser padronizar para modelos lineares
if padronizar:
    cols_para_normalizar = [col for col in num_cols if col in df.columns]
    scaler = StandardScaler()
    df[cols_para_normalizar] = scaler.fit_transform(df[cols_para_normalizar])
    joblib.dump(scaler, "D:/github/data-science/projetos/risco_credito/preprocessors/scaler.joblib")
    print("Scaler salvo em preprocessors/scaler.joblib")

print("\nDtypes finais das colunas:")
print(df.dtypes)

# --- Corrige categorias para string antes de salvar ---
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype(str)

# ... (seu código inteiro igual até aqui)

salvar_parquet(df, processed_parquet_path)

# --- Salvar também em CSV para Power BI ---
csv_path = processed_parquet_path.replace('.parquet', '.csv')
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"Arquivo CSV salvo para Power BI em: {csv_path}")

# --- Relatório simples ---
print("\nRelatório final:")
print(f"Linhas totais após limpeza: {df.shape[0]}")
for col in df.columns:
    n_na = df[col].isna().sum()
    n_duplic = df.duplicated(subset=[col], keep='first').sum()
    print(f"Coluna '{col}': faltantes={n_na}, valores duplicados={n_duplic}")

print("\nProcessamento concluído.")
