import yaml
import pandas as pd

def load_config(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config_path = "D:/github/data-science/projetos/risco_credito/config/config.yaml"
config = load_config(config_path)

excel_path = config['data']['raw']

df = pd.read_excel(excel_path, header=1)

print(f"Número de linhas importadas: {df.shape[0]}")
print("\nColunas originais:")
print(df.columns.tolist())

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
    'default payment next month': 'inadimplente_mes_seguinte'  # <- chave corrigida
}

df.rename(columns=traducao_colunas, inplace=True)
df.columns = [col.lower().replace(' ', '_') for col in df.columns]

# Remover duplicatas APÓS a renomeação
duplicadas = df.columns[df.columns.duplicated()].tolist()
if duplicadas:
    print(f"Atenção: colunas duplicadas após renomeação: {duplicadas}")
    df = df.loc[:, ~df.columns.duplicated()]
    print("Colunas duplicadas removidas.")

print("\nColunas após tradução e remoção de duplicatas:")
print(df.columns.tolist())

col_int = ['id', 'sexo', 'educacao', 'estado_civil', 'idade',
           'pagamento_mes_0', 'pagamento_mes_2', 'pagamento_mes_3',
           'pagamento_mes_4', 'pagamento_mes_5', 'pagamento_mes_6',
           'inadimplente_mes_seguinte']

col_float = [col for col in df.columns if 'limite_credito' in col or 'fatura_mes' in col or 'pagamento_mes' in col]

for col in col_int:
    if col in df.columns:
        print(f"Convertendo coluna '{col}' para inteiro. Tipo atual: {df[col].dtype}")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    else:
        print(f"Atenção: coluna '{col}' não encontrada para conversão inteira.")

for col in col_float:
    if col in df.columns:
        print(f"Convertendo coluna '{col}' para float. Tipo atual: {df[col].dtype}")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0).astype(float)
    else:
        print(f"Atenção: coluna '{col}' não encontrada para conversão float.")

print("\nTipos de dados após conversão:")
print(df.dtypes)
