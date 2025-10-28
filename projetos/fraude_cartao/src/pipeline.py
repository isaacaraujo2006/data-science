import pandas as pd
import yaml

# ===== 1. Carregar o config.yaml =====
with open("D:/github/data-science/projetos/fraude_cartao/config/config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# ===== 2. Caminho do arquivo bruto =====
raw_data_path = config['data']['raw']

# ===== 3. Ler o CSV =====
df = pd.read_csv(raw_data_path)

# ===== 4. Traduzir colunas (letra minúscula e sem espaços) =====
df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

# ===== 5. Exibir resultados =====
print(f"✅ Número de linhas: {df.shape[0]}")
print(f"✅ Número de colunas: {df.shape[1]}\n")

print("📋 Colunas e seus tipos:")
for col, dtype in df.dtypes.items():
    print(f"- {col}: {dtype}")
