import pandas as pd
import yaml
import os

# Caminho do config.yaml
config_path = r"D:/github/data-science/projetos/fraude_financeira/config/config.yaml"

# Carregar config
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Caminhos de entrada/saída
input_sample_path = config["data"]["processed_parquet"].replace(
    "processed.parquet", "sample_balanceada.parquet"
)
output_path_parquet = config["data"]["processed_parquet"]

print(f"📂 Lendo amostra balanceada de: {input_sample_path}")
df = pd.read_parquet(input_sample_path)

# 1️⃣ Normalizar nomes para minúsculo e underscore
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# 2️⃣ Dicionário de tradução (exemplo parcial)
traducao_colunas = {
    "transactionid": "id_transacao",
    "isfraud": "fraude",
    "transactiondt": "tempo_transacao",
    "transactionamt": "valor_transacao",
    "productcd": "codigo_produto",
    "card1": "cartao_1",
    "card2": "cartao_2",
    "card3": "cartao_3",
    "card4": "tipo_cartao",
    "card5": "cartao_5",
    "card6": "categoria_cartao",
    "addr1": "endereco_1",
    "addr2": "endereco_2",
    "p_emaildomain": "email_comprador",
    "r_emaildomain": "email_recebedor"
    # Adicionar mais conforme necessidade
}
df.rename(columns=traducao_colunas, inplace=True)

# 3️⃣ Ajuste de tipos
for col in df.columns:
    if df[col].dtype == "object":
        if df[col].nunique() < 50:
            df[col] = df[col].astype("category")
    elif pd.api.types.is_float_dtype(df[col]):
        # Se todos valores não-nulos são inteiros
        if (df[col].dropna() % 1 == 0).all():
            df[col] = df[col].astype("Int64")

# 4️⃣ Salvar arquivo processado
os.makedirs(os.path.dirname(output_path_parquet), exist_ok=True)
df.to_parquet(output_path_parquet, index=False, engine="pyarrow", compression="snappy")

print(f"✅ Dataset processado salvo em: {output_path_parquet}")
print(f"📊 Formato final: {df.shape}")
