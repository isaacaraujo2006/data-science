import pandas as pd
import yaml
import os

# Caminho config.yaml
config_path = r"D:/github/data-science/projetos/fraude_financeira/config/config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

input_path = config["data"]["raw"]
output_sample_path = os.path.join(
    os.path.dirname(input_path),
    "sample_balanceada.parquet"
)

# Carregar dataset
df = pd.read_parquet(input_path)

# Apenas registros com rótulo
df_train = df[df["isFraud"].notna()]

# Todas as fraudes
fraud_sample = df_train[df_train["isFraud"] == 1]

# Mesma quantidade de não fraude
nonfraud_sample = df_train[df_train["isFraud"] == 0].sample(
    len(fraud_sample), random_state=42
)

# Concatenar e embaralhar
sample_df = pd.concat([fraud_sample, nonfraud_sample]).sample(frac=1, random_state=42)

# Salvar
sample_df.to_parquet(output_sample_path, index=False, engine="pyarrow", compression="snappy")

print(f"✅ Amostra balanceada salva em {output_sample_path}")
print(sample_df["isFraud"].value_counts())
