import pandas as pd
import numpy as np
import yaml
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ===============================
# 1️⃣ Leitura de Configuração
# ===============================
config_path = r"D:/github/data-science/projetos/fraude_financeira/config/config.yaml"

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

input_sample_path = config["data"]["processed_parquet"].replace(
    "processed.parquet", "sample_balanceada.parquet"
)
output_path_parquet = config["data"]["processed_parquet"]
output_metrics_path = os.path.join(config["metrics"]["directory"], "analise_features.parquet")

print(f"📂 Lendo amostra balanceada de: {input_sample_path}")
df = pd.read_parquet(input_sample_path)

# ===============================
# 2️⃣ Normalização de nomes + tradução
# ===============================
df.columns = [col.lower().replace(" ", "_") for col in df.columns]
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
}
df.rename(columns=traducao_colunas, inplace=True)

# ===============================
# 3️⃣ Flags de nulos
# ===============================
for col in df.columns:
    df[f"{col}_isnull"] = df[col].isna().astype(int)

# ===============================
# 4️⃣ Remover colunas inúteis
# ===============================
colunas_remover = []
relatorio = []
num_linhas = len(df)

for col in df.columns:
    nulos = df[col].isna().sum()
    perc_nulos = nulos / num_linhas * 100
    nunique = df[col].nunique(dropna=True)

    if nunique <= 1:
        colunas_remover.append(col)
    if perc_nulos > 99:
        colunas_remover.append(col)

    relatorio.append({
        "coluna": col,
        "nulos": nulos,
        "%_nulos": perc_nulos,
        "nunique": nunique
    })

colunas_remover = list(set(colunas_remover))
df.drop(columns=colunas_remover, inplace=True)
print(f"🗑️ Colunas removidas por baixa variância/nulos: {len(colunas_remover)}")

# ===============================
# 5️⃣ Preenchimento inteligente
# ===============================
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=["object", "category"]).columns:
    df[col].fillna("desconhecido", inplace=True)
    df[col] = df[col].astype("category")

# ===============================
# 6️⃣ Encoding de categóricas
# ===============================
label_encoders = {}
for col in df.select_dtypes(include=["category", "object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ===============================
# 7️⃣ Tratamento de outliers
# ===============================
for col in df.select_dtypes(include=[np.number]).columns:
    if col != "fraude":
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        limite_inf = q1 - 1.5 * iqr
        limite_sup = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < limite_inf, limite_inf,
                          np.where(df[col] > limite_sup, limite_sup, df[col]))

# ===============================
# 8️⃣ Features derivadas
# ===============================
if "tempo_transacao" in df.columns:
    df["dias_desde_transacao"] = df["tempo_transacao"] / (60 * 60 * 24)
if "valor_transacao" in df.columns:
    df["valor_log"] = np.log1p(df["valor_transacao"])

# ===============================
# 9️⃣ Remover baixa variância
# ===============================
num_df = df.select_dtypes(include=[np.number])
selector = VarianceThreshold(threshold=0.01)
selector.fit(num_df)

colunas_baixa_var = num_df.columns[~selector.get_support()]
if len(colunas_baixa_var) > 0:
    df.drop(columns=colunas_baixa_var, inplace=True)
    print(f"🔻 Colunas removidas por baixa variância: {len(colunas_baixa_var)}")

# ===============================
# 🔟 Remover alta correlação
# ===============================
corr_matrix = df.select_dtypes(include=[np.number]).corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
col_corr_remover = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
if col_corr_remover:
    df.drop(columns=col_corr_remover, inplace=True)
    print(f"🔻 Colunas removidas por alta correlação (> 0.95): {len(col_corr_remover)}")

# ===============================
# 1️⃣1️⃣ Escalonamento numérico
# ===============================
scaler = StandardScaler()
num_cols = df.select_dtypes(include=[np.number]).columns.drop("fraude")
df[num_cols] = scaler.fit_transform(df[num_cols])

# ===============================
# 1️⃣2️⃣ Salvar dataset final
# ===============================
os.makedirs(os.path.dirname(output_path_parquet), exist_ok=True)
os.makedirs(os.path.dirname(output_metrics_path), exist_ok=True)

df.to_parquet(output_path_parquet, index=False, compression="snappy")
pd.DataFrame(relatorio).to_parquet(output_metrics_path, index=False)

print(f"✅ Dataset tratado salvo em: {output_path_parquet}")
print(f"✅ Relatório salvo em: {output_metrics_path}")
print(f"📊 Linhas finais: {len(df)}")
