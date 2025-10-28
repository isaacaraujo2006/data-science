import pandas as pd
import numpy as np
import os
import logging

# Configurar logging
logging.basicConfig(
    filename="preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Iniciando o pré-processamento dos dados.")

# Carregar o dataset
file_path = r"C:\Github\data-science\projetos\segmentacao_clientes\data\raw\customer_segmentation.csv"
df = pd.read_csv(file_path)

# Número de linhas importadas inicialmente
total_linhas = df.shape[0]
logging.info(f"Número total de linhas no início: {total_linhas}")

# Validar colunas esperadas
colunas_traduzidas = {
    "ID": "id",
    "Year_Birth": "ano_nascimento",
    "Education": "educacao",
    "Marital_Status": "estado_civil",
    "Income": "renda",
    "Kidhome": "criancas_em_casa",
    "Teenhome": "adolescentes_em_casa",
    "Dt_Customer": "data_cliente",
    "Recency": "recencia",
    "MntWines": "gastos_vinhos",
    "MntFruits": "gastos_frutas",
    "MntMeatProducts": "gastos_carnes",
    "MntFishProducts": "gastos_peixes",
    "MntSweetProducts": "gastos_doces",
    "MntGoldProds": "gastos_joias",
    "NumDealsPurchases": "compras_promocao",
    "NumWebPurchases": "compras_web",
    "NumCatalogPurchases": "compras_catalogo",
    "NumStorePurchases": "compras_loja",
    "NumWebVisitsMonth": "visitas_web_mes",
    "AcceptedCmp3": "aceitou_campanha3",
    "AcceptedCmp4": "aceitou_campanha4",
    "AcceptedCmp5": "aceitou_campanha5",
    "AcceptedCmp1": "aceitou_campanha1",
    "AcceptedCmp2": "aceitou_campanha2",
    "Complain": "reclamacao",
    "Z_CostContact": "custo_contato",
    "Z_Revenue": "receita",
    "Response": "resposta"
}

colunas_esperadas = set(colunas_traduzidas.keys())
colunas_faltantes = colunas_esperadas - set(df.columns)

if colunas_faltantes:
    raise ValueError(f"As seguintes colunas estão ausentes no dataset: {colunas_faltantes}")
else:
    df.rename(columns=colunas_traduzidas, inplace=True)

# Traduzir valores das colunas categóricas
mapa_escolaridade = {
    "Graduation": "Graduação",
    "Master": "Mestrado",
    "Basic": "Básico",
    "PhD": "Doutorado"
}
mapa_estado_civil = {
    "Married": "Casado(a)",
    "Together": "Juntos",
    "Single": "Solteiro(a)",
    "Divorced": "Divorciado(a)"
}

df["educacao"] = df["educacao"].replace(mapa_escolaridade)
df["estado_civil"] = df["estado_civil"].replace(mapa_estado_civil)

# Registrar categorias desconhecidas
categorias_esperadas = {
    "educacao": set(mapa_escolaridade.keys()),
    "estado_civil": set(mapa_estado_civil.keys())
}
for coluna, categorias in categorias_esperadas.items():
    categorias_desconhecidas = set(df[coluna]) - categorias
    if categorias_desconhecidas:
        logging.warning(f"Categorias desconhecidas em '{coluna}': {categorias_desconhecidas}")

# Contar dados duplicados
duplicatas = df.duplicated().sum()
percent_duplicatas = (duplicatas / total_linhas) * 100

# Contar valores faltantes
faltantes = df.isnull().sum()
print("Colunas com valores faltantes antes do preenchimento:")
print(faltantes)

# Preencher valores faltantes
for col in df.columns:
    if df[col].isnull().sum() > 0:
        # Se for numérica, preencher com mediana
        if df[col].dtype in ["float64", "int64"]:
            df[col].fillna(df[col].median(), inplace=True)
        # Se for categórica, preencher com o modo
        elif df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)

# Validar consistência após o preenchimento
assert not df.isnull().any().any(), "Ainda existem valores faltantes após preenchimento."

# Definir intervalos esperados para algumas colunas
intervalos = {
    "ano_nascimento": (1900, 2025),
    "renda": (0, 100000),
    "criancas_em_casa": (0, 2),
    "adolescentes_em_casa": (0, 2),
    "recencia": (0, 1000)
}
fora_intervalo = {col: ((df[col] < intervalo[0]) | (df[col] > intervalo[1])).sum() for col, intervalo in intervalos.items()}
percent_fora_intervalo = {col: (count / total_linhas) * 100 for col, count in fora_intervalo.items()}

# Identificar outliers utilizando o método alternativo (percentil 1% e 99%)
def tratar_outliers(df, colunas):
    for col in colunas:
        p1 = df[col].quantile(0.01)
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(lower=p1, upper=p99)
    return df

df = tratar_outliers(df, df.select_dtypes(include=["float64", "int64"]).columns)

# Relatório pós-tratamento
linhas_restantes = df.shape[0]
percent_removido = ((total_linhas - linhas_restantes) / total_linhas) * 100

print("\nRelatório pós-tratamento:")
print(f"Linhas restantes após tratamento: {linhas_restantes}")
print(f"Percentual de linhas removidas: {percent_removido:.2f}%")

# Validar consistência final
assert not df.isnull().any().any(), "Ainda existem valores faltantes."
assert not df.duplicated().any(), "Ainda existem duplicatas."

# Salvar dataset processado
output_dir = r"C:\Github\data-science\projetos\segmentacao_clientes\data\processed"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "processed.csv"), index=False)
df.to_parquet(os.path.join(output_dir, "processed.parquet"), index=False)
logging.info("Dataset processado salvo com sucesso nos formatos CSV e Parquet.")

print("\nProcessamento concluído e datasets salvos.")
