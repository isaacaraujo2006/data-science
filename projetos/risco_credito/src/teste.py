import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# --- Caminhos ---
models_dir = "D:/github/data-science/projetos/risco_credito/models"
modelo_path = os.path.join(models_dir, "catboost_optimized_calibrated.joblib")
threshold_path = os.path.join(models_dir, "catboost_optimized_threshold.txt")

# --- Carregar modelo calibrado e threshold ---
modelo = joblib.load(modelo_path)
with open(threshold_path, 'r') as f:
    threshold = float(f.read().strip())

print(f"Threshold carregado: {threshold:.2f}")

# --- Carregar base processada ---
df_full = pd.read_parquet("D:/github/data-science/projetos/risco_credito/data/processed/processed.parquet")

# --- Separar o conjunto de teste igual ao da modelagem ---
target_column = 'inadimplente_mes_seguinte'
X = df_full.drop(columns=[target_column])
y = df_full[target_column]
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
df_real_test = X_test.copy()
df_real_test['inadimplente_mes_seguinte'] = y_test.values

colunas_esperadas = [
    'id', 'limite_credito', 'sexo', 'educacao', 'estado_civil', 'idade',
    'pagamento_mes_0', 'pagamento_mes_2', 'pagamento_mes_3', 'pagamento_mes_4',
    'pagamento_mes_5', 'pagamento_mes_6', 'fatura_mes_1', 'fatura_mes_2',
    'fatura_mes_3', 'fatura_mes_4', 'fatura_mes_5', 'fatura_mes_6', 'pagamento_mes_1'
]
df_real_test = df_real_test[colunas_esperadas + ['inadimplente_mes_seguinte']]

# --- Prever probabilidades ---
probs = modelo.predict_proba(df_real_test[colunas_esperadas])[:, 1]

# --- Filtrar exemplos NA ZONA CINZENTA (próximos do threshold) ---
zona_cinzenta = (probs >= 0.20) & (probs <= 0.30)
df_zona_cinzenta = df_real_test.loc[zona_cinzenta].copy()
df_zona_cinzenta['probabilidade'] = probs[zona_cinzenta]
df_zona_cinzenta['predicao'] = (df_zona_cinzenta['probabilidade'] >= threshold).astype(int)
df_zona_cinzenta['true_label'] = df_zona_cinzenta['inadimplente_mes_seguinte']
df_zona_cinzenta.drop(columns=['inadimplente_mes_seguinte'], inplace=True)

# --- Selecionar 15 exemplos mistos na zona cinzenta ---
if len(df_zona_cinzenta) < 15:
    print(f"Atenção: só {len(df_zona_cinzenta)} exemplos na zona cinzenta (ajuste o intervalo se quiser mais casos).")
    n_exemplos = len(df_zona_cinzenta)
else:
    n_exemplos = 15

df_teste_mesa_cinza = df_zona_cinzenta.sample(n_exemplos, random_state=42).reset_index(drop=True)
df_teste_mesa_cinza['acertou'] = df_teste_mesa_cinza['predicao'] == df_teste_mesa_cinza['true_label']

print("\n=== Teste de Mesa (Zona Cinzenta: 0.20 <= prob <= 0.30) ===")
print(df_teste_mesa_cinza)

acertos = df_teste_mesa_cinza['acertou'].sum()
print(f"\nNúmero de acertos: {acertos} de {n_exemplos} exemplos")
print(f"Acurácia na zona cinzenta: {acertos / n_exemplos:.4f}")
