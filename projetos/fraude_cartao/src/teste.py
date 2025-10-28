import joblib
import pandas as pd

# Caminhos - ajuste se necessário
MODEL_PATH = "D:/github/data-science/projetos/fraude_cartao/models/xgb_final_model.joblib"
THRESHOLD_PATH = "D:/github/data-science/projetos/fraude_cartao/models/xgb_optimal_threshold.txt"

# 1. Carregar modelo e threshold
best_model = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    threshold = float(f.read().strip())

# 2. Preparar dados de teste (com as 32 features que o modelo espera)
data = {
    "time": [0],
    "v1": [-18],
    "v2": [9.4],
    "v3": [-14.2],
    "v4": [12.8],
    "v5": [-15],
    "v6": [0],
    "v7": [0],
    "v8": [0],
    "v9": [0],
    "v10": [0],
    "v11": [0],
    "v12": [0],
    "v13": [0],
    "v14": [0],
    "v15": [0],
    "v16": [0],
    "v17": [0],
    "v18": [0],
    "v19": [0],
    "v20": [0],
    "v21": [0],
    "v22": [0],
    "v23": [0],
    "v24": [0],
    "v25": [0],
    "v26": [0],
    "v27": [0],
    "v28": [0],
    "amount": [10000],
    "outlier_amount": [1],
    "hour_of_day": [2]
}

df_input = pd.DataFrame(data)

# 3. Fazer predição
probs = best_model.predict_proba(df_input)[:, 1]
print(f"Probabilidade de fraude: {probs[0]:.4f}")

prediction = "FRAUDE" if probs[0] >= threshold else "NÃO FRAUDE"
print(f"Predição final: {prediction}")
