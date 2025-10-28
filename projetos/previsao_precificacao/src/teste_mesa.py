# ============================================================
#  teste_mesa.py — Validação do modelo LightGBM com métricas reais
# ============================================================
import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# === Caminhos ===
BASE_DIR = r"D:\github\github\data-science\projetos\previsao_precificacao"
MODEL_PATH = os.path.join(BASE_DIR, "config", "models", "refinamento_lightgbm", "lightgbm_refinado.pkl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "predicoes", "teste_mesa_predicoes.parquet")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# === Carregar modelo ===
model = joblib.load(MODEL_PATH)

# === Geração de 15 exemplos artificiais coerentes ===
np.random.seed(42)
base_date = datetime(2025, 1, 1)

data = {
    "dt": [base_date + timedelta(days=i) for i in range(15)],
    "store_id": np.random.randint(0, 10, 15),
    "first_category_id": np.random.randint(0, 5, 15),
    "discount": np.round(np.random.uniform(0.0, 0.4, 15), 2),
    "activity_flag": np.random.randint(0, 2, 15),
    "avg_temperature": np.round(np.random.uniform(15, 35, 15), 1),
    "avg_humidity": np.round(np.random.uniform(30, 90, 15), 1),
    "precpt": np.round(np.random.uniform(0, 20, 15), 1),
    # Valores reais simulados (ground truth)
    "sale_amount": np.round(np.random.uniform(2.0, 5.0, 15), 3)
}

df = pd.DataFrame(data)

# === Engenharia de features ===
df["year"] = df["dt"].dt.year
df["month"] = df["dt"].dt.month
df["weekday"] = df["dt"].dt.dayofweek
df["day"] = df["dt"].dt.day
df["temp_x_desc"] = df["avg_temperature"] * df["discount"]
df["humid_x_act"] = df["avg_humidity"] * df["activity_flag"]

for lag in [1, 7, 14, 30]:
    df[f"lag_{lag}"] = df["sale_amount"].shift(lag, fill_value=df["sale_amount"].mean())
for window in [7, 14, 30]:
    df[f"rolling_mean_{window}"] = (
        df["sale_amount"].shift(1, fill_value=df["sale_amount"].mean()).rolling(window, min_periods=1).mean()
    )

for col in ["store_id", "first_category_id"]:
    df[col] = df[col].astype("category").cat.codes

# === Predição ===
X_test = df.drop(columns=["sale_amount", "dt"], errors="ignore")
preds = model.predict(X_test, num_iteration=model.best_iteration)
df["predicted_price"] = preds

# === Avaliação quantitativa ===
y_true = df["sale_amount"]
y_pred = df["predicted_price"]

rmse = mean_squared_error(y_true, y_pred, squared=False)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1))) * 100

metrics = {
    "RMSE": round(rmse, 4),
    "MAE": round(mae, 4),
    "R2": round(r2, 4),
    "MAPE": round(mape, 2)
}

# === Salvar resultados ===
df.to_parquet(OUTPUT_PATH, index=False)
df.to_csv(OUTPUT_PATH.replace(".parquet", ".csv"), index=False)

# === Exibir resultados ===
print("\n📈 RESULTADOS DO TESTE DE MESA:")
print(df[["dt", "store_id", "first_category_id", "sale_amount", "predicted_price"]].head(15))
print("\n📊 MÉTRICAS DE AVALIAÇÃO:")
for k, v in metrics.items():
    print(f"{k}: {v}")

print(f"\n✅ Resultados salvos em:\n- {OUTPUT_PATH}\n- {OUTPUT_PATH.replace('.parquet', '.csv')}")
