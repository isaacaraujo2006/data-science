# 5_teste.py - Teste de Mesa com thresholds de custo e F1

import os
import json
import random
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

# ============================
# Configurações
# ============================
config_path = r"D:/github/data-science/projetos/fraude_financeira/config/config.yaml"
import yaml
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

model_path     = config["models"]["directory"] + "/lightgbm_calibrated.joblib"
threshold_path = config["thresholds"]["optimal_threshold_path"]
data_path      = config["data"]["processed_parquet"]

# ============================
# Carrega modelo e thresholds
# ============================
print("📂 Carregando modelo...")
model = joblib.load(model_path)

print("📂 Lendo thresholds...")
with open(threshold_path, "r", encoding="utf-8") as f:
    th_dict = json.load(f)

threshold_custo = th_dict["custo"]
threshold_f1    = th_dict["f1"]

print(f"🔹 Threshold custo: {threshold_custo}")
print(f"🔹 Threshold F1: {threshold_f1}")

# ============================
# Carrega dados
# ============================
df = pd.read_parquet(data_path)
y = df["fraude"].astype(int)
X = df.drop(columns=["fraude"])

# ============================
# Seleção de 15 exemplos
# ============================
indices = random.sample(range(len(X)), 15)
X_sample = X.iloc[indices]
y_sample = y.iloc[indices]

# ============================
# Previsões
# ============================
probas = model.predict_proba(X_sample)[:, 1]
pred_custo = (probas >= threshold_custo).astype(int)
pred_f1    = (probas >= threshold_f1).astype(int)

# ============================
# Resultado individual
# ============================
print("\n📊 Resultados individuais:")
print("Idx\tReal\tPred_Custo\tPred_F1\tProba")
for idx, real, pc, pf, p in zip(indices, y_sample, pred_custo, pred_f1, probas):
    print(f"{idx}\t{real}\t{pc}\t\t{pf}\t\t{p:.4f}")

# ============================
# Resumo
# ============================
acc_custo = accuracy_score(y_sample, pred_custo)
acc_f1    = accuracy_score(y_sample, pred_f1)

print("\n📌 Resumo Final:")
print(f"✅ Acertos (Custo): {acc_custo*100:.2f}% ({acc_custo*15:.0f}/15)")
print(f"✅ Acertos (F1):    {acc_f1*100:.2f}% ({acc_f1*15:.0f}/15)")
