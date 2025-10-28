import os
import joblib
import numpy as np
import pandas as pd

# === Caminhos do projeto ===
DATA_PATH = "D:/github/data-science/projetos/fraude_cartao/data/processed/processed.csv"
MODEL_PATH = "D:/github/data-science/projetos/fraude_cartao/models/histgb_model_calibrated.pkl"

# === Carregar modelo e threshold ===
model_output = joblib.load(MODEL_PATH)
model = model_output['model']
threshold = model_output['threshold']

# === Confirmar índice da classe 1 ===
fraude_index = list(model.classes_).index(1)

# === Carregar dados ===
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["class"])
y = df["class"]

# === Obter probabilidades da classe 1 ===
probs_all = model.predict_proba(X)[:, fraude_index]

# === Zona cinzenta (certeza média) ===
zona_cinza_idx = (probs_all >= 0.20) & (probs_all <= 0.80)
df_cinza = df.loc[zona_cinza_idx].copy()
df_cinza['prob_fraude'] = probs_all[zona_cinza_idx]

# === Separar zona cinzenta por classe ===
df_cinza_fraude = df_cinza[df_cinza['class'] == 1]
df_cinza_nao_fraude = df_cinza[df_cinza['class'] == 0]

# === Definir quantidades para amostrar (metade de cada, ou máximo possível se tiver pouco) ===
n_total = 15
n_fraude = min(len(df_cinza_fraude), n_total // 2)
n_nao_fraude = min(len(df_cinza_nao_fraude), n_total - n_fraude)

if n_fraude + n_nao_fraude < n_total:
    raise ValueError(f"❌ Não há exemplos suficientes dentro da zona cinzenta para balancear a amostra. Fraude disponíveis: {len(df_cinza_fraude)}, Não fraude disponíveis: {len(df_cinza_nao_fraude)}")

# === Amostrar aleatoriamente ===
amostra_fraude = df_cinza_fraude.sample(n=n_fraude, random_state=42)
amostra_nao_fraude = df_cinza_nao_fraude.sample(n=n_nao_fraude, random_state=42)

testes = pd.concat([amostra_fraude, amostra_nao_fraude]).sample(frac=1, random_state=42)  # embaralhar
X_testes = testes.drop(columns=["class", "prob_fraude"])
y_testes = testes["class"]

# === Fazer predições com threshold ajustado ===
probs = model.predict_proba(X_testes)[:, fraude_index]
preds = (probs >= threshold).astype(int)

# === Teste de Mesa: Avaliação ===
print("\n=== 🧪 Teste de Mesa Balanceado - Dentro da Zona Cinzenta (15 Casos) ===\n")

acertos = 0
erros = 0

for i, (idx, prob, pred, real) in enumerate(zip(testes.index, probs, preds, y_testes), 1):
    label_real = "FRAUDE" if real == 1 else "NÃO FRAUDE"
    label_pred = "FRAUDE" if pred == 1 else "NÃO FRAUDE"
    cor = "🔴" if label_pred == "FRAUDE" else "🟢"

    print(f"{cor} Caso {i}")
    print(f"🧠 Probabilidade de fraude: {prob:.4f}")
    print(f"🚩 Predição final: {label_pred}")
    print(f"🎯 Classe real: {label_real}")
    print("📋 Atributos da transação:")
    print(testes.loc[[idx]].drop(columns=["class", "prob_fraude"]))
    print("-" * 90)

    if pred == real:
        acertos += 1
    else:
        erros += 1

# === Resumo final ===
print(f"\n✅ Acertos: {acertos}")
print(f"❌ Erros: {erros}")
print(f"📊 Acurácia dentro da zona cinzenta: {acertos / 15:.4f}")
