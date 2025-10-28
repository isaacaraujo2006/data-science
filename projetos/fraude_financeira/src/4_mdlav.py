# 5_treino_lightgbm_custo_final.py
import os
import json
import yaml
import joblib
import optuna
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    classification_report, confusion_matrix, brier_score_loss
)

# ===============================
# 1) Configurações e caminhos
# ===============================
config_path = r"D:/github/data-science/projetos/fraude_financeira/config/config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_path       = config["data"]["processed_parquet"]
models_dir      = config["models"]["directory"]
metrics_dir     = config["metrics"]["directory"]
fig_dir         = config["reports"]["figures"]
threshold_path  = config["thresholds"]["optimal_threshold_path"]
params_json     = os.path.join(models_dir, "lightgbm_best_params.json")
features_txt    = os.path.join(models_dir, "features_usadas.txt")
metrics_parquet = os.path.join(metrics_dir, "lightgbm_metrics_custo.parquet")
importances_parquet = os.path.join(metrics_dir, "lightgbm_feature_importances.parquet")
sensibilidade_parquet = os.path.join(metrics_dir, "lightgbm_sensibilidade.parquet")
model_path      = os.path.join(models_dir, "lightgbm_calibrated.joblib")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(os.path.dirname(threshold_path), exist_ok=True)

# ===============================
# 2) Leitura e split
# ===============================
print(f"📂 Lendo dataset processado de: {data_path}")
df = pd.read_parquet(data_path)

y = df["fraude"].astype(int)
X = df.drop(columns=["fraude"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.20, stratify=y_train, random_state=42
)

# ===============================
# 3) Otimização Optuna
# ===============================
def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "n_estimators": trial.suggest_int("n_estimators", 150, 900),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1e-1),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 1e-1),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = lgb.LGBMClassifier(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

print("🚀 Otimizando hiperparâmetros...")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40, show_progress_bar=False)
best_params = study.best_params
print(f"✅ Melhores parâmetros: {best_params}")

# ===============================
# 4) Early stopping
# ===============================
best_params_clean = {k: v for k, v in best_params.items() if k != "n_estimators"}
model_es = lgb.LGBMClassifier(**best_params_clean, n_estimators=best_params["n_estimators"], random_state=42, n_jobs=-1)
model_es.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
)
best_iters = getattr(model_es, "best_iteration_", best_params["n_estimators"])
print(f"🛑 Melhor iteração: {best_iters}")

# ===============================
# 5) Treino final + calibração
# ===============================
model_final = lgb.LGBMClassifier(**best_params_clean, n_estimators=best_iters, random_state=42, n_jobs=-1)
model_final.fit(X_train, y_train)
cal = CalibratedClassifierCV(base_estimator=model_final, method="isotonic", cv=5)
cal.fit(X_train, y_train)

# ===============================
# 6) Funções auxiliares
# ===============================
C_FP, C_FN = 1.0, 10.0

def custo_threshold(y_true, y_proba, t):
    pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return C_FP*fp + C_FN*fn

def ks_statistic(y_true, y_proba):
    from scipy.stats import ks_2samp
    return ks_2samp(y_proba[y_true==0], y_proba[y_true==1]).statistic

def avaliar(model, X, y, threshold, nome):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y, proba)
    pr_auc = average_precision_score(y, proba)
    brier = brier_score_loss(y, proba)
    ks = ks_statistic(y, proba)
    rep = classification_report(y, pred, target_names=["nao_fraude", "fraude"], output_dict=True, zero_division=0)
    cm = confusion_matrix(y, pred)
    cost = custo_threshold(y, proba, threshold)

    print(f"\n📊 {nome} — AUC={auc:.4f} | PR-AUC={pr_auc:.4f} | Brier={brier:.4f} | KS={ks:.4f}")
    print(classification_report(y, pred, target_names=["nao_fraude", "fraude"], zero_division=0))
    print(f"Matriz de Confusão ({nome}):\n{cm}")
    print(f"💰 Custo ({nome}): {cost:.2f}")

    return {
        "dataset": nome,
        "threshold": threshold,
        "AUC": auc, "PR_AUC": pr_auc, "Brier": brier, "KS": ks,
        "Precision": rep["fraude"]["precision"], "Recall": rep["fraude"]["recall"], "F1": rep["fraude"]["f1-score"],
        "TN": cm[0,0], "FP": cm[0,1], "FN": cm[1,0], "TP": cm[1,1],
        "Cost": cost, "C_FP": C_FP, "C_FN": C_FN
    }

# ===============================
# 7) Threshold ótimo por custo e F1
# ===============================
proba_tr = cal.predict_proba(X_train)[:, 1]
thresholds = np.linspace(0.01, 0.99, 197)

best_t_custo = min(thresholds, key=lambda t: custo_threshold(y_train, proba_tr, t))
best_t_f1 = max(thresholds, key=lambda t: f1_score(y_train, (proba_tr >= t).astype(int)))

print(f"💰 Melhor threshold custo: {best_t_custo:.4f} | custo={custo_threshold(y_train, proba_tr, best_t_custo):.2f}")
print(f"📈 Melhor threshold F1: {best_t_f1:.4f} | F1={f1_score(y_train, (proba_tr >= best_t_f1).astype(int)):.4f}")

# ===============================
# 8) Avaliação final
# ===============================
resultados = []
resultados.append(avaliar(cal, X_train, y_train, best_t_custo, "Treino (Custo)"))
resultados.append(avaliar(cal, X_test,  y_test,  best_t_custo, "Teste (Custo)"))
resultados.append(avaliar(cal, X_train, y_train, best_t_f1,    "Treino (F1)"))
resultados.append(avaliar(cal, X_test,  y_test,  best_t_f1,    "Teste (F1)"))

# ===============================
# 9) Gráficos
# ===============================
plt.figure()
plt.plot(thresholds, [custo_threshold(y_train, proba_tr, t) for t in thresholds])
plt.axvline(best_t_custo, color="r", linestyle="--", label="Melhor Custo")
plt.axvline(best_t_f1, color="g", linestyle="--", label="Melhor F1")
plt.legend(); plt.xlabel("Threshold"); plt.ylabel("Custo"); plt.title("Custo vs Threshold")
plt.savefig(os.path.join(fig_dir, "custo_vs_threshold.png"))

# ===============================
# 10) Salvar artefatos
# ===============================
joblib.dump(cal, model_path)
with open(threshold_path, "w") as f:
    f.write(json.dumps({"custo": best_t_custo, "f1": best_t_f1}))
pd.DataFrame(resultados).to_parquet(metrics_parquet, index=False)

importances = pd.DataFrame({
    "feature": X.columns,
    "importance": model_final.booster_.feature_importance(importance_type="gain")
}).sort_values(by="importance", ascending=False)
importances.to_parquet(importances_parquet, index=False)

with open(params_json, "w", encoding="utf-8") as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)
with open(features_txt, "w", encoding="utf-8") as f:
    f.write("\n".join(X.columns))

print("\n✅ Pipeline concluído com melhorias e dupla avaliação.")
