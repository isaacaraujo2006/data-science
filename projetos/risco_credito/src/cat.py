import os
import joblib
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    classification_report, confusion_matrix
)
from scipy.stats import uniform, randint
from collections import Counter

# --- Configuração ---
with open("D:/github/data-science/projetos/risco_credito/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

df = pd.read_parquet(config['data']['processed_parquet'])
target_column = "inadimplente_mes_seguinte"
X = df.drop(columns=[target_column])
y = df[target_column]

# --- Divisão treino/val/test ---
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
)

print("Distribuição original das classes (treino):", Counter(y_train))

# --- Pipeline com SMOTETomek e CatBoost ---
pipe_cat = Pipeline([
    ('smote', SMOTETomek(random_state=42)),
    ('catboost', CatBoostClassifier(
        random_state=42,
        verbose=0,
        thread_count=4,
        early_stopping_rounds=50    # <-- Já setado aqui!
    ))
])

param_dist = {
    'catboost__depth': randint(4, 9),
    'catboost__learning_rate': uniform(0.01, 0.09),
    'catboost__l2_leaf_reg': randint(1, 10),
    'catboost__iterations': [150, 200, 300],
}

cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipe_cat,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',
    cv=cv,
    n_jobs=4,
    verbose=2,
    random_state=42
)

search.fit(X_train, y_train)
print(f"\nMelhores parâmetros: {search.best_params_}")

# --- NÃO precisa setar params depois! Só fit normalmente: ---
best_model = search.best_estimator_
best_model.fit(X_trainval, y_trainval)

# --- Calibração isotônica ---
calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv='prefit')
calibrated.fit(X_val, y_val)

# --- Distribuição das probabilidades ---
y_probs_val = calibrated.predict_proba(X_val)[:, 1]
plt.figure(figsize=(8,4))
plt.hist(y_probs_val, bins=40, color='royalblue', alpha=0.7)
plt.title('Distribuição das probabilidades (Validação)')
plt.xlabel('Probabilidade prevista (classe 1)')
plt.ylabel('Quantidade')
plt.tight_layout()
plt.savefig(os.path.join(config['paths']['figures_dir'], 'probs_val_hist.png'))
plt.close()

# --- Otimização de threshold ---
thresholds = np.linspace(0, 1, 101)
metrics = [{
    'threshold': t,
    'f1': f1_score(y_val, y_probs_val >= t),
    'precision': precision_score(y_val, y_probs_val >= t),
    'recall': recall_score(y_val, y_probs_val >= t),
    'accuracy': accuracy_score(y_val, y_probs_val >= t)
} for t in thresholds]
df_metrics = pd.DataFrame(metrics)
idx_f1 = df_metrics['f1'].idxmax()
threshold_f1 = df_metrics.loc[idx_f1, 'threshold']

print(f"\nThreshold ótimo (F1-score): {threshold_f1:.2f} | F1: {df_metrics.loc[idx_f1, 'f1']:.4f}")
best_threshold = threshold_f1
print(f"\n[USANDO] Threshold ótimo (F1-score): {best_threshold:.2f}")

# --- Curva F1/Accuracy vs. Threshold ---
plt.figure(figsize=(8,4))
plt.plot(df_metrics['threshold'], df_metrics['f1'], label='F1-score')
plt.plot(df_metrics['threshold'], df_metrics['accuracy'], label='Accuracy')
plt.xlabel('Threshold')
plt.ylabel('Métrica')
plt.legend()
plt.title('F1-score e Accuracy vs. Threshold')
plt.tight_layout()
plt.savefig(os.path.join(config['paths']['figures_dir'], 'thresholds_f1_acc.png'))
plt.close()

# --- Avaliação final no conjunto de teste ---
y_probs_test = calibrated.predict_proba(X_test)[:, 1]
y_pred_test = (y_probs_test >= best_threshold).astype(int)
report = classification_report(y_test, y_pred_test, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# --- Matriz de confusão e custo ---
cm = confusion_matrix(y_test, y_pred_test)
print("\nMatriz de Confusão (teste):")
print(cm)
plt.figure(figsize=(4,4))
plt.imshow(cm, cmap='Blues', alpha=0.7)
plt.title("Matriz de Confusão (Teste)")
plt.ylabel("Verdadeiro")
plt.xlabel("Previsto")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(config['paths']['figures_dir'], 'matriz_confusao_teste.png'))
plt.close()

tn, fp, fn, tp = cm.ravel()
custo_total = fp*1 + fn*10
print(f"\nCusto total de erros: {custo_total} (FP={fp}, FN={fn})")

# --- Feature Importance ---
feature_imp = best_model.named_steps['catboost'].get_feature_importance()
features = X.columns
df_feat_imp = pd.DataFrame({'feature': features, 'importance': feature_imp}).sort_values(by='importance', ascending=False)
df_feat_imp.to_csv(os.path.join(config['paths']['reports_dir'], 'feature_importance_catboost.csv'), index=False)

# --- Salvar artefatos ---
model_path = os.path.join(config['models']['directory'], "catboost_optimized_calibrated.joblib")
joblib.dump(calibrated, model_path)
with open(os.path.join(config['models']['directory'], "catboost_optimized_threshold.txt"), 'w') as f:
    f.write(str(best_threshold))
df_report.to_csv(os.path.join(config['paths']['reports_dir'], 'report_catboost_optimized.csv'))

print(f"\nF1-score final (classe 1): {df_report.loc['1', 'f1-score']:.4f}")
print(f"Acurácia final: {df_report.loc['accuracy', 'precision']:.4f}")
print(f"Relatórios, gráficos e modelo salvos.")

print("\n✅ Dica: cheque matriz de confusão, custo, feature importance e repita seleção para extrair ainda mais do seu dado!")
