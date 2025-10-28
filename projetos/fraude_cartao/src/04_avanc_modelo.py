# 04_avanc_modelo_histgb.py

import os
import joblib
import yaml
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from tqdm import tqdm
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.experimental import enable_hist_gradient_boosting  # necessário para habilitar HistGB
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split, learning_curve, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, f1_score, average_precision_score, confusion_matrix,
    recall_score, precision_score, accuracy_score
)

# === Registro de tempo ===
start_time = time.time()

# === Carregar configurações do projeto ===
with open("D:/github/data-science/projetos/fraude_cartao/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# === Configurar logging ===
log_dir = config['paths']['logs_dir']
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'modelagem_histgb.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# === Carregar dados ===
data_path = config['data']['processed_csv']
target_column = "class"
df = pd.read_csv(data_path)
X = df.drop(columns=[target_column])
y = df[target_column]

# === Dividir dados ===
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)

# === Pipeline com SMOTE e HistGradientBoostingClassifier ===
smote = SMOTE(random_state=42, k_neighbors=3, n_jobs=4)

histgb = HistGradientBoostingClassifier(random_state=42)

pipeline = Pipeline([
    ('smote', smote),
    ('histgb', histgb)
])

# === Espaço de busca de hiperparâmetros ===
param_grid = {
    'histgb__max_iter': [100, 200, 300],
    'histgb__max_depth': [3, 5, 7, None],
    'histgb__learning_rate': [0.01, 0.05, 0.1],
    'histgb__max_leaf_nodes': [15, 31, 63],
    'histgb__min_samples_leaf': [20, 50, 100],
    'histgb__l2_regularization': [0.0, 0.1, 0.5]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=15,
    scoring='f1',
    cv=cv,
    n_jobs=4,
    verbose=2,
    random_state=42
)

# === Treinar com dados de treino ===
search.fit(X_train, y_train)

best_model = search.best_estimator_
logging.info(f"Melhores hiperparâmetros: {search.best_params_}")

# === Calibrar o modelo com validação ===
calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_val, y_val)

# === Encontrar o threshold ótimo baseado no conjunto de validação ===
y_probs_val = calibrated_model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0, 1, 101)

best_metrics = {'recall': {'threshold': 0, 'value': 0},
                'precision': {'threshold': 0, 'value': 0},
                'f1': {'threshold': 0, 'value': 0},
                'accuracy': {'threshold': 0, 'value': 0}}

for t in tqdm(thresholds, desc="Avaliando thresholds"):
    preds = (y_probs_val >= t).astype(int)
    recall = recall_score(y_val, preds)
    precision = precision_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds)
    accuracy = accuracy_score(y_val, preds)
    if recall > best_metrics['recall']['value']:
        best_metrics['recall'] = {'threshold': t, 'value': recall}
    if precision > best_metrics['precision']['value']:
        best_metrics['precision'] = {'threshold': t, 'value': precision}
    if f1 > best_metrics['f1']['value']:
        best_metrics['f1'] = {'threshold': t, 'value': f1}
    if accuracy > best_metrics['accuracy']['value']:
        best_metrics['accuracy'] = {'threshold': t, 'value': accuracy}

chosen_threshold = best_metrics['f1']['threshold']

# === Avaliação final no conjunto de teste ===
y_probs_test = calibrated_model.predict_proba(X_test)[:, 1]
y_pred_test = (y_probs_test >= chosen_threshold).astype(int)

report = classification_report(y_test, y_pred_test, output_dict=True, digits=4)
df_report = pd.DataFrame(report).transpose()
auc_pr = average_precision_score(y_test, y_probs_test)
cm = confusion_matrix(y_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()
cost = fp * 1 + fn * 10  # Exemplo de custo customizado

logging.info(f"Relatório de Classificação:\n{df_report}")
logging.info(f"AUC-PR: {auc_pr:.4f}")
logging.info(f"Matriz de Confusão:\n{cm}")
logging.info(f"Custo total: {cost}")

# === Cross-validation para estabilidade do modelo ===
scores = cross_val_score(best_model.named_steps['histgb'], X_train, y_train, cv=cv, scoring='f1')
logging.info(f"F1 Score médio (cross_val): {scores.mean():.4f}")

# === Salvar modelo calibrado e threshold ===
model_output = {
    'model': calibrated_model,
    'threshold': chosen_threshold,
    'metrics': best_metrics
}

os.makedirs(config['models']['directory'], exist_ok=True)
joblib.dump(model_output, os.path.join(config['models']['directory'], "histgb_model_calibrated.pkl"))
joblib.dump(smote, os.path.join(os.path.dirname(config['preprocessors']['preprocessor_path']), "smote.pkl"))

# === Curva de aprendizado ===
train_sizes, train_scores, val_scores = learning_curve(
    best_model.named_steps['histgb'], X_train, y_train,
    cv=cv, scoring='f1', n_jobs=4, train_sizes=np.linspace(0.2, 1.0, 5)
)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Treino')
plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validação')
plt.title("Curva de Aprendizado - F1 Score")
plt.xlabel("Tamanho do Treino")
plt.ylabel("F1 Score")
plt.legend()
plt.grid()
plt.tight_layout()
fig_dir = config['paths']['figures_dir']
os.makedirs(fig_dir, exist_ok=True)
plt.savefig(os.path.join(fig_dir, "curva_aprendizado_histgb.png"))
plt.close()

# === SHAP (explicabilidade) ===
explainer = shap.Explainer(best_model.named_steps['histgb'])
X_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
shap_values = explainer(X_sample)
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "shap_summary_plot_histgb.png"))
plt.close()

# === Tempo total ===
end_time = time.time()
h, m, s = int((end_time - start_time) // 3600), int(((end_time - start_time) % 3600) // 60), int((end_time - start_time) % 60)
logging.info(f"Tempo total: {h}h {m}min {s}s")
print("✅ Etapa de modelagem com HistGradientBoosting concluída com sucesso.")
