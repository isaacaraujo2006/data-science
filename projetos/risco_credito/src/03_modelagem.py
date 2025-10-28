# 04_modeloavanc_competicao.py

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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    train_test_split,
    learning_curve,
    cross_val_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report, f1_score, average_precision_score, confusion_matrix,
    recall_score, precision_score, accuracy_score
)

# === Registro de tempo ===
start_time = time.time()

# === Carregar configurações do projeto ===
with open("D:/github/data-science/projetos/risco_credito/config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# === Configurar logging ===
log_dir = config['paths']['logs_dir']
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'modelagem_competicao.log')
logging.basicConfig(filename=log_path, level=logging.INFO, format='%(asctime)s - %(message)s')

# === Carregar dados ===
data_path = config['data']['processed_parquet']
target_column = "inadimplente_mes_seguinte"
df = pd.read_parquet(data_path)
X = df.drop(columns=[target_column])
y = df[target_column]

logging.info(f"Colunas do DataFrame: {list(df.columns)}")

num_neg = sum(y == 0)
num_pos = sum(y == 1)
scale_pos_weight = num_neg / num_pos
logging.info(f"scale_pos_weight calculado: {scale_pos_weight:.3f}")

# === Dividir dados ===
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42)

# Criar SMOTE uma vez só para reuso
smote = SMOTE(random_state=42, k_neighbors=5, n_jobs=4)

# === Função para treinar, calibrar e avaliar cada modelo ===
def treinar_e_avaliar(model_name, pipeline, param_grid, early_stopping_params=None):

    logging.info(f"\n\n===== Treinando {model_name} =====")
    print(f"\n===== Treinando {model_name} =====")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=25,
        scoring='f1',
        cv=cv,
        n_jobs=4,
        verbose=2,
        random_state=42
    )

    # Treinar RandomizedSearchCV sem early stopping (não é aceito diretamente em pipeline)
    search.fit(X_train, y_train)
    best_params = search.best_params_
    logging.info(f"{model_name} - Melhores hiperparâmetros: {best_params}")
    print(f"{model_name} - Melhores hiperparâmetros: {best_params}")

    # Extrair parâmetros para o estimador (removendo prefixos do pipeline)
    prefix = list(pipeline.named_steps.keys())[1] + '__'  # normalmente 'lgbm__' ou 'xgb__' etc
    est_params = {k.replace(prefix, ''): v for k, v in best_params.items()}
    est_params['n_estimators'] = 1000  # para early stopping

    # Instanciar o modelo final com os melhores parâmetros
    if model_name == 'LightGBM':
        model_final = LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, n_jobs=4, **est_params)
    elif model_name == 'XGBoost':
        model_final = XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, n_jobs=4, use_label_encoder=False, eval_metric='logloss', **est_params)
    elif model_name == 'CatBoost':
        # CatBoost tratado à parte (não usado aqui)
        raise ValueError("Para CatBoost use a função dedicada treinar_catboost()")
    elif model_name == 'RandomForest':
        model_final = RandomForestClassifier(random_state=42, n_jobs=4, **est_params)
    else:
        raise ValueError("Modelo não suportado")

    pipeline_final = Pipeline([
        ('smote', smote),
        (list(pipeline.named_steps.keys())[1], model_final)
    ])

    # Treinamento final com early stopping se possível
    if model_name in ['LightGBM', 'XGBoost']:
        pipeline_final.fit(
            X_trainval, y_trainval,
            **early_stopping_params
        )
    else:
        pipeline_final.fit(X_trainval, y_trainval)

    # Calibrar
    calibrated = CalibratedClassifierCV(pipeline_final, method='sigmoid', cv='prefit')
    calibrated.fit(X_val, y_val)

    # Buscar threshold ótimo no conjunto validação
    y_probs_val = calibrated.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    best_metrics = {'recall': {'threshold': 0, 'value': 0},
                    'precision': {'threshold': 0, 'value': 0},
                    'f1': {'threshold': 0, 'value': 0},
                    'accuracy': {'threshold': 0, 'value': 0}}

    for t in thresholds:
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
    logging.info(f"{model_name} - Threshold escolhido (F1): {chosen_threshold}")
    print(f"{model_name} - Threshold escolhido (F1): {chosen_threshold}")

    # Avaliação final no conjunto de teste
    y_probs_test = calibrated.predict_proba(X_test)[:, 1]
    y_pred_test = (y_probs_test >= chosen_threshold).astype(int)

    report = classification_report(y_test, y_pred_test, output_dict=True, digits=4)
    df_report = pd.DataFrame(report).transpose()

    auc_pr = average_precision_score(y_test, y_probs_test)
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    cost = fp * 1 + fn * 10

    logging.info(f"{model_name} - Relatório:\n{df_report}")
    logging.info(f"{model_name} - AUC-PR: {auc_pr:.4f}")
    logging.info(f"{model_name} - Matriz de Confusão:\n{cm}")
    logging.info(f"{model_name} - Custo total: {cost}")

    # Salvar relatório CSV
    report_dir = config['paths']['reports_dir']
    os.makedirs(report_dir, exist_ok=True)
    df_report.to_csv(os.path.join(report_dir, f'classification_report_{model_name}.csv'), index=True)

    # Cross-validation f1 no treino (exceto CatBoost)
    if model_name != 'CatBoost':
        scores = cross_val_score(model_final, X_train, y_train, cv=cv, scoring='f1')
        logging.info(f"{model_name} - F1 Score médio (cross_val): {scores.mean():.4f}")

        # Curva de aprendizado
        train_sizes, train_scores, val_scores = learning_curve(
            model_final, X_train, y_train,
            cv=cv, scoring='f1', n_jobs=4, train_sizes=np.linspace(0.2, 1.0, 5)
        )
        plt.figure(figsize=(8, 6))
        plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Treino')
        plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Validação')
        plt.title(f"Curva de Aprendizado - F1 Score - {model_name}")
        plt.xlabel("Tamanho do Treino")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        fig_dir = config['paths']['figures_dir']
        os.makedirs(fig_dir, exist_ok=True)
        plt.savefig(os.path.join(fig_dir, f"curva_aprendizado_{model_name}.png"))
        plt.close()

    # SHAP para LightGBM e XGBoost
    if model_name in ['LightGBM', 'XGBoost']:
        explainer = shap.TreeExplainer(model_final)
        X_sample = X_test.sample(n=min(300, len(X_test)), random_state=42)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
        shap.summary_plot(shap_values_to_plot, X_sample, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"shap_summary_plot_{model_name}.png"))
        plt.close()

    return {
        'model_name': model_name,
        'model': calibrated,
        'threshold': chosen_threshold,
        'f1': best_metrics['f1']['value'],
        'auc_pr': auc_pr,
        'classification_report': df_report,
        'confusion_matrix': cm,
        'cost': cost,
    }

# === Definições dos pipelines e parâmetros por modelo ===

# LightGBM
pipe_lgbm = Pipeline([
    ('smote', smote),
    ('lgbm', LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight, n_jobs=4))
])
params_lgbm = {
    'lgbm__n_estimators': [100, 200, 300, 400],
    'lgbm__max_depth': [3, 5, 7, 10, None],
    'lgbm__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'lgbm__num_leaves': [15, 31, 50, 63],
    'lgbm__min_child_samples': [20, 30, 50, 100],
    'lgbm__reg_alpha': [0.0, 0.1, 0.5, 1.0],
    'lgbm__reg_lambda': [0.0, 0.1, 0.5, 1.0],
    'lgbm__feature_fraction': [0.7, 0.8, 0.9, 1.0],
    'lgbm__bagging_fraction': [0.7, 0.8, 0.9, 1.0],
    'lgbm__bagging_freq': [0, 1, 5],
    'lgbm__verbose': [-1]
}
early_stop_lgbm = {
    'lgbm__early_stopping_rounds': 30,
    'lgbm__eval_metric': 'auc',
    'lgbm__eval_set': [(X_val, y_val)],
    'lgbm__verbose': False
}

# XGBoost
pipe_xgb = Pipeline([
    ('smote', smote),
    ('xgb', XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, n_jobs=4, use_label_encoder=False, eval_metric='logloss'))
])
params_xgb = {
    'xgb__n_estimators': [100, 200, 300, 400],
    'xgb__max_depth': [3, 5, 7, 10],
    'xgb__learning_rate': [0.01, 0.03, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.8, 0.9, 1.0],
    'xgb__colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'xgb__gamma': [0, 0.1, 0.3, 0.5],
    'xgb__reg_alpha': [0.0, 0.1, 0.5],
    'xgb__reg_lambda': [0.0, 0.1, 0.5],
    'xgb__min_child_weight': [1, 5, 10],
}
early_stop_xgb = {
    'xgb__early_stopping_rounds': 30,
    'xgb__eval_metric': 'auc',
    'xgb__eval_set': [(X_val, y_val)],
    'xgb__verbose': False
}

# CatBoost (não usa pipeline direto para SMOTE, smote manual)
params_cat = {
    'iterations': [500, 800, 1000],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.03, 0.05],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 50, 100]
}

# Random Forest
pipe_rf = Pipeline([
    ('smote', smote),
    ('rf', RandomForestClassifier(random_state=42, n_jobs=4))
])
params_rf = {
    'rf__n_estimators': [100, 200, 300],
    'rf__max_depth': [None, 10, 20, 30],
    'rf__min_samples_split': [2, 5, 10],
    'rf__min_samples_leaf': [1, 2, 4]
}

# === Treino e avaliação CatBoost separado ===
def treinar_catboost():
    logging.info("\n\n===== Treinando CatBoost =====")
    print("\n===== Treinando CatBoost =====")

    # SMOTE manual no treino
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    X_trainval_sm, y_trainval_sm = smote.fit_resample(X_trainval, y_trainval)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cat = CatBoostClassifier(random_seed=42, verbose=0)

    search = RandomizedSearchCV(cat, param_distributions=params_cat, n_iter=25,
                                scoring='f1', cv=cv, n_jobs=4, verbose=2, random_state=42)
    search.fit(X_train_sm, y_train_sm)

    best_params = search.best_params_
    logging.info(f"CatBoost - Melhores hiperparâmetros: {best_params}")
    print(f"CatBoost - Melhores hiperparâmetros: {best_params}")

    # Treinamento final com early stopping usando Pool
    train_pool = Pool(X_trainval_sm, y_trainval_sm)
    val_pool = Pool(X_val, y_val)

    model_final = CatBoostClassifier(random_seed=42, verbose=0, **best_params)
    model_final.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30, verbose=0)

    # Calibração direto no modelo
    calibrated = CalibratedClassifierCV(model_final, method='sigmoid', cv='prefit')
    calibrated.fit(X_val, y_val)

    # Threshold ótimo
    y_probs_val = calibrated.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0, 1, 101)
    best_metrics = {'recall': {'threshold': 0, 'value': 0},
                    'precision': {'threshold': 0, 'value': 0},
                    'f1': {'threshold': 0, 'value': 0},
                    'accuracy': {'threshold': 0, 'value': 0}}

    for t in thresholds:
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
    logging.info(f"CatBoost - Threshold escolhido (F1): {chosen_threshold}")
    print(f"CatBoost - Threshold escolhido (F1): {chosen_threshold}")

    # Avaliação no teste
    y_probs_test = calibrated.predict_proba(X_test)[:, 1]
    y_pred_test = (y_probs_test >= chosen_threshold).astype(int)

    report = classification_report(y_test, y_pred_test, output_dict=True, digits=4)
    df_report = pd.DataFrame(report).transpose()

    auc_pr = average_precision_score(y_test, y_probs_test)
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    cost = fp * 1 + fn * 10

    logging.info(f"CatBoost - Relatório:\n{df_report}")
    logging.info(f"CatBoost - AUC-PR: {auc_pr:.4f}")
    logging.info(f"CatBoost - Matriz de Confusão:\n{cm}")
    logging.info(f"CatBoost - Custo total: {cost}")

    report_dir = config['paths']['reports_dir']
    os.makedirs(report_dir, exist_ok=True)
    df_report.to_csv(os.path.join(report_dir, 'classification_report_CatBoost.csv'), index=True)

    return {
        'model_name': 'CatBoost',
        'model': calibrated,
        'threshold': chosen_threshold,
        'f1': best_metrics['f1']['value'],
        'auc_pr': auc_pr,
        'classification_report': df_report,
        'confusion_matrix': cm,
        'cost': cost,
    }

# === Rodar treino e avaliação ===
resultados = []

# LightGBM
resultados.append(treinar_e_avaliar('LightGBM', pipe_lgbm, params_lgbm, early_stop_lgbm))

# XGBoost
resultados.append(treinar_e_avaliar('XGBoost', pipe_xgb, params_xgb, early_stop_xgb))

# CatBoost
resultados.append(treinar_catboost())

# Random Forest
resultados.append(treinar_e_avaliar('RandomForest', pipe_rf, params_rf))

# === Resumo final ===
print("\n\n=== Resultados Finais da Competição ===")
best_model = None
best_f1 = 0
best_auc = 0
for res in resultados:
    print(f"\nModelo: {res['model_name']}")
    print(f"F1-score no teste: {res['f1']:.4f}")
    print(f"AUC-PR no teste: {res['auc_pr']:.4f}")
    if res['f1'] > best_f1:
        best_f1 = res['f1']
        best_model = res['model_name']
    if res['auc_pr'] > best_auc:
        best_auc = res['auc_pr']

print(f"\nMelhor modelo por F1-score: {best_model} com F1={best_f1:.4f}")
print(f"Melhor modelo por AUC-PR: com AUC-PR={best_auc:.4f}")

# === Tempo total ===
end_time = time.time()
h, m, s = int((end_time - start_time) // 3600), int(((end_time - start_time) % 3600) // 60), int((end_time - start_time) % 60)
logging.info(f"Tempo total: {h}h {m}min {s}s")
print(f"\nTempo total: {h}h {m}min {s}s")
print("✅ Competição de modelos concluída.")
