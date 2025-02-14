import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import logging
import os
import joblib
import yaml

# Carregar o arquivo de configuração
config_path = r'C:/Github/data-science/projetos/churn_clientes/config/config.yaml'
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Configurar logging
logs_dir = config['paths']['logs_dir']
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logging.basicConfig(filename=os.path.join(logs_dir, 'modeling.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")

# Definir a seed
seed = 42

# Caminho do dataset
dataset_path = config['data']['processed']

# Verificar se o arquivo existe
if not os.path.exists(dataset_path):
    logging.error(f"Arquivo não encontrado: {dataset_path}")
    raise FileNotFoundError(f"Arquivo não encontrado: {dataset_path}")

# Carregar os dados
logging.info('Carregando dados')
df = pd.read_parquet(dataset_path)

# Codificação One-Hot Encoding para variáveis categóricas
df = pd.get_dummies(df, drop_first=True)

# Garantir que a coluna 'churn' esteja binária {0, 1}
df['churn'] = df['churn'].apply(lambda x: 1 if x > 0 else 0)

# Separar features e alvo
X = df.drop('churn', axis=1)
y = df['churn']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Normalizar e padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Salvar o scaler
scaler_path = config['preprocessors']['path']
if not os.path.exists(scaler_path):
    os.makedirs(scaler_path)
joblib.dump(scaler, os.path.join(scaler_path, 'scaler.joblib'))

logging.info('Dados carregados, pré-processados e scaler salvo.')
print("### Etapa 1 Concluída: Carregamento e Pré-processamento dos Dados ###")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import joblib
import logging

# Função para treinar e avaliar o modelo
def avaliar_modelo(modelo, X_train, X_test, y_train, y_test, nome_modelo):
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]
    acuracia = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    relatorio_classificacao = classification_report(y_test, y_pred)
    
    logging.info(f"\n### {nome_modelo} ###\n")
    logging.info(relatorio_classificacao)
    logging.info(f"Acurácia: {acuracia:.2f}")
    logging.info(f"AUC-ROC: {auc_roc:.2f}")
    
    print(f"\n### {nome_modelo} ###\n")
    print(relatorio_classificacao)
    print(f"Acurácia: {acuracia:.2f}")
    print(f"AUC-ROC: {auc_roc:.2f}")
    
    return modelo, acuracia, auc_roc, y_pred_proba, relatorio_classificacao

# Treinar e testar o modelo com os parâmetros padrão
modelo_lr = LogisticRegression(random_state=42)
modelo_lr, acuracia, auc_roc, y_pred_proba_lr, relatorio_classificacao_lr = avaliar_modelo(modelo_lr, X_train, X_test, y_train, y_test, "Regressão Logística")

# Encontrar os melhores hiperparâmetros usando GridSearchCV
param_grid = {
    'C': [0.1, 1.0, 10.0],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}

grid_search = GridSearchCV(estimator=LogisticRegression(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

melhores_parametros = grid_search.best_params_
modelo_lr_otimizado = grid_search.best_estimator_

# Salvar os hiperparâmetros
hyperparameters_path = os.path.join(config['models']['directory'], 'logistic_regression_hyperparameters.joblib')
if not os.path.exists(config['models']['directory']):
    os.makedirs(config['models']['directory'])
joblib.dump(melhores_parametros, hyperparameters_path)

logging.info("\n### Melhor Modelo: Regressão Logística com Grid Search ###\n")
logging.info(f"Melhores Parâmetros: {melhores_parametros}")

# Prever e avaliar o modelo otimizado
modelo_lr_otimizado, acuracia_lr_otimizado, auc_roc_lr_otimizado, y_pred_proba_lr_otimizado, relatorio_classificacao_lr_otimizado = avaliar_modelo(modelo_lr_otimizado, X_train, X_test, y_train, y_test, "Regressão Logística Otimizada")

# Salvar o modelo otimizado
model_path = os.path.join(config['models']['directory'], 'logistic_regression_model.joblib')
joblib.dump(modelo_lr_otimizado, model_path)

logging.info('Modelo treinado, otimizado e salvo.')
print("### Etapa 2 Concluída: Treinamento e Avaliação do Modelo ###")

from sklearn.metrics import precision_recall_curve

# Validação cruzada
y_pred_cross_val = cross_val_predict(modelo_lr_otimizado, X_train, y_train, cv=5, method='predict_proba')[:, 1]
acuracia_cross_val = accuracy_score(y_train, np.round(y_pred_cross_val))
auc_roc_cross_val = roc_auc_score(y_train, y_pred_cross_val)

relatorio_cross_val = classification_report(y_train, np.round(y_pred_cross_val))

logging.info("\n### Validação Cruzada ###\n")
logging.info(relatorio_cross_val)
logging.info(f"Acurácia: {acuracia_cross_val:.2f}")
logging.info(f"AUC-ROC: {auc_roc_cross_val:.2f}")

print("\n### Validação Cruzada ###\n")
print(relatorio_cross_val)
print(f"Acurácia: {acuracia_cross_val:.2f}")
print(f"AUC-ROC: {auc_roc_cross_val:.2f}")

# Encontrar o melhor threshold
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba_lr_otimizado)
f1_scores = 2 * recall * precision / (recall + precision)
melhor_threshold = thresholds[np.argmax(f1_scores)]

# Aplicar o melhor threshold
y_pred_threshold = (y_pred_proba_lr_otimizado >= melhor_threshold).astype(int)

relatorio_threshold = classification_report(y_test, y_pred_threshold)
acuracia_threshold = accuracy_score(y_test, y_pred_threshold)
auc_roc_threshold = roc_auc_score(y_test, y_pred_proba_lr_otimizado)

logging.info("\n### Melhor Threshold ###\n")
logging.info(f"Melhor Threshold: {melhor_threshold:.2f}")
logging.info(relatorio_threshold)
logging.info(f"Acurácia: {acuracia_threshold:.2f}")
logging.info(f"AUC-ROC: {auc_roc_threshold:.2f}")

print("\n### Melhor Threshold ###\n")
print(f"Melhor Threshold: {melhor_threshold:.2f}")
print(relatorio_threshold)
print(f"Acurácia: {acuracia_threshold:.2f}")
print(f"AUC-ROC: {auc_roc_threshold:.2f}")

# Salvar as métricas
metrics_path = os.path.join(config['metrics']['directory'], 'metrics.txt')
if not os.path.exists(config['metrics']['directory']):
    os.makedirs(config['metrics']['directory'])
with open(metrics_path, 'w') as f:
    f.write(f"### Regressão Logística ###\n\n{relatorio_classificacao_lr}\n\n")
    f.write(f"Acurácia: {acuracia:.2f}\n")
    f.write(f"AUC-ROC: {auc_roc:.2f}\n\n")

    f.write(f"### Regressão Logística Otimizada ###\n\n{relatorio_classificacao_lr_otimizado}\n\n")
    f.write(f"Acurácia: {acuracia_lr_otimizado:.2f}\n")
    f.write(f"AUC-ROC: {auc_roc_lr_otimizado:.2f}\n\n")

    f.write(f"### Validação Cruzada ###\n\n{relatorio_cross_val}\n\n")
    f.write(f"Acurácia: {acuracia_cross_val:.2f}\n")
    f.write(f"AUC-ROC: {auc_roc_cross_val:.2f}\n\n")

    f.write(f"### Melhor Threshold ###\n\n{relatorio_threshold}\n\n")
    f.write(f"Acurácia: {acuracia_threshold:.2f}\n")
    f.write(f"AUC-ROC: {auc_roc_threshold:.2f}\n")

logging.info('Validação cruzada, ajuste de threshold e salvamento das métricas concluídos.')
print("### Etapa 3 Concluída: Validação Cruzada e Ajuste de Threshold ###")