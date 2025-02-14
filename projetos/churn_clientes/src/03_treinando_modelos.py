import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import warnings
import logging
import os

# Configurar logging
logs_dir = r'C:/Github/data-science/projetos/churn_clientes/logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)
logging.basicConfig(filename=os.path.join(logs_dir, 'modeling.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

warnings.filterwarnings("ignore")

# Caminho do dataset
dataset_path = r'C:/Github/data-science/projetos/churn_clientes/data/processed/processed.parquet'

# Carregar os dados
logging.info('Carregando dados')
df = pd.read_parquet(dataset_path)

# Codificação One-Hot Encoding para variáveis categóricas
df = pd.get_dummies(df, drop_first=True)

# Garantir que a coluna 'churn' seja de tipo categórico
df['churn'] = df['churn'].astype('int')

# Separar features e alvo
X = df.drop('churn', axis=1)
y = df['churn']

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar e padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Função para treinar e avaliar modelos
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
    
    return acuracia, auc_roc

# Modelos a serem treinados
modelos = [
    (LogisticRegression(), "Regressão Logística"),
    (DecisionTreeClassifier(), "Árvore de Decisão"),
    (RandomForestClassifier(), "Floresta Aleatória"),
    (MLPClassifier(max_iter=300), "Rede Neural")
]

resultados = []

# Treinar e avaliar cada modelo
for modelo, nome_modelo in modelos:
    acuracia, auc_roc = avaliar_modelo(modelo, X_train, X_test, y_train, y_test, nome_modelo)
    resultados.append((nome_modelo, acuracia, auc_roc))

# Selecionar o melhor modelo
melhor_modelo = max(resultados, key=lambda item: item[2])  # Baseado na AUC-ROC

logging.info("\n### Melhor Modelo Final ###\n")
logging.info(f"Modelo: {melhor_modelo[0]}")
logging.info(f"Acurácia: {melhor_modelo[1]:.2f}")
logging.info(f"AUC-ROC: {melhor_modelo[2]:.2f}")

print("\n### Melhor Modelo Final ###")
print(f"Modelo: {melhor_modelo[0]}")
print(f"Acurácia: {melhor_modelo[1]:.2f}")
print(f"AUC-ROC: {melhor_modelo[2]:.2f}")
