import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Configurando log
logging.basicConfig(level=logging.INFO)

# Carregar dados
data = pd.read_csv(r'C:\Github\data-science\projetos\churn_clientes\data\raw\rclientes.csv')

# Exibindo colunas disponíveis
logging.info(f"Colunas disponíveis: {data.columns.tolist()}")

# Codificar variáveis categóricas
data = pd.get_dummies(data, drop_first=True)  # One-hot encoding para variáveis categóricas

# Separando variáveis independentes e dependentes
X = data.drop(['Exited'], axis=1)  # Remover a coluna alvo 'Exited'
y = data['Exited']

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalando as variáveis numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inicializando o modelo RandomForest
rf_model = RandomForestClassifier(random_state=42)

# Treinando o modelo no conjunto de treino
logging.info("Treinando o modelo Random Forest...")

# Encontrando o melhor conjunto de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_scaled, y_train)

# Melhor modelo com hiperparâmetros otimizados
logging.info(f"Melhores Hiperparâmetros: {grid_search.best_params_}")

# Aplicando o melhor modelo aos dados de treino e teste
best_rf_model = grid_search.best_estimator_

# Previsões e avaliação do modelo otimizado
y_train_pred_best = best_rf_model.predict(X_train_scaled)
y_test_pred_best = best_rf_model.predict(X_test_scaled)

# Relatório de classificação para o modelo otimizado
logging.info("Relatório de Classificação para Treino (Modelo Otimizado):")
logging.info(classification_report(y_train, y_train_pred_best))

logging.info("Relatório de Classificação para Teste (Modelo Otimizado):")
logging.info(classification_report(y_test, y_test_pred_best))

# Matriz de Confusão para Treino e Teste
logging.info("Matriz de Confusão para Treino:")
logging.info(confusion_matrix(y_train, y_train_pred_best))

logging.info("Matriz de Confusão para Teste:")
logging.info(confusion_matrix(y_test, y_test_pred_best))

# Validação cruzada
cross_val_train = cross_val_score(best_rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cross_val_test = cross_val_score(best_rf_model, X_test_scaled, y_test, cv=5, scoring='accuracy')

logging.info(f"Validação Cruzada (Treino): {cross_val_train.mean()} ± {cross_val_train.std()}")
logging.info(f"Validação Cruzada (Teste): {cross_val_test.mean()} ± {cross_val_test.std()}")

# Encontrando o melhor threshold
probabilities = best_rf_model.predict_proba(X_test_scaled)[:, 1]
thresholds = np.arange(0.0, 1.1, 0.05)

best_threshold = 0.5
best_f1 = 0
for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)
    f1 = f1_score(y_test, predictions)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

logging.info(f"Melhor Threshold: {best_threshold} com F1-Score: {best_f1}")

# Aplicando o melhor threshold nos dados de teste
y_test_pred_best_threshold = (probabilities >= best_threshold).astype(int)

# Relatório de classificação com o melhor threshold
logging.info("Relatório de Classificação com o Melhor Threshold:")
logging.info(classification_report(y_test, y_test_pred_best_threshold))

# Matriz de Confusão com o melhor threshold
logging.info("Matriz de Confusão com o Melhor Threshold:")
logging.info(confusion_matrix(y_test, y_test_pred_best_threshold))
