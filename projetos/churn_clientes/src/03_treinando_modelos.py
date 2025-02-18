import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import logging
from sklearn.preprocessing import LabelEncoder

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Carregar os dados
df = pd.read_csv('C:/Github/data-science/projetos/churn_clientes/data/raw/rclientes.csv')

# Exibir as primeiras linhas para identificar problemas
logging.info(f'Dados originais: {df.head()}')

# 1. Tratar valores nulos - remover ou preencher
df = df.dropna()  # Remove linhas com valores nulos, ou use df.fillna() para preencher

# 2. Corrigir problemas de formatação (se necessário)
# Substituindo o valor "P'an" por um valor correto como "France"
df['Geography'] = df['Geography'].replace("P'an", "France")

# 3. Verificar valores únicos nas colunas categóricas e substituir valores problemáticos
logging.info(f'Valores únicos em "Geography": {df["Geography"].unique()}')
logging.info(f'Valores únicos em "Gender": {df["Gender"].unique()}')

# Codificar variáveis categóricas
label_encoder = LabelEncoder()

# Corrigir "Geography" para valores numéricos
df['Geography'] = df['Geography'].replace({"France": 0, "Spain": 1, "Germany": 2})
# Corrigir "Gender" para valores numéricos
df['Gender'] = df['Gender'].replace({"Female": 0, "Male": 1})

# Exibir as primeiras linhas após a transformação para garantir que tudo foi feito corretamente
logging.info(f'Dados após tratamento: {df.head()}')

# Remover colunas irrelevantes para o modelo
df = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

# Verificar se existem valores não numéricos nas colunas que deveriam ser numéricas
for col in df.select_dtypes(include=['object']).columns:
    unique_values = df[col].unique()
    if any(isinstance(val, str) for val in unique_values):
        logging.warning(f'A coluna "{col}" ainda contém valores não numéricos: {unique_values}')

# Definindo X (features) e y (target)
X = df.drop(columns='Exited')  # Todas as colunas, exceto a coluna 'Exited'
y = df['Exited']  # A coluna 'Exited' é o alvo

# Divisão em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Oversampling com SMOTE para balancear as classes
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 2. Treinamento com ajuste de pesos no Random Forest (class_weight='balanced')
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train_res, y_train_res)

# Previsões no conjunto de teste
y_pred = rf.predict(X_test)

# 3. Avaliação do modelo com as métricas padrão
accuracy = rf.score(X_test, y_test)
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
logging.info(f'Accuracy: {accuracy}')
logging.info(f'ROC AUC: {roc_auc}')

# Classification Report
logging.info('Classification Report:')
logging.info(classification_report(y_test, y_pred))

# Confusion Matrix
logging.info('Confusion Matrix:')
logging.info(confusion_matrix(y_test, y_pred))

# 4. Ajuste do limiar de decisão para melhorar recall da classe 1 (churn)
y_pred_proba = rf.predict_proba(X_test)[:, 1]
threshold = 0.3  # Limiar ajustado
y_pred_adjusted = (y_pred_proba >= threshold).astype(int)

# Avaliação após ajuste do limiar
logging.info(f'Ajuste do limiar de decisão (Threshold = {threshold}):')
logging.info(f'Classification Report (Threshold = {threshold}):')
logging.info(classification_report(y_test, y_pred_adjusted))

logging.info(f'Confusion Matrix (Threshold = {threshold}):')
logging.info(confusion_matrix(y_test, y_pred_adjusted))

# 5. Avaliação de outros modelos (como no seu código original)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC, NuSVC
from xgboost import XGBClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': rf,  # Usando o Random Forest treinado
    'Gradient Boosting': GradientBoostingClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'SVC': SVC(probability=True),
    'NuSVC': NuSVC(probability=True),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

for name, model in models.items():
    logging.info(f'Treinando o modelo {name}...')
    model.fit(X_train_res, y_train_res)  # Treinamento com os dados balanceados

    # Avaliação de cada modelo
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    accuracy = model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    logging.info(f'{name} Results:')
    logging.info(f'Accuracy: {accuracy}')
    logging.info(f'ROC AUC: {roc_auc}')
    logging.info('Classification Report:')
    logging.info(classification_report(y_test, y_pred))
    logging.info('Confusion Matrix:')
    logging.info(confusion_matrix(y_test, y_pred))

    # Ajuste do limiar de decisão para cada modelo
    y_pred_adjusted = (y_pred_proba >= threshold).astype(int)
    logging.info(f'{name} - Ajuste do limiar de decisão (Threshold = {threshold}):')
    logging.info(f'Classification Report (Threshold = {threshold}):')
    logging.info(classification_report(y_test, y_pred_adjusted))
    logging.info(f'Confusion Matrix (Threshold = {threshold}):')
    logging.info(confusion_matrix(y_test, y_pred_adjusted))
