import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report
import json

# Caminho do arquivo CSV
data_path = "C:/Github/data-science/projetos/churn_clientes/data/raw/rclientes.csv"

# Carregar o dataset
df = pd.read_csv(data_path)

# Traduzir colunas
df.rename(columns={
    'Age': 'idade',
    'Balance': 'saldo',
    'NumOfProducts': 'numero_produtos',
    'EstimatedSalary': 'salario_estimado',
    'Tenure': 'tempo_relacionamento',
    'CreditScore': 'pontuacao_credito',
    'Geography': 'geografia',
    'Gender': 'genero'
}, inplace=True)

# Criar a coluna 'tem_cartao_credito' (considerando que o cliente tenha cartão de crédito se a pontuação de crédito for maior que um limite)
df['tem_cartao_credito'] = df['pontuacao_credito'].apply(lambda x: 1 if x > 500 else 0)

# Selecionar apenas as 5 características desejadas
features = ['idade', 'saldo', 'tempo_relacionamento', 'salario_estimado', 'tem_cartao_credito']

# Reajustar o StandardScaler
scaler = StandardScaler()
scaler.fit(df[features])

# Salvar o novo StandardScaler
joblib.dump(scaler, 'C:/Github/data-science/projetos/churn_clientes/preprocessors/scaler_5_features.joblib')
print("Novo StandardScaler com 5 características ajustado e salvo com sucesso!")

# Remover colunas desnecessárias
df.drop(columns=['CustomerId', 'Surname'], errors='ignore', inplace=True)

# Remover duplicatas
df.drop_duplicates(inplace=True)

# Tratar valores faltantes
df.fillna(df.mean(numeric_only=True), inplace=True)

# Criar variável alvo (Churn)
np.random.seed(42)
df['classe'] = np.random.choice([0, 1], size=len(df)).astype(int)

# One-Hot Encoding para variáveis categóricas
df = pd.get_dummies(df, columns=['geografia', 'genero'], drop_first=True)

# Normalização
df[features] = scaler.transform(df[features])

# Garantir que o índice seja do tipo pd.Index
df = df.reset_index(drop=True)

# Separar features e target
X = df[features]
y = df['classe'].astype(int)  # Garante que y seja inteiro

# Usando SMOTE para gerar exemplos sintéticos
smote = SMOTE(random_state=42)

# Aplicando SMOTE
X_res, y_res = smote.fit_resample(X, y)

# Divisão treino/teste (70% treino, 30% teste)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Modelo XGBoost com Ajuste de Hiperparâmetros
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                              scale_pos_weight=1,  # Melhor ajuste de regularização
                              lambda_=1, alpha=1)  # Regularização L2 e L1

# Treinamento do modelo XGBoost
xgb_model.fit(X_train, y_train)

# Previsões
y_pred_train = xgb_model.predict(X_train)

# Relatório de classificação completo
print(f"Relatório de Classificação - XGBoost - Treino:\n", classification_report(y_train, y_pred_train))

# Ajuste de Threshold
y_pred_prob_train = xgb_model.predict_proba(X_train)[:, 1]
thresholds = np.linspace(0.1, 0.9, 9)
best_threshold = 0.5  # Default threshold, ajustaremos depois

# Encontrar o melhor threshold baseado em F1-Score
best_f1 = 0
for threshold in thresholds:
    y_pred_threshold = (y_pred_prob_train >= threshold).astype(int)
    f1_score = classification_report(y_train, y_pred_threshold, output_dict=True)['weighted avg']['f1-score']
    if f1_score > best_f1:
        best_f1 = f1_score
        best_threshold = threshold

print(f"\nMelhor Threshold: {best_threshold}")

# Aplicar o melhor threshold
y_pred_train_threshold = (xgb_model.predict_proba(X_train)[:, 1] >= best_threshold).astype(int)

# Relatório de classificação com o threshold ajustado
print(f"Relatório de Classificação com Threshold Ajustado - Treino:\n", classification_report(y_train, y_pred_train_threshold))

# Salvar o modelo XGBoost
joblib.dump(xgb_model, 'C:/Github/data-science/projetos/churn_clientes/models/final_model.joblib')

# Salvar o scaler
joblib.dump(scaler, 'C:/Github/data-science/projetos/churn_clientes/preprocessors/scaler.joblib')

# Salvar o pré-processador (informações sobre as colunas)
preprocessor = {'columns': list(X.columns)}
with open('C:/Github/data-science/projetos/churn_clientes/preprocessors/preprocessor.json', 'w') as f:
    json.dump(preprocessor, f)

print("Modelo, scaler e pré-processador de treino salvos.")
