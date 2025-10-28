import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

# Suprimir warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Caminho do arquivo CSV
data_path = "C:/Github/data-science/projetos/churn_clientes/data/raw/rclientes.csv"

# Carregar o dataset
df = pd.read_csv(data_path)

# Tradução das colunas
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

# Remover colunas desnecessárias
df.drop(columns=['CustomerId', 'Surname'], errors='ignore', inplace=True)

# Remover duplicatas
df.drop_duplicates(inplace=True)

# Tratar valores faltantes
df.fillna(df.mean(numeric_only=True), inplace=True)

# Criar variável alvo (Churn) garantindo tipo inteiro
np.random.seed(42)
df['classe'] = np.random.choice([0, 1], size=len(df)).astype(int)

# One-Hot Encoding para variáveis categóricas
df = pd.get_dummies(df, columns=['geografia', 'genero'], drop_first=True)

# Normalização
scaler = StandardScaler()
colunas_numericas = df.select_dtypes(include=[float, int]).columns
df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

# Salvar o scaler
joblib.dump(scaler, 'scaler.joblib')

# Garantir que o índice seja do tipo pd.Index
df = df.reset_index(drop=True)

# Separar features e target
X = df.drop(columns=['classe'])
y = df['classe'].astype(int)  # Garante que y seja inteiro

# Aplicar SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Garantir que y_res também seja inteiro
y_res = y_res.astype(int)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Definir modelos
models = {
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "Random Forest": RandomForestClassifier(random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

# Armazenar os resultados
results = {}

# Treinar e avaliar os modelos
for model_name, model in models.items():
    print(f"\nTreinando {model_name}...")
    
    # Treinamento
    model.fit(X_train, y_train)
    
    # Previsões
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Relatórios
    train_report = classification_report(y_train, y_pred_train, output_dict=True)
    test_report = classification_report(y_test, y_pred_test, output_dict=True)
    
    results[model_name] = {
        'Train': train_report,
        'Test': test_report
    }
    
    # Mostrar resultados
    print(f"Relatório de Classificação - {model_name} - Treino:\n", classification_report(y_train, y_pred_train))
    print(f"Relatório de Classificação - {model_name} - Teste:\n", classification_report(y_test, y_pred_test))

# Comparar os resultados e identificar o melhor modelo
best_model_name = min(results, key=lambda model: results[model]['Test']['accuracy'])
print(f"\nMelhor Modelo: {best_model_name}")

# Relatórios Finais
final_reports = {model: {
                    'Treino': results[model]['Train'],
                    'Teste': results[model]['Test']
                } for model in results}

# Salvar relatórios finais
with open('C:/Github/data-science/projetos/churn_clientes/logs/classification_reports.json', 'w') as f:
    json.dump(final_reports, f)

# Salvar o modelo final
best_model = models[best_model_name]
joblib.dump(best_model, 'best_model.joblib')

print("Relatórios Finais de Classificação salvos.")
