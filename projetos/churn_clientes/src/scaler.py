import joblib

# Caminho para o arquivo do StandardScaler
scaler_path = 'C:/Github/data-science/projetos/churn_clientes/preprocessors/scaler_8_features.joblib'

# Carregar o StandardScaler
scaler = joblib.load(scaler_path)

# Imprimir as médias e escalas ajustadas
print("Mean:", scaler.mean_)
print("Scale:", scaler.scale_)
print("Feature Names:", scaler.feature_names_in_)
