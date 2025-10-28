import joblib

# Caminho para o arquivo do modelo
model_path = 'C:/Github/data-science/projetos/churn_clientes/models/final_model.joblib'

# Carregar o modelo
model = joblib.load(model_path)

# Imprimir informações sobre o modelo
print(model)