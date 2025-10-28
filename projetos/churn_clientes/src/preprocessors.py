import json

# Caminho para o arquivo JSON do pré-processador
preprocessor_path = 'C:/Github/data-science/projetos/churn_clientes/preprocessors/preprocessor.json'

# Carregar o pré-processador
with open(preprocessor_path, 'r') as f:
    preprocessor = json.load(f)

# Imprimir o conteúdo do pré-processador
print(json.dumps(preprocessor, indent=4))
