import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import os

# Configuração de logs
logs_dir = r"C:\Github\data-science\projetos\segmentacao_clientes\logs"
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, "modeling.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Iniciando o pipeline de modelagem.")

# Carregar o dataset processado
file_path = r"C:\Github\data-science\projetos\segmentacao_clientes\data\processed\processed.csv"
df = pd.read_csv(file_path)

# Exibindo barra de progresso para leitura inicial
with tqdm(total=1, desc="Carregando Dataset") as pbar:
    pbar.update(1)

# Etapa 1: Divisão de dados em treinamento e teste (70/30)
logging.info("Dividindo os dados em conjuntos de treinamento e teste.")
X = df.drop(columns=["id", "resposta", "data_cliente"])  # Remova colunas irrelevantes
y = None  # Como K-Means é não supervisionado, não temos um y (variável alvo)

# Dividindo em treinamento e teste
X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
logging.info(f"Tamanho do conjunto de treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")

# Exibindo barra de progresso para divisão
with tqdm(total=1, desc="Dividindo Dados") as pbar:
    pbar.update(1)

# Etapa 2: Engenharia de Features
logging.info("Aplicando engenharia de features.")
# Criar uma nova variável combinando gastos totais
X_train["gastos_totais"] = (
    X_train[["gastos_vinhos", "gastos_frutas", "gastos_carnes", "gastos_peixes", "gastos_doces", "gastos_joias"]].sum(axis=1)
)
X_test["gastos_totais"] = (
    X_test[["gastos_vinhos", "gastos_frutas", "gastos_carnes", "gastos_peixes", "gastos_doces", "gastos_joias"]].sum(axis=1)
)

# Exibindo barra de progresso para engenharia de features
with tqdm(total=1, desc="Engenharia de Features") as pbar:
    pbar.update(1)

# Etapa 3: Pré-processamento com One-Hot Encoding e Normalização
logging.info("Codificando variáveis categóricas e normalizando os dados.")
categorical_features = X_train.select_dtypes(include=["object"]).columns
numerical_features = X_train.select_dtypes(include=["float64", "int64"]).columns

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)  # Adicionado handle_unknown="ignore"
    ]
)

# Pré-processar os conjuntos de dados
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Exibindo barra de progresso para normalização
with tqdm(total=1, desc="Normalizando Dados") as pbar:
    pbar.update(1)

# Etapa 4: Aplicar o K-Means Clustering
logging.info("Treinando o modelo K-Means.")
# Determinando o número ideal de clusters usando o método do cotovelo
wcss = []
for k in tqdm(range(2, 11), desc="Executando Método do Cotovelo"):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train_preprocessed)
    wcss.append(kmeans.inertia_)

# Salvar gráfico do cotovelo para interpretação
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), wcss, marker="o")
plt.title("Método do Cotovelo")
plt.xlabel("Número de Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
cotovelo_path = os.path.join(logs_dir, "cotovelo.png")
plt.savefig(cotovelo_path)
plt.close()
logging.info(f"Gráfico do método do cotovelo salvo em: {cotovelo_path}")

# Treinar modelo final com o número ideal de clusters (e.g., k=5)
kmeans_final = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans_final.fit(X_train_preprocessed)

# Avaliação com Coeficiente de Silhueta
silhouette_avg = silhouette_score(X_train_preprocessed, kmeans_final.labels_)
logging.info(f"Coeficiente de Silhueta para k=5: {silhouette_avg:.4f}")

# Exibindo barra de progresso para treinamento do modelo
with tqdm(total=1, desc="Treinando K-Means") as pbar:
    pbar.update(1)

# Etapa 5: Relatório de Classificação de Treino
# Adicionar os clusters como rótulos ao conjunto de treinamento
X_train["cluster"] = kmeans_final.labels_

# Relatório dos clusters (estatísticas de cada grupo)
relatorio_clusters = X_train.groupby("cluster").mean(numeric_only=True)  # Resolvido FutureWarning
logging.info("Gerando o relatório de clusters (treino).")

# Salvar relatório em CSV
relatorio_path = os.path.join(logs_dir, "relatorio_clusters_treino.csv")
relatorio_clusters.to_csv(relatorio_path)
logging.info(f"Relatório de clusters salvo em: {relatorio_path}")

# Relatório de Cluster no Console
print("\nRelatório de Clusters (Treinamento):")
print(relatorio_clusters)

logging.info("Pipeline de modelagem concluído com sucesso.")
print("\nPipeline concluído. Relatório salvo no diretório de logs.")

# Salvar os dados com os clusters atribuídos
clustered_data_path = r"C:\Github\data-science\projetos\segmentacao_clientes\data\processed\clustered_data.csv"
X_train["id"] = df.loc[X_train.index, "id"]  # Reincorporar a coluna 'id' original
X_train.to_csv(clustered_data_path, index=False)
logging.info(f"Dados segmentados salvos em: {clustered_data_path}")

# Reincorporando a coluna 'id' ao dataset segmentado
X_train["id"] = df.loc[X_train.index, "id"]  # Recupera os IDs dos clientes
X_train.to_csv(clustered_data_path, index=False)

logging.info(f"Dados segmentados salvos em: {clustered_data_path}")
print(f"Dados segmentados salvos em: {clustered_data_path}")