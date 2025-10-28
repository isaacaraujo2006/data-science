import os
import pandas as pd
import joblib
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# Caminhos e configurações
PROCESSED_DATA_PATH = "C:/Github/data-science/projetos/analise_sentimento/data/processed/dataset_traduzido_e_tratado.parquet"
MODEL_SAVE_PATH = "C:/Github/data-science/projetos/analise_sentimento/models/modelo_sentimento.joblib"

# Função para treinar, avaliar e salvar o modelo supervisionado
def treinar_e_avaliar_modelo(dataset_path, model_save_path):
    """
    Treina um modelo supervisionado, exibe as métricas de desempenho e salva o modelo treinado.
    """
    print(f"Carregando dataset para treinamento: {dataset_path}...")
    df = pd.read_parquet(dataset_path)
    
    # Verificar e remover textos vazios
    df = df.dropna(subset=['tweet'])  # Remove linhas onde 'tweet' é NaN
    df = df[df['tweet'].str.strip() != '']  # Remove linhas onde 'tweet' é vazio ou contém apenas espaços em branco
    
    # Divisão de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        df['tweet'], df['sentimento'], test_size=0.2, random_state=42
    )
    
    # Criar pipeline com CountVectorizer e RandomForest
    pipeline = make_pipeline(
        CountVectorizer(stop_words=None),  # Não remove stop words
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    )
    
    print("Treinando modelo supervisionado...")
    n_chunks = 10  # Dividir os dados de treinamento em 10 partes
    chunk_size = len(X_train) // n_chunks
    for i in tqdm(range(n_chunks), desc="Treinando Modelo"):
        start_index = i * chunk_size
        end_index = start_index + chunk_size
        if i == n_chunks - 1:  # Última parte inclui todos os dados restantes
            end_index = len(X_train)
        pipeline.fit(X_train[start_index:end_index], y_train[start_index:end_index])
    
    print("Avaliando o modelo...")
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    print("Métricas de desempenho:")
    print(report)
    
    # Salvar modelo treinado
    print(f"Salvando modelo treinado em {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print("Modelo salvo com sucesso!")

# Executar o pipeline
if __name__ == "__main__":
    treinar_e_avaliar_modelo(PROCESSED_DATA_PATH, MODEL_SAVE_PATH)
