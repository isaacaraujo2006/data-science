import os
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

# Configurações de caminho
MODEL_PATH = "C:/Github/data-science/projetos/analise_sentimento/models/modelo_sentimento.joblib"
PROCESSED_DATA_PATH = "C:/Github/data-science/projetos/analise_sentimento/data/processed/dataset_traduzido_e_tratado.parquet"

# Função para carregar o modelo supervisionado
def carregar_modelo_supervisionado(model_path):
    """
    Carrega o modelo supervisionado já treinado.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"O modelo supervisionado não foi encontrado em: {model_path}")
    print(f"Carregando modelo supervisionado de: {model_path}...")
    return joblib.load(model_path)

# Função para análise de sentimento com VADER
def analisar_sentimento_vader(texto):
    """
    Utiliza VADER para análise de sentimento.
    """
    analyzer = SentimentIntensityAnalyzer()
    if not texto or pd.isna(texto):
        return "Neutro"
    scores = analyzer.polarity_scores(texto)
    if scores["compound"] >= 0.1:
        return "Positivo"
    elif scores["compound"] <= -0.1:
        return "Negativo"
    else:
        return "Neutro"

# Função para análise de sentimento com TextBlob
def analisar_sentimento_textblob(texto):
    """
    Utiliza TextBlob para análise de sentimento.
    """
    if not texto or pd.isna(texto):
        return "Neutro"
    polarity = TextBlob(texto).sentiment.polarity
    if polarity > 0.1:
        return "Positivo"
    elif polarity < -0.1:
        return "Negativo"
    else:
        return "Neutro"

# Função para análise de sentimento com o modelo supervisionado
def analisar_sentimento_modelo_supervisionado(texto, modelo_ml):
    """
    Utiliza o modelo supervisionado para análise de sentimento e mapeia os valores numéricos para rótulos de categoria.
    """
    if not texto or pd.isna(texto):
        return "Neutro"
    # Mapeamento de valores numéricos para rótulos de sentimento
    mapping = {0: "Negativo", 1: "Neutro", 4: "Positivo"}
    predicao = modelo_ml.predict([texto])[0]
    return mapping.get(predicao, "Neutro")

# Função para determinar o sentimento final por voto majoritário
def determinar_sentimento_final(sentimentos):
    """
    Determina o sentimento final com base no voto majoritário.
    """
    votos = pd.Series(sentimentos).value_counts()
    return votos.idxmax()

# Pipeline para analisar mensagens
def pipeline_analise_sentimento(mensagens, modelo_ml):
    """
    Realiza a análise de sentimento utilizando VADER, TextBlob e Modelo Supervisionado
    e determina o resultado por voto majoritário.
    """
    resultados = []
    for mensagem in mensagens:
        print(f"\nAnalisando mensagem: \"{mensagem}\"")
        
        # Análise com VADER
        sentimento_vader = analisar_sentimento_vader(mensagem)
        print(f"Sentimento pelo VADER: {sentimento_vader}")
        
        # Análise com TextBlob
        sentimento_textblob = analisar_sentimento_textblob(mensagem)
        print(f"Sentimento pelo TextBlob: {sentimento_textblob}")
        
        # Análise com Modelo Supervisionado
        sentimento_modelo = analisar_sentimento_modelo_supervisionado(mensagem, modelo_ml)
        print(f"Sentimento pelo Modelo Supervisionado: {sentimento_modelo}")
        
        # Determinar sentimento final
        sentimento_final = determinar_sentimento_final(
            [sentimento_vader, sentimento_textblob, sentimento_modelo]
        )
        print(f"Sentimento final (voto majoritário): {sentimento_final}")
        resultados.append((mensagem, sentimento_final))
    
    return resultados

# Função principal
if __name__ == "__main__":
    # Carregar modelo supervisionado
    modelo_supervisionado = carregar_modelo_supervisionado(MODEL_PATH)
    
    # Mensagens de teste
    mensagens_teste = [
        "I absolutely love this product!",
        "This is the worst experience I've ever had.",
        "It's okay, not great but not bad.",
        "Amazing quality, exceeded expectations!",
        "Terrible, will never recommend this to anyone."
    ]
    
    # Executar pipeline
    resultados = pipeline_analise_sentimento(mensagens_teste, modelo_supervisionado)
    
    print("\nResultados da análise de sentimento:")
    for mensagem, sentimento in resultados:
        print(f"Mensagem: \"{mensagem}\" --> Sentimento: {sentimento}")
