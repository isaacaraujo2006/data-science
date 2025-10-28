import os
import streamlit as st
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib

# Configurações de caminho
MODEL_PATH = "C:/Github/data-science/projetos/analise_sentimento/models/modelo_sentimento.joblib"

# Função para carregar o modelo supervisionado
@st.cache_resource
def carregar_modelo_supervisionado(model_path):
    """
    Carrega o modelo supervisionado já treinado.
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"O modelo supervisionado não foi encontrado em: {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Erro ao carregar o modelo supervisionado: {e}")
        return None

# Função para análise de sentimento com VADER
def analisar_sentimento_vader(texto):
    """
    Utiliza VADER para análise de sentimento.
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        if not texto or pd.isna(texto):
            return "Neutro"
        scores = analyzer.polarity_scores(texto)
        if scores["compound"] >= 0.15:
            return "Positivo"
        elif scores["compound"] <= -0.05:
            return "Negativo"
        else:
            return "Neutro"
    except Exception as e:
        st.warning(f"Erro ao analisar texto com VADER: {e}")
        return "Neutro"

# Função para análise de sentimento com TextBlob
def analisar_sentimento_textblob(texto):
    """
    Utiliza TextBlob para análise de sentimento.
    """
    try:
        if not texto or pd.isna(texto):
            return "Neutro"
        polarity = TextBlob(texto).sentiment.polarity
        if polarity > 0.15:
            return "Positivo"
        elif polarity < -0.15:
            return "Negativo"
        else:
            return "Neutro"
    except Exception as e:
        st.warning(f"Erro ao analisar texto com TextBlob: {e}")
        return "Neutro"

# Função para análise de sentimento com o modelo supervisionado
def analisar_sentimento_modelo_supervisionado(texto, modelo_ml):
    """
    Utiliza o modelo supervisionado para análise de sentimento.
    """
    try:
        if not texto or pd.isna(texto):
            return "Neutro"
        # Mapeamento de valores numéricos para rótulos de sentimento
        mapping = {0: "Negativo", 1: "Neutro", 4: "Positivo"}
        predicao = modelo_ml.predict([texto])[0]
        return mapping.get(predicao, "Neutro")
    except Exception as e:
        st.warning(f"Erro ao analisar texto com Modelo Supervisionado: {e}")
        return "Neutro"

# Função para determinar o sentimento final por voto majoritário
def determinar_sentimento_final(sentimentos):
    """
    Determina o sentimento final com base no voto majoritário.
    """
    try:
        votos = pd.Series(sentimentos).value_counts()
        return votos.idxmax()
    except Exception as e:
        st.warning(f"Erro ao determinar sentimento final: {e}")
        return "Neutro"

# Função principal para realizar a análise
def analisar_mensagem(mensagem, modelo_ml):
    """
    Realiza a análise de sentimento utilizando VADER, TextBlob e Modelo Supervisionado.
    """
    try:
        # Detecção por padrões críticos
        padroes_criticos = ["não sabe", "horrível", "péssimo", "muito ruim", "frustrante", "terrível experiência"]
        if any(padrao in mensagem.lower() for padrao in padroes_criticos):
            return {
                "Mensagem": mensagem,
                "Sentimento VADER": "N/A",
                "Sentimento TextBlob": "N/A",
                "Sentimento Modelo Supervisionado": "N/A",
                "Sentimento Final": "Negativo (Padrão Crítico Detectado)"
            }
        
        # Análise pelos três métodos
        sentimento_vader = analisar_sentimento_vader(mensagem)
        sentimento_textblob = analisar_sentimento_textblob(mensagem)
        sentimento_modelo = analisar_sentimento_modelo_supervisionado(mensagem, modelo_ml)

        # Determinar o sentimento final
        sentimento_final = determinar_sentimento_final(
            [sentimento_vader, sentimento_textblob, sentimento_modelo]
        )
        return {
            "Mensagem": mensagem,
            "Sentimento VADER": sentimento_vader,
            "Sentimento TextBlob": sentimento_textblob,
            "Sentimento Modelo Supervisionado": sentimento_modelo,
            "Sentimento Final": sentimento_final
        }
    except Exception as e:
        st.error(f"Erro ao analisar mensagem: {e}")
        return {"Mensagem": mensagem, "Sentimento Final": "Erro"}

# Inicialização da página Streamlit
def main():
    st.title("💬 Análise de Sentimento com Voto Majoritário")
    st.write("Defina o sentimento de qualquer mensagem utilizando VADER, TextBlob e um modelo supervisionado!")

    # Carregar o modelo supervisionado
    modelo_supervisionado = carregar_modelo_supervisionado(MODEL_PATH)

    if modelo_supervisionado:
        # Entrada de mensagem pelo usuário
        mensagem = st.text_area("Digite sua mensagem:", value="", height=150)
        
        if st.button("Analisar Sentimento"):
            if mensagem.strip():
                resultado = analisar_mensagem(mensagem.strip(), modelo_supervisionado)
                st.subheader("Resultados da Análise:")
                st.write(f"**Mensagem:** {resultado['Mensagem']}")
                st.write(f"**Sentimento Final:** {resultado['Sentimento Final']}")
            else:
                st.warning("Por favor, digite uma mensagem para análise.")

if __name__ == "__main__":
    main()
