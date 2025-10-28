import os
import logging
import time
import pandas as pd
import yaml
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler
import pickle
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.util import ngrams
from symspellpy.symspellpy import SymSpell
from tqdm import tqdm  # Biblioteca para a barra de progresso

# Configuração do logger
logs_dir = "C:/Github/data-science/projetos/analise_sentimento/logs"
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(logs_dir, "data_processing.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Decorador para medir tempo de execução das funções
def medir_tempo(func):
    def wrapper(*args, **kwargs):
        inicio = time.time()
        print(f"Iniciando: {func.__name__}...")
        resultado = func(*args, **kwargs)
        fim = time.time()
        print(f"Finalizado: {func.__name__} - Tempo de execução: {fim - inicio:.2f} segundos")
        logger.info(f"Tempo de execução - {func.__name__}: {fim - inicio:.2f} segundos")
        return resultado
    return wrapper

# Caminhos e configurações
PROCESSED_DIR = "C:/Github/data-science/projetos/analise_sentimento/data/processed"
SAMPLE_DATA_PATH = "C:/Github/data-science/projetos/analise_sentimento/data/raw/impuro/amostra_asentimentos.parquet"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Inicializar SymSpell
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = "C:/Github/data-science/projetos/analise_sentimento/resources/SymSpell/frequency_dictionary_en_82_765.txt"
bigram_path = "C:/Github/data-science/projetos/analise_sentimento/resources/SymSpell/frequency_bigramdictionary_en_243_342.txt"
sym_spell.load_dictionary(dictionary_path, 0, 1)
sym_spell.load_bigram_dictionary(bigram_path, 0, 2)

# Carregar contrações de um arquivo (para inglês)
contractions = {}
contractions_path = "C:/Github/data-science/projetos/analise_sentimento/resources/contractions.txt"
with open(contractions_path, 'r') as file:
    for line in file:
        contraido, expandido = line.strip().split(',')
        contractions[contraido] = expandido

# Dicionário de gírias em inglês
dicionario_girias = {
    "u": "you",
    "r": "are",
    "btw": "by the way",
    "idk": "I don't know",
    "omg": "oh my god",
    "imo": "in my opinion",
    "fyi": "for your information",
    "brb": "be right back",
    "lol": "laughing out loud",
    "lmao": "laughing my ass off",
    "smh": "shaking my head",
    "yolo": "you only live once"
}

# Funções para pré-processamento de texto
def expandir_contracoes(texto):
    """Expande contrações no texto."""
    for contraido, expandido in contractions.items():
        texto = re.sub(fr"\b{contraido}\b", expandido, texto)
    return texto

def substituir_girias(texto):
    """Substitui gírias no texto."""
    palavras = texto.split()
    palavras_corrigidas = [dicionario_girias.get(palavra.lower(), palavra) for palavra in palavras]
    return " ".join(palavras_corrigidas)

def corrigir_ortografia(texto):
    """Corrige erros ortográficos usando SymSpell."""
    suggestions = sym_spell.lookup_compound(texto, 2)  # Corrige o uso do SymSpell
    if suggestions:
        return suggestions[0].term
    return texto

def remover_emojis(texto):
    """Remove emojis do texto."""
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', texto)

def remover_links_usernames(texto):
    """Remove links e usernames."""
    texto = re.sub(r'http\S+', '', texto)
    texto = re.sub(r'@\w+', '', texto)
    return texto

def preprocessar_pipeline(texto):
    """Pipeline completo de normalização de texto."""
    texto = remover_emojis(texto)
    texto = remover_links_usernames(texto)
    texto = expandir_contracoes(texto)
    texto = substituir_girias(texto)
    texto = corrigir_ortografia(texto)
    return texto

@medir_tempo
def carregar_dataset():
    """Carrega, renomeia colunas, aplica pré-processamento e adiciona colunas para sentimentos."""
    try:
        print(f"Carregando dataset do caminho: {SAMPLE_DATA_PATH}...")
        df = pd.read_parquet(SAMPLE_DATA_PATH)

        print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.")

        # Renomear colunas, se necessário
        colunas_traduzidas = {
            "0": "sentimento",
            "1467810369": "id_usuario",
            "Mon Apr 06 22:19:45 PDT 2009": "data",
            "NO_QUERY": "consulta",
            "_TheSpecialOne_": "usuario",
            "@switchfoot http://twitpic.com/2y1zl - Awww, that's a bummer.  You shoulda got David Carr of Third Day to do it. ;D": "tweet"
        }
        df.rename(columns=colunas_traduzidas, inplace=True)
        print("Colunas renomeadas:", df.columns)

        # Garantir que a coluna `tweet` esteja presente
        if 'tweet' not in df.columns:
            raise KeyError("A coluna 'tweet' não foi encontrada no dataset. Verifique o arquivo.")

        # Inicializar a barra de progresso no Pandas
        tqdm.pandas(desc="Processando")

        # Aplicação do pré-processamento
        print("Iniciando pré-processamento com barra de progresso...")
        df['tweet'] = df['tweet'].progress_apply(preprocessar_pipeline)

        # Adicionar colunas vazias para sentimentos
        print("Adicionando colunas vazias para sentimentos...")
        df['senti_vader'] = None
        df['senti_textblob'] = None
        df['senti_model'] = None
        print("Colunas adicionadas com sucesso!")

        return df
    except Exception as e:
        print(f"Erro ao carregar e processar o dataset: {e}")
        raise

@medir_tempo
def salvar_dataset(df, nome):
    """Salva o dataset normalizado e tratado."""
    caminho_csv = os.path.join(PROCESSED_DIR, f"{nome}.csv")
    caminho_parquet = os.path.join(PROCESSED_DIR, f"{nome}.parquet")
    df.to_csv(caminho_csv, index=False)
    df.to_parquet(caminho_parquet, index=False)
    print(f"Dataset salvo em: {caminho_csv} e {caminho_parquet}")

# Execução do pipeline
if __name__ == "__main__":
    try:
        print("Iniciando pipeline completo...")
        df = carregar_dataset()
        salvar_dataset(df, "dataset_traduzido_e_tratado")
        print("Pipeline completo concluído com sucesso!")
    except Exception as e:
        print(f"Erro no pipeline: {e}")
        logger.error(f"Erro no pipeline: {e}")
