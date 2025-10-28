import os
import pandas as pd
from tqdm import tqdm
from textblob import TextBlob

# Configuração do logger
logs_dir = "C:/Github/data-science/projetos/analise_sentimento/logs"
os.makedirs(logs_dir, exist_ok=True)

# Caminhos e configurações
PROCESSED_DATA_PATH = "C:/Github/data-science/projetos/analise_sentimento/data/processed/dataset_traduzido_e_tratado.parquet"
OUTPUT_DIR = "C:/Github/data-science/projetos/analise_sentimento/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função para análise de sentimento com TextBlob
def analisar_sentimento_textblob(texto):
    """Utiliza TextBlob para análise de sentimento."""
    if not texto or pd.isna(texto):  # Lida com textos vazios ou nulos
        return 'Neutro'
    
    # Obtém polaridade com TextBlob
    polarity = TextBlob(texto).sentiment.polarity
    
    # Classificação baseada na polaridade
    if polarity > 0.1:  # Ajusta limiar para ser mais preciso
        return 'Positivo'
    elif polarity < -0.1:  # Ajusta limiar para definir negativo
        return 'Negativo'
    else:
        return 'Neutro'

# Pipeline principal para TextBlob
def pipeline_textblob():
    """Pipeline refinado para análise de sentimentos com TextBlob."""
    try:
        # Carregar o dataset processado
        print(f"Carregando dataset do caminho: {PROCESSED_DATA_PATH}...")
        df = pd.read_parquet(PROCESSED_DATA_PATH)
        print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.")

        # Configurar tqdm para barra de progresso
        tqdm.pandas(desc="Analisando Sentimentos")

        # Garantir o nome correto da coluna
        if 'texto_tratado' not in df.columns:
            raise KeyError("A coluna 'texto_tratado' não foi encontrada no dataset processado.")
        
        # Aplicação da análise de sentimento
        print("Iniciando análise de sentimento com TextBlob...")
        df['sentimento_textblob'] = df['texto_tratado'].progress_apply(analisar_sentimento_textblob)
        print("Análise de sentimento concluída.")

        # Salvando os resultados
        output_path_csv = os.path.join(OUTPUT_DIR, "dataset_textblob_analise.csv")
        output_path_parquet = os.path.join(OUTPUT_DIR, "dataset_textblob_analise.parquet")
        df.to_csv(output_path_csv, index=False)
        df.to_parquet(output_path_parquet, index=False)
        print(f"Dataset com análises refinadas salvo em: {output_path_csv} e {output_path_parquet}")
        
        # Retornar DataFrame para testes adicionais
        return df
    except Exception as e:
        print(f"Erro no pipeline TextBlob refinado: {e}")

# Teste de mesa para avaliar TextBlob
def teste_mesa():
    """Teste de mesa refinado para validação."""
    # Mensagens do teste de mesa
    testes_de_mesa = [
        "I absolutely love this!",  # Positivo
        "This is the worst thing ever.",  # Negativo
        "It's okay, not amazing but not bad.",  # Neutro
        "Terrible experience, I hated it.",  # Negativo
        "Superb quality, exceeded my expectations!",  # Positivo
        "Meh, it's fine.",  # Neutro
        "Amazing product! Will buy again.",  # Positivo
        "Poor construction and very disappointing.",  # Negativo
        "Not too bad, but not great either.",  # Neutro
        "Absolutely wonderful, I loved it!",  # Positivo
    ]

    # Resultados esperados
    resultados_esperados = [
        "Positivo", "Negativo", "Neutro", "Negativo", "Positivo",
        "Neutro", "Positivo", "Negativo", "Neutro", "Positivo"
    ]

    # Análise com TextBlob
    print("\nIniciando teste de mesa refinado com TextBlob...")
    resultados_textblob = [analisar_sentimento_textblob(texto) for texto in testes_de_mesa]

    # Comparação dos resultados
    acertos = sum([1 for esperado, textblob in zip(resultados_esperados, resultados_textblob) if esperado == textblob])
    total_testes = len(resultados_esperados)

    # Exibir resultados do teste de mesa refinado
    print("\nResultados do Teste de Mesa Refinado com TextBlob:")
    for i, (texto, esperado, textblob) in enumerate(zip(testes_de_mesa, resultados_esperados, resultados_textblob), 1):
        print(f"{i:02d}. {texto} --> Esperado: {esperado}, TextBlob: {textblob}")
    print(f"\nTotal de acertos no teste de mesa: {acertos}/{total_testes} ({(acertos / total_testes) * 100:.2f}%)")

# Executar pipeline e teste refinado
if __name__ == "__main__":
    # Pipeline refinado TextBlob
    pipeline_textblob()

    # Teste de mesa refinado
    teste_mesa()
