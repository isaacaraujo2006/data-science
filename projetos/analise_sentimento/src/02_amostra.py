import pandas as pd
import os

# Caminho para o dataset raw
RAW_DATA_PATH = "C:/Github/data-science/projetos/analise_sentimento/data/raw/impuro/asentimentos.parquet"

# Caminho para salvar a amostra
SAMPLE_OUTPUT_PATH = "C:/Github/data-science/projetos/analise_sentimento/data/raw/impuro/amostra_asentimentos.parquet"

def gerar_amostra(dataset_path, amostra_percentual, output_path):
    """
    Carrega um dataset, gera uma amostra percentual e salva a amostra em um arquivo.
    
    Args:
        dataset_path (str): Caminho para o dataset original.
        amostra_percentual (float): Porcentagem da amostra (entre 0 e 1).
        output_path (str): Caminho para salvar a amostra.
    """
    try:
        # Carregando o dataset
        print(f"Carregando dataset do caminho: {dataset_path}...")
        if dataset_path.endswith('.parquet'):
            df = pd.read_parquet(dataset_path)
        elif dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path, encoding='latin1', on_bad_lines='skip')
        else:
            raise ValueError("Formato de arquivo não suportado. Use .csv ou .parquet.")
        
        print(f"Dataset carregado: {df.shape[0]} linhas e {df.shape[1]} colunas.")

        # Gerando a amostra
        print(f"Gerando uma amostra de {amostra_percentual * 100:.0f}% do dataset...")
        amostra_df = df.sample(frac=amostra_percentual, random_state=42)
        print(f"Amostra gerada: {amostra_df.shape[0]} linhas.")

        # Salvando a amostra
        print(f"Salvando a amostra no caminho: {output_path}...")
        if output_path.endswith('.parquet'):
            amostra_df.to_parquet(output_path, index=False)
        elif output_path.endswith('.csv'):
            amostra_df.to_csv(output_path, index=False)
        else:
            raise ValueError("Formato de arquivo não suportado para salvar. Use .csv ou .parquet.")
        
        print("Amostra salva com sucesso!")
    except Exception as e:
        print(f"Erro ao gerar a amostra: {e}")

# Executando o programa com 20% de amostra
if __name__ == "__main__":
    gerar_amostra(RAW_DATA_PATH, 0.2, SAMPLE_OUTPUT_PATH)  # Alterado de 0.4 para 0.2