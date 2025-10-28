import pandas as pd

# caminho de entrada e saída
parquet_path = r"D:\github\data-science\projetos\previsao_cesta_demanda\data\processed\processed.parquet"
csv_path = r"D:\github\data-science\projetos\previsao_cesta_demanda\data\processed\previsao_cesta_demanda.csv"

# leitura e conversão
df = pd.read_parquet(parquet_path)
df.to_csv(csv_path, index=False, encoding='utf-8')

print(f"Arquivo salvo em: {csv_path}")
