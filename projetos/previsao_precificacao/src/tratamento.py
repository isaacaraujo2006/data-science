import pandas as pd
import numpy as np
import yaml
import os
from scipy import stats
from tqdm import tqdm
import traceback
import time
import psutil

print("=== PIPELINE DE LEITURA E TRATAMENTO (ULTRA OTIMIZADO) ===\n")
start_time = time.time()

try:
    # === Ler config ===
    config_path = r"D:\github\github\data-science\projetos\previsao_precificacao\config\config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    raw_path = config["data"]["raw"]
    processed_path = config["data"]["processed_parquet"]

    # === Ler dataset ===
    print(f"Lendo dataset em:\n{raw_path}\n")
    df = pd.read_parquet(raw_path, engine="pyarrow")
    print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
    print(f"Memória inicial: {psutil.Process().memory_info().rss / 1e9:.2f} GB\n")

    # === Padronizar nomes ===
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    print("Nomes de colunas normalizados.\n")

    # === Conversão de tipos ===
    print("Convertendo tipos de dados...\n")
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

    for col in ["hours_sale", "hours_stock_status"]:
        if col in df.columns:
            mask = df[col].map(type).isin([list, np.ndarray])
            count = mask.sum()
            if count > 0:
                print(f"Coluna '{col}' contém {count:,} registros com listas/arrays — convertendo em blocos...\n")
                indices = np.where(mask)[0]
                chunk_size = max(1, len(indices) // 100)

                for i in tqdm(range(0, len(indices), chunk_size),
                              desc=f"Convertendo {col}",
                              ncols=100, colour='green', unit='bloco'):
                    chunk_idx = indices[i:i+chunk_size]
                    df.iloc[chunk_idx, df.columns.get_loc(col)] = df.iloc[chunk_idx, df.columns.get_loc(col)].astype(str)

    # === Conversão final e otimização de tipos ===
    dtype_map = {
        "city_id": "int32",
        "store_id": "int32",
        "management_group_id": "int32",
        "first_category_id": "int32",
        "second_category_id": "int32",
        "third_category_id": "int32",
        "product_id": "int32",
        "sale_amount": "float32",
        "stock_hour6_22_cnt": "int32",
        "discount": "float32",
        "holiday_flag": "int8",
        "activity_flag": "int8",
        "precpt": "float32",
        "avg_temperature": "float32",
        "avg_humidity": "float32",
        "avg_wind_level": "float32"
    }

    for col, typ in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(typ, errors="ignore")

    df[["hours_sale", "hours_stock_status"]] = df[["hours_sale", "hours_stock_status"]].astype("category", errors="ignore")

    print("\nTipos de dados após conversão:\n", df.dtypes)
    print(f"\nMemória após tratamento: {psutil.Process().memory_info().rss / 1e9:.2f} GB\n")

    # === Valores faltantes ===
    missing = df.isna().mean() * 100
    if missing.any():
        print("Percentual de valores faltantes (%):\n", missing[missing > 0], "\n")
    else:
        print("Nenhum valor faltante encontrado.\n")

    # === Duplicados ===
    dup = df.duplicated().sum()
    print(f"Total de linhas duplicadas: {dup} ({dup/len(df)*100:.4f}%)\n")

    # === Outliers (amostra reduzida) ===
    print("Calculando outliers (amostra reduzida)...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    sample = df[num_cols].sample(n=min(200_000, len(df)), random_state=42)
    z = (sample - sample.mean()) / sample.std()
    pct_outliers = (np.abs(z) > 3).sum() / len(sample) * 100

    print("Percentual de outliers por variável numérica (%):")
    for c, p in zip(num_cols, pct_outliers):
        print(f"{c}: {p:.4f}%")
    print()

    # === Salvar parquet otimizado ===
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df.to_parquet(processed_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Dataset processado salvo em:\n{processed_path}")

    # === Estatísticas finais ===
    elapsed = (time.time() - start_time) / 60
    print(f"\nMemória final: {psutil.Process().memory_info().rss / 1e9:.2f} GB")
    print(f"=== PIPELINE CONCLUÍDO COM SUCESSO EM {elapsed:.2f} min ===")

except Exception:
    print("\nERRO DETECTADO:\n")
    traceback.print_exc()
