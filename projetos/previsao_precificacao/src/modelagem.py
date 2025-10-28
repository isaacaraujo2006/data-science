import pandas as pd
import numpy as np
import os
import time
import json
import yaml
import logging
import psutil
import warnings
import random
import joblib
import subprocess
import importlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor

# === CONFIGURAÇÕES INICIAIS ===
warnings.filterwarnings("ignore")
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/modelagem_baseline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Verifica se 'tabulate' está instalado
try:
    importlib.import_module("tabulate")
except ImportError:
    subprocess.check_call(["pip", "install", "tabulate==0.9.0"])

start_time = time.time()
logging.info("=== INÍCIO DO PIPELINE DE MODELAGEM BASELINE ===")

try:
    # === Leitura de Config ===
    config_path = r"D:\github\github\data-science\projetos\previsao_precificacao\config\config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_path = config["data"]["processed_parquet"]
    results_dir = os.path.join(os.path.dirname(config_path), "models", "baseline_results")
    os.makedirs(results_dir, exist_ok=True)

    # === Leitura e Amostragem ===
    cols = [
        "dt", "store_id", "first_category_id", "sale_amount",
        "discount", "activity_flag", "avg_temperature", "avg_humidity", "precpt"
    ]
    df = pd.read_parquet(processed_path, columns=cols, engine="pyarrow")
    print(f"Dataset original: {df.shape[0]:,} linhas | {df.shape[1]} colunas")

    df_sample = (
        df.groupby(["store_id", "first_category_id"], group_keys=False)
          .apply(lambda x: x.sample(n=min(len(x), 50), random_state=SEED))
    )
    df_sample = df_sample.sample(n=min(len(df_sample), 250_000), random_state=SEED)
    print(f"Amostra usada: {df_sample.shape[0]:,} linhas")

    # === Engenharia de Features ===
    df_sample["dt"] = pd.to_datetime(df_sample["dt"], errors="coerce")
    df_sample["year"] = df_sample["dt"].dt.year
    df_sample["month"] = df_sample["dt"].dt.month
    df_sample["weekday"] = df_sample["dt"].dt.dayofweek

    for col in ["store_id", "first_category_id"]:
        le = LabelEncoder()
        df_sample[col] = le.fit_transform(df_sample[col])

    target = "sale_amount"
    features = [c for c in df_sample.columns if c not in ["sale_amount", "dt"]]

    X = df_sample[features]
    y = df_sample[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    # === Modelos e Hiperparâmetros ===
    modelos = {
        "LightGBM": {
            "model": LGBMRegressor(
                random_state=SEED,
                n_jobs=-1,
                verbose=-1,
                force_col_wise=True,
                colsample_bytree=None
            ),
            "param_grid": {
                "num_leaves": [31, 63, 127],
                "learning_rate": [0.05, 0.1, 0.2],
                "n_estimators": [200, 500, 800],
                "feature_fraction": [0.8, 0.9, 1.0],
                "min_child_samples": [20, 50, 100]
            }
        },
        "CatBoost": {
            "model": CatBoostRegressor(
                random_seed=SEED,
                verbose=100,
                allow_writing_files=False
            ),
            "param_grid": {
                "depth": [6, 8, 10],
                "learning_rate": [0.05, 0.1, 0.2],
                "iterations": [300, 500],
                "l2_leaf_reg": [3, 5, 7]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=SEED, n_jobs=-1),
            "param_grid": {
                "n_estimators": [200, 400],  # reduz tempo
                "max_depth": [10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2]
            }
        }
    }

    resultados = []
    best_models = {}

    # === Loop de Treino e Avaliação ===
    for nome, cfg in tqdm(modelos.items(), desc="Treinando Modelos", colour="green"):
        print(f"\nTreinando {nome}...")
        logging.info(f"Treinando {nome}...")
        model_start = time.time()

        model = cfg["model"]
        params = cfg["param_grid"]

        if nome == "CatBoost":
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=10,
                scoring="neg_root_mean_squared_error",
                cv=3,
                verbose=2,
                random_state=SEED,
                n_jobs=-1
            )
        elif nome == "RandomForest":
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=10,
                scoring="neg_root_mean_squared_error",
                cv=3,
                verbose=2,
                random_state=SEED,
                n_jobs=1
            )
        else:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=params,
                n_iter=15,
                scoring="neg_root_mean_squared_error",
                cv=3,
                verbose=0,
                random_state=SEED,
                n_jobs=-1
            )

        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_models[nome] = best_model

        # checkpoint imediato
        joblib.dump(best_model, os.path.join(results_dir, f"temp_best_{nome}.pkl"))

        preds = best_model.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / np.clip(y_test, 1e-8, None))) * 100

        duracao = (time.time() - model_start) / 60
        print(f"{nome} concluído em {duracao:.2f} min")

        resultados.append({
            "modelo": nome,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "melhores_parametros": search.best_params_,
            "tempo_min": duracao
        })

        logging.info(f"[{nome}] RMSE={rmse:.4f} | MAE={mae:.4f} | R²={r2:.4f} | MAPE={mape:.2f}% | Tempo={duracao:.2f} min")
        logging.info(f"[{nome}] Melhores parâmetros: {search.best_params_}")

    # === Salvar resultados ===
    resultados_df = pd.DataFrame(resultados).sort_values("rmse")
    resultados_df.to_csv(os.path.join(results_dir, "resultados_baseline.csv"), index=False)
    with open(os.path.join(results_dir, "resultados_baseline.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=4, ensure_ascii=False)

    melhor = resultados_df.iloc[0]
    melhor_nome = melhor["modelo"]
    print("\n=== RESULTADOS FINAIS ===")
    print(resultados_df)
    print(f"\nMelhor modelo: {melhor_nome} (RMSE={melhor['rmse']:.4f})")

    best_model_path = os.path.join(results_dir, f"best_model_{melhor_nome}.pkl")
    joblib.dump(best_models[melhor_nome], best_model_path)
    logging.info(f"Melhor modelo salvo em {best_model_path}")

    plt.figure(figsize=(8, 5))
    plt.bar(resultados_df["modelo"], resultados_df["rmse"])
    plt.title("Comparativo de RMSE por Modelo")
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "comparativo_rmse.png"), dpi=150)
    plt.close()

    from tabulate import tabulate
    report_path = os.path.join(results_dir, "report_modelagem.md")
    with open(report_path, "w", encoding="utf-8") as md:
        md.write("# 🤖 Relatório de Modelagem Baseline\n\n")
        md.write("## Resultados Comparativos\n\n")
        md.write(tabulate(resultados_df, headers='keys', tablefmt='github', showindex=False))
        md.write("\n\n## Melhor Modelo\n")
        md.write(f"- **{melhor_nome}**\n")
        md.write(f"- RMSE: {melhor['rmse']:.4f}\n")
        md.write(f"- MAE: {melhor['mae']:.4f}\n")
        md.write(f"- R²: {melhor['r2']:.4f}\n")
        md.write(f"- MAPE: {melhor['mape']:.2f}%\n\n")
        md.write("## Hiperparâmetros\n")
        md.write("```json\n" + json.dumps(melhor["melhores_parametros"], indent=4) + "\n```\n")
        md.write("\n## 📊 Comparativo Gráfico\n")
        md.write("![comparativo_rmse](comparativo_rmse.png)\n")

    mem_final = psutil.Process().memory_info().rss / 1e9
    elapsed = (time.time() - start_time) / 60
    print(f"\n✅ Pipeline concluído em {elapsed:.2f} min | Memória final: {mem_final:.2f} GB")
    print(f"📄 Relatório salvo em: {report_path}")
    print(f"💾 Modelo salvo em: {best_model_path}")

    logging.info("Pipeline concluído com sucesso.")

except Exception:
    logging.exception("Erro durante a execução do pipeline:")
    raise
