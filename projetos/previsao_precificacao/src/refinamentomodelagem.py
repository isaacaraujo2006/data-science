import os
import time
import json
import yaml
import logging
import joblib
import warnings
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import psutil
import gc

try:
    from optuna.integration import LightGBMPruningCallback
except ImportError:
    from optuna_integration.lightgbm import LightGBMPruningCallback

warnings.filterwarnings("ignore")
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/refinamento_lightgbm.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SEED = 42
np.random.seed(SEED)
start_time = time.time()
logging.info("=== INÍCIO DO REFINAMENTO LIGHTGBM (final sênior) ===")

try:
    # === Leitura de Config ===
    config_path = r"D:\github\github\data-science\projetos\previsao_precificacao\config\config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_path = config["data"]["processed_parquet"]
    results_dir = os.path.join(os.path.dirname(config_path), "models", "refinamento_lightgbm")
    os.makedirs(results_dir, exist_ok=True)

    # === Leitura e pré-processamento ===
    cols = [
        "dt", "store_id", "first_category_id", "sale_amount",
        "discount", "activity_flag", "avg_temperature", "avg_humidity", "precpt"
    ]
    df = pd.read_parquet(processed_path, columns=cols, engine="pyarrow")
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.sort_values(["store_id", "dt"]).reset_index(drop=True)

    if len(df) > 500_000:
        df = df.sort_values("dt").tail(500_000).reset_index(drop=True)
        logging.info("Dataset reduzido para as últimas 500.000 amostras (ordem temporal mantida).")

    # === Engenharia de Features ===
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    df["weekday"] = df["dt"].dt.dayofweek
    df["day"] = df["dt"].dt.day
    df["temp_x_desc"] = df["avg_temperature"] * df["discount"]
    df["humid_x_act"] = df["avg_humidity"] * df["activity_flag"]

    grouped = df.groupby("store_id")["sale_amount"]
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = grouped.shift(lag)
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = grouped.shift(1).rolling(window).mean()

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ["store_id", "first_category_id"]:
        df[col] = df[col].astype("category").cat.codes

    split_point = int(len(df) * 0.9)
    train_df, holdout_df = df.iloc[:split_point], df.iloc[split_point:]
    y_train, y_holdout = train_df["sale_amount"], holdout_df["sale_amount"]
    X_train = train_df.drop(columns=["sale_amount", "dt"])
    X_holdout = holdout_df.drop(columns=["sale_amount", "dt"])

    gc.collect()

    # === Função de Avaliação ===
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
            "random_state": SEED,
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 150),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
        }

        tscv = TimeSeriesSplit(n_splits=3)
        rmses = []
        for train_idx, val_idx in tscv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            dtrain = lgb.Dataset(X_tr, y_tr)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain)

            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                num_boost_round=800,
                early_stopping_rounds=100,
                verbose_eval=False,
                callbacks=[LightGBMPruningCallback(trial, "rmse")],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration)
            rmses.append(mean_squared_error(y_val, preds, squared=False))

        trial.set_user_attr("best_iteration", getattr(model, "best_iteration", 400))
        return np.mean(rmses)

    # === Otimização com Optuna ===
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=SEED, multivariate=True))
    study.optimize(objective, n_trials=30, n_jobs=1, timeout=3600)

    best_params = study.best_params
    best_iter = study.best_trial.user_attrs.get("best_iteration", 400)

    if best_params["learning_rate"] > 0.05:
        best_params["learning_rate"] /= 2
        best_iter *= 2

    print("\nMelhores hiperparâmetros encontrados:")
    print(json.dumps(best_params, indent=4))
    logging.info(f"Melhores hiperparâmetros: {best_params}")

    # === Treino Final ===
    dtrain_final = lgb.Dataset(X_train, y_train)
    best_params.update({
        "objective": "regression",
        "metric": ["rmse", "mae"],
        "verbosity": -1,
        "random_state": SEED,
    })
    model_final = lgb.train(best_params, dtrain_final, num_boost_round=best_iter + 100)

    model_pkl_path = os.path.join(results_dir, "lightgbm_refinado.pkl")
    model_txt_path = os.path.join(results_dir, "lightgbm_refinado.txt")
    joblib.dump(model_final, model_pkl_path)
    model_final.save_model(model_txt_path)
    print(f"\n✅ Modelos salvos em:\n- {model_pkl_path}\n- {model_txt_path}")

    # === Avaliação no Holdout ===
    preds = model_final.predict(X_holdout, num_iteration=model_final.best_iteration)
    rmse = mean_squared_error(y_holdout, preds, squared=False)
    mae = mean_absolute_error(y_holdout, preds)
    r2 = r2_score(y_holdout, preds)
    mape = np.mean(np.abs((y_holdout - preds) / np.maximum(np.abs(y_holdout), 1))) * 100

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}
    print("\n📊 Métricas no Holdout:")
    print(json.dumps(metrics, indent=4))
    logging.info(f"Métricas Holdout: {metrics}")

    with open(os.path.join(results_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    fi_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model_final.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    fi_df.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(fi_df["feature"], fi_df["importance"])
    plt.title("Importância das Features - LightGBM Refinado")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "feature_importance.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.scatter(y_holdout, preds, alpha=0.3)
    plt.xlabel("Valores reais")
    plt.ylabel("Preditos")
    plt.title("Dispersão: Reais vs Preditos")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dispersao_preditos.png"), dpi=150)
    plt.close()

    # === Função de previsão automática ===
    def predict_new_data(model_path, new_data_path):
        model = joblib.load(model_path)
        df_new = pd.read_parquet(new_data_path)
        df_new["dt"] = pd.to_datetime(df_new["dt"], errors="coerce")
        df_new["year"] = df_new["dt"].dt.year
        df_new["month"] = df_new["dt"].dt.month
        df_new["weekday"] = df_new["dt"].dt.dayofweek
        df_new["day"] = df_new["dt"].dt.day
        df_new["temp_x_desc"] = df_new["avg_temperature"] * df_new["discount"]
        df_new["humid_x_act"] = df_new["avg_humidity"] * df_new["activity_flag"]

        grouped = df_new.groupby("store_id")["sale_amount"]
        for lag in [1, 7, 14, 30]:
            df_new[f"lag_{lag}"] = grouped.shift(lag)
        for window in [7, 14, 30]:
            df_new[f"rolling_mean_{window}"] = grouped.shift(1).rolling(window).mean()

        df_new.dropna(inplace=True)
        df_new.reset_index(drop=True, inplace=True)
        for col in ["store_id", "first_category_id"]:
            df_new[col] = df_new[col].astype("category").cat.codes

        X_new = df_new.drop(columns=["sale_amount", "dt"], errors="ignore")
        preds = model.predict(X_new, num_iteration=model.best_iteration)
        df_new["predicted_price"] = preds
        output_path = os.path.join(results_dir, "predicoes_novos_dados.parquet")
        df_new[["dt", "store_id", "predicted_price"]].to_parquet(output_path, index=False)
        print(f"\n📈 Previsões salvas em: {output_path}")
        return df_new[["dt", "store_id", "predicted_price"]]

    elapsed = (time.time() - start_time) / 60
    mem_final = psutil.Process().memory_info().rss / 1e9
    gc.collect()

    print(f"\n⏱️ Tempo total: {elapsed:.2f} min | Memória final: {mem_final:.2f} GB")
    logging.info(f"Refinamento concluído em {elapsed:.2f} min | Modelo salvo em {model_pkl_path}")

except Exception:
    logging.exception("Erro durante o refinamento:")
    raise
logging.info("=== FIM DO REFINAMENTO LIGHTGBM ===")