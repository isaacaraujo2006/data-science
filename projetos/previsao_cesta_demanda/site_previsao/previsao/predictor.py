# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime

# ===============================
# CONFIGURAÇÃO DO MODELO
# ===============================
MODEL_PATH = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/lightgbm_refinado.joblib")
saved = joblib.load(MODEL_PATH)
model = saved["model"]
features = saved["features"]
target_transform = saved.get("target_transform", "log1p")

# ===============================
# LOGGING
# ===============================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "predict.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ===============================
# PRÉ-PROCESSAMENTO
# ===============================
def _recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "product_density" not in df and {"product_weight_g", "product_volume_cm3"} <= set(df.columns):
        df["product_density"] = df["product_weight_g"] / df["product_volume_cm3"]

    for col in ["price", "freight_value", "product_weight_g", "product_volume_cm3", "product_density"]:
        df[f"{col}_log"] = np.log1p(pd.to_numeric(df[col], errors="coerce")).fillna(0)
    return df


# ===============================
# FUNÇÕES DE PREVISÃO
# ===============================
def predict_manual(price, freight, weight, volume):
    """Previsão para um único conjunto de inputs (modo manual)."""
    df = pd.DataFrame([{
        "price": price,
        "freight_value": freight,
        "product_weight_g": weight,
        "product_volume_cm3": volume
    }])
    df = _recompute_features(df)
    X = df.reindex(columns=features, fill_value=0)
    y_pred = model.predict(X)
    y_pred = np.expm1(y_pred) if target_transform == "log1p" else y_pred

    logging.info(f"Previsão manual | price={price}, freight={freight}, weight={weight}, volume={volume} -> {y_pred[0]:.2f}")
    return float(y_pred[0])


def predict_batch(df_input: pd.DataFrame) -> pd.DataFrame:
    """Previsão para datasets (modo automático)."""
    df = _recompute_features(df_input.copy())
    X = df.reindex(columns=features, fill_value=0)
    y_pred = model.predict(X)
    df["demanda_prevista"] = np.expm1(y_pred) if target_transform == "log1p" else y_pred

    logging.info(f"Previsão automática | linhas={len(df_input)} concluída em {datetime.now()}")
    return df
