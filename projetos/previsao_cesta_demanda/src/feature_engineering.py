# feature_engineering.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys
from datetime import datetime

# =====================================================
# CONFIG
# =====================================================
DEFAULT_DATA = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/data/processed/processed.parquet")
DEFAULT_OUT = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/data/processed/processed_features.parquet")
DEFAULT_LOG = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/logs/feature_engineering.log")

# =====================================================
# FEATURE ENGINEERING FUNCTIONS
# =====================================================
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "order_purchase_timestamp" in df.columns:
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"], errors="coerce")
        df["feat_order_week"] = df["order_purchase_timestamp"].dt.isocalendar().week.astype(int)
        df["feat_is_weekend"] = df["order_purchase_timestamp"].dt.dayofweek.isin([5,6]).astype(int)
        df["feat_quarter"] = df["order_purchase_timestamp"].dt.quarter
        df["feat_is_high_season"] = df["order_purchase_timestamp"].dt.month.isin([11, 12]).astype(int)
        df["feat_is_holiday_month"] = (df["order_purchase_timestamp"].dt.month == 12).astype(int)

    if {"order_delivered_customer_date", "order_purchase_timestamp"}.issubset(df.columns):
        df["feat_days_to_delivery"] = (
            (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
        )

    if {"order_delivered_customer_date", "order_estimated_delivery_date"}.issubset(df.columns):
        df["feat_delivery_delay"] = (
            (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
        )

    if {"order_purchase_timestamp", "review_creation_date"}.issubset(df.columns):
        df["feat_days_to_review"] = (
            (df["review_creation_date"] - df["order_purchase_timestamp"]).dt.days
        )

    return df

def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"price", "product_volume_cm3"}.issubset(df.columns):
        df["feat_price_per_volume"] = np.where(df["product_volume_cm3"] > 0,
                                               df["price"] / df["product_volume_cm3"], np.nan)

    if {"product_category_name_english", "price"}.issubset(df.columns):
        df["feat_category_avg_price"] = df.groupby("product_category_name_english")["price"].transform("mean")
        df["feat_category_price_std"] = df.groupby("product_category_name_english")["price"].transform("std")
        df["feat_price_vs_cat_mean"] = df["price"] / df["feat_category_avg_price"]

    if {"product_category_name_english", "review_score"}.issubset(df.columns):
        df["feat_category_avg_review"] = df.groupby("product_category_name_english")["review_score"].transform("mean")
        df["feat_category_review_std"] = df.groupby("product_category_name_english")["review_score"].transform("std")

    if "product_category_name_english" in df.columns and "order_purchase_timestamp" in df.columns:
        df["feat_category_demand_month"] = (
            df.groupby([
                "product_category_name_english",
                df["order_purchase_timestamp"].dt.to_period("M")
            ])["order_id"].transform("count")
        ).astype(int)

    return df

def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    if "customer_unique_id" in df.columns:
        df["feat_customer_orders_count"] = df.groupby("customer_unique_id")["order_id"].transform("nunique")

    if {"customer_unique_id", "payment_value_sum"}.issubset(df.columns):
        df["feat_customer_avg_ticket"] = df.groupby("customer_unique_id")["payment_value_sum"].transform("mean")

    if {"customer_unique_id", "order_purchase_timestamp"}.issubset(df.columns):
        last_purchase = df.groupby("customer_unique_id")["order_purchase_timestamp"].transform("max")
        df["feat_customer_recency_days"] = (
            (df["order_purchase_timestamp"] - last_purchase).dt.days
        )

    return df

def add_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"freight_value", "price"}.issubset(df.columns):
        df["feat_freight_ratio"] = np.where(df["price"] > 0, df["freight_value"] / df["price"], np.nan)

    if {"freight_value", "payment_value_sum"}.issubset(df.columns):
        df["feat_freight_vs_ticket"] = np.where(df["payment_value_sum"] > 0,
                                                df["freight_value"] / df["payment_value_sum"], np.nan)

    if {"customer_state", "sla_diff_days"}.issubset(df.columns):
        df["feat_state_avg_sla"] = df.groupby("customer_state")["sla_diff_days"].transform("mean")

    return df

# =====================================================
# MAIN PIPELINE
# =====================================================
def run_feature_engineering(data_path: Path, out_path: Path, log_path: Path):
    if not data_path.exists():
        print(f"[ERRO] Arquivo processado não encontrado: {data_path}")
        sys.exit(1)

    print(f"[OK] Carregando dataset: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"[INFO] Linhas: {df.shape[0]:,} | Colunas: {df.shape[1]:,}")

    # Aplicar features
    df = add_time_features(df)
    df = add_product_features(df)
    df = add_customer_features(df)
    df = add_logistics_features(df)

    # Garantir consistência: converter Period → string
    for col in df.select_dtypes(include=["period[M]"]).columns:
        print(f"[INFO] Convertendo Period para string: {col}")
        df[col] = df[col].astype(str)

    # Garantir que não tenha infinitos/NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Salvar parquet
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[OK] Features salvas em: {out_path}")

    # Log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} - Features criadas em {out_path} ({df.shape})\n")

    print(f"[LOG] Registro adicionado em: {log_path}")
    print("\n[INFO] Colunas finais criadas:")
    new_cols = [c for c in df.columns if c.startswith("feat_")]
    print(new_cols[:20], "...")

# =====================================================
# ENTRY POINT
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA, help="Caminho para parquet processado")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Caminho para parquet com features")
    parser.add_argument("--log", default=DEFAULT_LOG, help="Arquivo de log")
    args = parser.parse_args()

    run_feature_engineering(Path(args.data), Path(args.out), Path(args.log))

if __name__ == "__main__":
    main()
