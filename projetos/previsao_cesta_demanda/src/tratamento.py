# tratamento.py
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from ydata_profiling import ProfileReport
from sklearn.preprocessing import RobustScaler
from scipy.stats import ks_2samp
import joblib

DEFAULT_CONFIG = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/config/config.yaml")
DEFAULT_SCALERS = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/scalers.joblib")

# =====================================================
# CONFIG & HELPERS
# =====================================================
def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config não encontrado: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
    )
    return df

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    date_cols = [c for c in df.columns if "date" in c or "timestamp" in c]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors="coerce")

    for c in ["price", "freight_value", "payment_value_sum"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    measure_cols = [
        "product_weight_g", "product_length_cm", "product_height_cm", "product_width_cm",
        "product_name_length", "product_description_length", "product_photos_qty"
    ]
    for c in measure_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    id_cols = [c for c in df.columns if "id" in c]
    for c in id_cols:
        df[c] = df[c].astype("string", errors="ignore")

    return df

# =====================================================
# DATA QUALITY REPORT
# =====================================================
def data_quality_report(df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    results = []
    dup_count = df.duplicated().sum()
    results.append({"column": "ALL", "check": "duplicados",
                    "count": dup_count,
                    "percent": round((dup_count / total_rows) * 100, 2)})

    for col in df.columns:
        col_data = df[col]
        missing = col_data.isna().sum()
        if missing > 0:
            results.append({"column": col, "check": "faltantes",
                            "count": int(missing),
                            "percent": round((missing / total_rows) * 100, 2)})

        if pd.api.types.is_numeric_dtype(col_data):
            negatives = (col_data < 0).sum()
            if negatives > 0:
                results.append({"column": col, "check": "valores_negativos",
                                "count": int(negatives),
                                "percent": round((negatives / total_rows) * 100, 2)})

            q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outliers = ((col_data < lower) | (col_data > upper)).sum()
            if outliers > 0:
                results.append({"column": col, "check": "outliers_iqr",
                                "count": int(outliers),
                                "percent": round((outliers / total_rows) * 100, 2)})
    return pd.DataFrame(results)

# =====================================================
# CLEANING RULES
# =====================================================
def cap_with_business_rules(df: pd.DataFrame) -> pd.DataFrame:
    rules = {
        "price": (1, 20000),
        "freight_value": (0, 5000),
        "product_weight_g": (10, 60000),
        "product_length_cm": (1, 200),
        "product_height_cm": (1, 200),
        "product_width_cm": (1, 200),
    }
    for col, (low, high) in rules.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Winsorization (1%–99%) dentro dos limites de negócio
            q_low, q_high = df[col].quantile(0.01), df[col].quantile(0.99)
            df[col] = df[col].clip(lower=max(low, q_low), upper=min(high, q_high))
    return df

def treat_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col]
        df[col] = series.mask(series < 0, np.nan)

        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = np.where(series > upper, upper,
                           np.where(series < lower, lower, series))

        if df[col].isna().sum() > 0:
            if "product_category_name" in df.columns:
                df[col] = df.groupby("product_category_name")[col].transform(
                    lambda x: x.fillna(x.median())
                )
            df[col] = df[col].fillna(df[col].median())
    return df

def treat_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in cat_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna("unknown")
    return df

# =====================================================
# DRIFT MONITORING
# =====================================================
def psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index"""
    quantiles = np.linspace(0, 1, buckets + 1)
    expected_perc = np.histogram(expected.dropna(), bins=np.quantile(expected.dropna(), quantiles))[0] / len(expected.dropna())
    actual_perc = np.histogram(actual.dropna(), bins=np.quantile(expected.dropna(), quantiles))[0] / len(actual.dropna())
    psi_value = np.sum((expected_perc - actual_perc) * np.log((expected_perc + 1e-9) / (actual_perc + 1e-9)))
    return psi_value

def drift_report(df_raw: pd.DataFrame, df_clean: pd.DataFrame, output_file: Path):
    changes = []
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df_raw.columns:
            try:
                raw = pd.to_numeric(df_raw[col], errors="coerce").dropna()
                clean = pd.to_numeric(df_clean[col], errors="coerce").dropna()
                if len(raw) > 0 and len(clean) > 0:
                    mean_raw, mean_clean = raw.mean(), clean.mean()
                    drift_mean = 100 * (mean_clean - mean_raw) / (mean_raw + 1e-9)
                    ks_stat, ks_p = ks_2samp(raw, clean)
                    psi_val = psi(raw, clean)
                    changes.append({
                        "column": col,
                        "mean_drift_percent": round(drift_mean, 2),
                        "ks_stat": round(ks_stat, 4),
                        "ks_pvalue": round(ks_p, 4),
                        "psi": round(psi_val, 4)
                    })
            except Exception as e:
                print(f"[WARN] Drift não calculado para {col}: {e}")

    pd.DataFrame(changes).to_csv(output_file, index=False, encoding="utf-8")
    print(f"[OK] Drift report salvo em: {output_file}")

# =====================================================
# MAIN PIPELINE
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Caminho para config.yaml (padrão: {DEFAULT_CONFIG})")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    raw_parquet = Path(cfg["data"]["raw"])
    out_parquet_clean = Path(cfg["data"]["processed_parquet"])
    out_parquet_rawcheck = out_parquet_clean.parent / "processed_rawcheck.parquet"
    logs_file = Path(cfg["logs"]["file"])
    quality_file = Path(cfg["reports"]["figures"]).parent / "data_quality_report.csv"
    drift_file = Path(cfg["reports"]["figures"]).parent / "drift_report.csv"
    profile_html = Path(cfg["reports"]["figures"]).parent / "data_profile.html"

    if not raw_parquet.exists():
        print(f"[ERRO] Parquet bruto não encontrado: {raw_parquet}")
        sys.exit(1)

    df = pd.read_parquet(raw_parquet)
    total_rows = len(df)
    print(f"[OK] Dataset carregado: {raw_parquet}")
    print(f"Linhas: {df.shape[0]:,} | Colunas: {df.shape[1]:,}")

    df = normalize_columns(df)

    rename_map = {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length"
    }
    df = df.rename(columns=rename_map)

    df = fix_dtypes(df)

    quality_df = data_quality_report(df, total_rows)
    quality_df.to_csv(quality_file, index=False, encoding="utf-8")
    print(f"[OK] Relatório de qualidade salvo em: {quality_file}")

    out_parquet_rawcheck.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet_rawcheck, index=False)
    print(f"[OK] Versão rawcheck salva em: {out_parquet_rawcheck}")

    # Cleaning
    df_clean = cap_with_business_rules(df.copy())
    df_clean = treat_numeric_columns(df_clean)
    df_clean = treat_categorical_columns(df_clean)

    # Features derivadas
    dims = ["product_length_cm", "product_height_cm", "product_width_cm"]
    if set(dims).issubset(df_clean.columns):
        df_clean["product_volume_cm3"] = (
            df_clean["product_length_cm"] * df_clean["product_height_cm"] * df_clean["product_width_cm"]
        )
    if {"product_weight_g", "product_volume_cm3"}.issubset(df_clean.columns):
        df_clean["product_density"] = df_clean["product_weight_g"] / df_clean["product_volume_cm3"]

    if "order_purchase_timestamp" in df_clean.columns:
        df_clean["order_year"] = df_clean["order_purchase_timestamp"].dt.year
        df_clean["order_month"] = df_clean["order_purchase_timestamp"].dt.month
        df_clean["order_dow"] = df_clean["order_purchase_timestamp"].dt.dayofweek

    # Transformações log
    for col in ["price", "freight_value", "product_weight_g", "product_volume_cm3", "product_density", "payment_value_sum"]:
        if col in df_clean.columns:
            df_clean[col + "_log"] = np.log1p(df_clean[col])

    # Escalonamento robusto (um único dicionário de scalers salvo)
    scalers = {}
    for col in ["price", "freight_value", "payment_value_sum"]:
        if col in df_clean.columns:
            scaler = RobustScaler()
            df_clean[col + "_scaled"] = scaler.fit_transform(df_clean[[col]])
            scalers[col] = scaler
    DEFAULT_SCALERS.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, DEFAULT_SCALERS)
    print(f"[OK] Scalers salvos em: {DEFAULT_SCALERS}")

    # Separar features dependentes do cliente
    customer_features = [c for c in df_clean.columns if "customer" in c or "feat_customer" in c]
    df_clean.attrs["customer_features"] = customer_features
    print(f"[INFO] Features dependentes do cliente identificadas: {customer_features}")

    # Exporta clean
    out_parquet_clean.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(out_parquet_clean, index=False)
    print(f"[OK] Versão clean salva em: {out_parquet_clean}")

    drift_report(df, df_clean, drift_file)

    profile = ProfileReport(df_clean, title="Data Profiling Report", explorative=True, minimal=True, samples=None)
    profile.to_file(profile_html)
    print(f"[OK] Profiling report salvo em: {profile_html}")

    logs_file.parent.mkdir(parents=True, exist_ok=True)
    with logs_file.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} - Processado {raw_parquet} -> {out_parquet_clean} ({df_clean.shape})\n")

    print(f"[LOG] Registro adicionado em: {logs_file}")
    print("\nTipos de dados finais:")
    print(df_clean.dtypes)

if __name__ == "__main__":
    main()
