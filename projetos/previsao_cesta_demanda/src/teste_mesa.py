# teste_mesa.py
# -*- coding: utf-8 -*-

import pandas as pd
import joblib
from pathlib import Path
import argparse
import shap
import numpy as np

DEFAULT_MODEL = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/lightgbm_refinado.joblib")
DEFAULT_DATA  = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/data/processed/processed_features.parquet")

# =====================================================
# Cenários de teste (editáveis)
# =====================================================
scenarios = [
    {"price": 100,  "freight_value": 20,  "product_weight_g": 800,  "product_volume_cm3": 2000},
    {"price": 250,  "freight_value": 40,  "product_weight_g": 1200, "product_volume_cm3": 3000},
    {"price": 50,   "freight_value": 10,  "product_weight_g": 500,  "product_volume_cm3": 1500},
    {"price": 500,  "freight_value": 70,  "product_weight_g": 2500, "product_volume_cm3": 6000},
    {"price": 80,   "freight_value": 15,  "product_weight_g": 900,  "product_volume_cm3": 2500},
    {"price": 350,  "freight_value": 60,  "product_weight_g": 2000, "product_volume_cm3": 4000},
    {"price": 120,  "freight_value": 25,  "product_weight_g": 1000, "product_volume_cm3": 2200},
    {"price": 700,  "freight_value": 100, "product_weight_g": 4000, "product_volume_cm3": 8000},
    {"price": 40,   "freight_value": 5,   "product_weight_g": 300,  "product_volume_cm3": 1000},
    {"price": 150,  "freight_value": 30,  "product_weight_g": 1100, "product_volume_cm3": 2800},
    {"price": 200,  "freight_value": 35,  "product_weight_g": 1500, "product_volume_cm3": 3500},
    {"price": 90,   "freight_value": 18,  "product_weight_g": 750,  "product_volume_cm3": 1900},
    {"price": 60,   "freight_value": 12,  "product_weight_g": 600,  "product_volume_cm3": 1600},
    {"price": 400,  "freight_value": 80,  "product_weight_g": 3000, "product_volume_cm3": 7000},
    {"price": 1000, "freight_value": 150, "product_weight_g": 5000, "product_volume_cm3": 12000},
]

# =====================================================
# Helpers
# =====================================================
def _apply_dtypes(df, dtypes: dict):
    """Força o dtype conforme mapeamento salvo no modelo."""
    if not dtypes:
        return df
    for col, dtype in dtypes.items():
        if col in df.columns:
            try:
                if "category" in dtype:
                    df[col] = df[col].astype("category")
                else:
                    df[col] = df[col].astype(dtype)
            except Exception:
                pass
    return df


def _inverse_transform(arr, target_transform: str):
    """Inverte a transformação log1p, se usada."""
    return np.expm1(arr) if target_transform == "log1p" else arr


def _recompute_logs(df):
    """Recalcula colunas log e densidade de forma segura."""
    if "product_density" not in df.columns:
        if {"product_weight_g", "product_volume_cm3"} <= set(df.columns):
            w = pd.to_numeric(df["product_weight_g"], errors="coerce")
            v = pd.to_numeric(df["product_volume_cm3"], errors="coerce")
            df["product_density"] = np.where((v > 0) & (~w.isna()), w / v, np.nan)
        else:
            df["product_density"] = np.nan

    for col in ["price", "freight_value", "product_weight_g", "product_volume_cm3", "product_density"]:
        log_col = f"{col}_log"
        if col in df.columns:
            df[log_col] = np.log1p(pd.to_numeric(df[col], errors="coerce")).fillna(0)
        else:
            df[log_col] = 0.0
    return df


def _anchor_rows(df_base, features, how="last", k=1):
    """Retorna linha-base realista a partir do dataset."""
    df_feat = df_base.copy()
    for c in features:
        if c not in df_feat.columns:
            df_feat[c] = 0
    df_feat = df_feat[features].copy()

    if how == "last":
        df_last = df_feat.tail(k)
        if len(df_last) > 0:
            return df_last.reset_index(drop=True)

    med = {}
    for c in df_feat.columns:
        if pd.api.types.is_numeric_dtype(df_feat[c]):
            med[c] = df_feat[c].median()
        else:
            m = df_feat[c].mode(dropna=True)
            med[c] = m.iloc[0] if not m.empty else "unknown"
    return pd.DataFrame([med])


def _build_df_test(df_base, features, dtypes_map, scenarios, anchor="last"):
    """Cria DataFrame de teste baseado em cenários customizados."""
    df_anchor = _anchor_rows(df_base, features, how=anchor, k=1)
    df_anchor = _apply_dtypes(df_anchor, dtypes_map)

    n = len(scenarios)
    df_test = pd.concat([df_anchor]*n, ignore_index=True)
    df_test = df_test.reindex(columns=features, fill_value=0)

    for i, sc in enumerate(scenarios):
        for col, val in sc.items():
            if col in df_test.columns:
                df_test.loc[i, col] = val

    df_test = _recompute_logs(df_test)
    return df_test[features]

# =====================================================
# Predição (modelo completo + residual dinâmico)
# =====================================================
def _align_categories(df, dtypes_map, prep_state):
    """Força dtype e categorias idênticas às do treino."""
    cat_ref = prep_state.get("cat_categories", {}) if prep_state else {}
    for col, dtype in (dtypes_map or {}).items():
        if "category" in dtype and col in df.columns:
            df[col] = df[col].astype("category")
            if col in cat_ref:
                cats = cat_ref[col]
                df[col] = df[col].astype(object)
                df[col] = np.where(df[col].isin(cats), df[col], "unknown")
                df[col] = pd.Categorical(df[col], categories=cats)
    return df


def _predict_models(saved, df_full, df_dyn):
    """Predição robusta compatível com LightGBM."""
    model_full = saved["model"]
    model_dyn  = saved.get("model_dyn")
    alpha      = float(saved.get("best_alpha", 0.0))
    target_tr  = saved.get("target_transform", "log1p")

    prep_state = saved.get("prep_state", {})
    cat_ref = prep_state.get("cat_categories", {})

    # 🧩 alinhar dtypes e categorias
    df_full = _align_categories(df_full, saved.get("dtypes", {}), prep_state)
    df_dyn  = _align_categories(df_dyn,  saved.get("dtypes_dyn", {}), prep_state)

    # 🧱 garantir mesmas colunas e ordem do treino
    expected_full = prep_state.get("safe_columns", df_full.columns.tolist())
    for col in expected_full:
        if col not in df_full.columns:
            df_full[col] = 0
    df_full = df_full[expected_full]

    if model_dyn is not None:
        expected_dyn = [c for c in expected_full if c in saved.get("features_dyn", [])]
        for col in expected_dyn:
            if col not in df_dyn.columns:
                df_dyn[col] = 0
        df_dyn = df_dyn[expected_dyn]

    # 🔒 forçar mesmas colunas categóricas do treino
    cat_cols_full = getattr(model_full, "categorical_feature_", [])
    for c in cat_cols_full:
        if c in df_full.columns:
            df_full[c] = df_full[c].astype("category")

    cat_cols_dyn = getattr(model_dyn, "categorical_feature_", []) if model_dyn is not None else []
    for c in cat_cols_dyn:
        if c in df_dyn.columns:
            df_dyn[c] = df_dyn[c].astype("category")

    # 🔮 predição principal
    yhat_full = model_full.predict(df_full, categorical_feature=cat_cols_full)

    # 🔁 residual dinâmico
    if model_dyn is not None and alpha > 0:
        res_pred = model_dyn.predict(df_dyn, categorical_feature=cat_cols_dyn)
        yhat_dyn = yhat_full + alpha * res_pred
    else:
        yhat_dyn = yhat_full.copy()

    y_full = _inverse_transform(yhat_full, target_tr)
    y_dyn  = _inverse_transform(yhat_dyn,  target_tr)
    return np.clip(y_full, 0, None), np.clip(y_dyn, 0, None)

# =====================================================
# Avaliações
# =====================================================
def eval_price_mode(saved, df_base, anchor):
    """Executa teste de mesa simples com valores fixos."""
    df_full_test = _build_df_test(df_base, saved["features"],     saved.get("dtypes", {}),     scenarios, anchor)
    df_dyn_test  = _build_df_test(df_base, saved["features_dyn"], saved.get("dtypes_dyn", {}), scenarios, anchor)

    y_full, y_dyn = _predict_models(saved, df_full_test, df_dyn_test)

    print("\n" + "="*70)
    print("[TESTE DE MESA - PREVISÃO DIRETA]")
    for i, sc in enumerate(scenarios):
        print(f"Cenário {i+1}: {sc}")
        print(f" → Prev FULL={y_full[i]:.2f} | Prev DYN={y_dyn[i]:.2f}")
        print("-"*70)


def eval_elasticity_mode(saved, df_base, target_kind, anchor):
    """Aplica variação de ±10% no preço e verifica consistência da elasticidade."""
    df_full_0 = _build_df_test(df_base, saved["features"],     saved.get("dtypes", {}),     scenarios, anchor)
    df_dyn_0  = _build_df_test(df_base, saved["features_dyn"], saved.get("dtypes_dyn", {}), scenarios, anchor)

    y_full_0, y_dyn_0 = _predict_models(saved, df_full_0, df_dyn_0)

    def bump(df, factor):
        d = df.copy()
        if "price" in d.columns:
            d["price"] = pd.to_numeric(d["price"], errors="coerce") * factor
        return _recompute_logs(d)

    df_full_up = bump(df_full_0, 1.10); df_dyn_up = bump(df_dyn_0, 1.10)
    df_full_dn = bump(df_full_0, 0.90); df_dyn_dn = bump(df_dyn_0, 0.90)

    y_full_up, y_dyn_up = _predict_models(saved, df_full_up, df_dyn_up)
    y_full_dn, y_dyn_dn = _predict_models(saved, df_full_dn, df_dyn_dn)

    print("\n" + "="*70)
    print(f"[TESTE DE MESA - ELASTICIDADE ({target_kind}) ±10%]")
    for i, sc in enumerate(scenarios):
        b, u, d = y_full_0[i], y_full_up[i], y_full_dn[i]
        ok = ((u >= b) and (b >= d)) if target_kind == "revenue" else ((u <= b) and (b <= d))
        print(f"Cenário {i+1}: {sc}")
        print(f" → FULL: base={b:.2f} | +10%={u:.2f} | -10%={d:.2f} | {'✔' if ok else '✘'}")
        print("-"*70)

# =====================================================
# ENTRY POINT
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data",  default=DEFAULT_DATA)
    parser.add_argument("--mode",  choices=["elasticity","price"], default="elasticity")
    parser.add_argument("--anchor", choices=["last","median"], default="last")
    args = parser.parse_args()

    saved   = joblib.load(Path(args.model))
    df_base = pd.read_parquet(Path(args.data))

    target_name = saved.get("target", "payment_value_sum")
    target_kind = "revenue" if ("value" in target_name or "revenue" in target_name) else "demand"

    if args.mode == "price":
        eval_price_mode(saved, df_base, args.anchor)
    else:
        eval_elasticity_mode(saved, df_base, target_kind, args.anchor)

if __name__ == "__main__":
    main()
