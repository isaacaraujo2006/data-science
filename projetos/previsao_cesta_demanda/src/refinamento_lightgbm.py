# refinamento_lightgbm.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pathlib import Path
import argparse, sys, joblib, time, random, json, hashlib, logging
from datetime import datetime

from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import TimeSeriesSplit, HalvingRandomSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import shap

# ==============================
# CONFIG
# ==============================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

DEFAULT_DATA   = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/data/processed/processed_features.parquet")
DEFAULT_OUT    = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/lightgbm_refinado.joblib")
DEFAULT_LOG    = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/logs/refinamento_lightgbm.log")
DEFAULT_PARAMS = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/best_params.json")
DEFAULT_REPORT = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/report.json")

PIPELINE_VERSION = datetime.now().strftime("%Y%m%d_%H%M%S")

# ==============================
# LOGGING CONFIG
# ==============================
DEFAULT_LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=DEFAULT_LOG,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ==============================
# MÉTRICAS
# ==============================
def regression_metrics(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    mdape = np.median(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    p90 = np.percentile(np.abs(y_true - y_pred), 90)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mape),
        "MdAPE": float(mdape),
        "ErrP90": float(p90)
    }

# ==============================
# FEATURIZAÇÃO TEMPORAL
# ==============================
def add_time_features(df, target, time_col="order_purchase_timestamp", id_col=None):
    dfx = df.copy()
    if time_col in dfx.columns:
        dfx = dfx.sort_values(time_col).reset_index(drop=True)
        dfx["month"] = dfx[time_col].dt.month
        dfx["dow"] = dfx[time_col].dt.dayofweek
        if id_col and id_col in dfx.columns:
            grp = dfx.groupby(id_col)
            dfx[f"{target}_lag1"]  = grp[target].shift(1)
            dfx[f"{target}_lag2"]  = grp[target].shift(2)
            dfx[f"{target}_lag4"]  = grp[target].shift(4)
            dfx[f"{target}_lag12"] = grp[target].shift(12)
            dfx[f"{target}_ma3"]   = grp[target].shift(1).rolling(3).mean()
        else:
            dfx[f"{target}_lag1"]  = dfx[target].shift(1)
            dfx[f"{target}_lag2"]  = dfx[target].shift(2)
            dfx[f"{target}_lag4"]  = dfx[target].shift(4)
            dfx[f"{target}_lag12"] = dfx[target].shift(12)
            dfx[f"{target}_ma3"]   = dfx[target].shift(1).rolling(3).mean()
    return dfx

# ==============================
# HPO
# ==============================
def train_lightgbm(X, y, params_grid, scoring, cache_path=None):
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if {"num_leaves", "max_depth"}.issubset(cached.keys()):
                print(f"[INFO] Reutilizando parâmetros de {cache_path}")
                return cached
            else:
                print("[WARN] Cache incompatível. Recalculando...")
        except Exception:
            print("[WARN] Erro ao ler cache. Recalculando...")

    tscv = TimeSeriesSplit(n_splits=5)
    base_model = lgb.LGBMRegressor(random_state=SEED, n_jobs=-1, n_estimators=2000)

    search = HalvingRandomSearchCV(
        base_model,
        param_distributions=params_grid,
        factor=3,
        min_resources=200,
        max_resources=2000,
        scoring=scoring,
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=SEED,
        error_score="raise"
    )
    search.fit(X, y)
    best_params = search.best_params_
    if cache_path:
        cache_path.write_text(json.dumps(best_params))
    return best_params

# ==============================
# MONOTONICIDADE
# ==============================
def build_monotone_constraints(cols, target_kind="revenue"):
    out = []
    for c in cols:
        cl = c.lower()
        if target_kind == "demand" and ("price" in cl or "freight" in cl):
            out.append(-1)
        else:
            out.append(0)
    return out

# ==============================
# PREPROCESS
# ==============================
def fit_preprocess(X_train, target, scale_features=False):
    Xt = X_train.copy()
    drop_cols = [c for c in Xt.columns if c.endswith("_id")]
    datetime_cols = Xt.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    Xt = Xt.drop(columns=drop_cols + datetime_cols, errors="ignore")

    for col in Xt.select_dtypes(include=["object"]).columns:
        Xt[col] = Xt[col].astype(str).astype("category")

    cat_cols = Xt.select_dtypes(include=["category"]).columns.tolist()
    cat_categories = {}
    for c in cat_cols:
        cats = Xt[c].cat.categories
        if "unknown" not in cats:
            cats = cats.append(pd.Index(["unknown"]))
        Xt[c] = Xt[c].cat.set_categories(cats)
        cat_categories[c] = list(cats)

    num_cols = Xt.select_dtypes(include=[np.number]).columns.tolist()
    num_medians = Xt[num_cols].median()
    Xt[num_cols] = Xt[num_cols].fillna(num_medians)

    for c in cat_cols:
        Xt[c] = Xt[c].astype("object").where(Xt[c].notna(), "unknown").astype("category")
        Xt[c] = Xt[c].cat.set_categories(cat_categories[c])

    for col in ["price", "freight_value", "product_weight_g", "product_volume_cm3", "product_density"]:
        if col in Xt.columns and f"{col}_log" not in Xt.columns:
            Xt[f"{col}_log"] = np.log1p(pd.to_numeric(Xt[col], errors="coerce")).fillna(0)

    num_cols_after = Xt.select_dtypes(include=[np.number]).columns.tolist()
    selector = VarianceThreshold(threshold=0.0)
    Xt_num = pd.DataFrame(selector.fit_transform(Xt[num_cols_after]),
                          columns=np.array(num_cols_after)[selector.get_support()],
                          index=Xt.index)
    Xt = pd.concat([Xt_num, Xt.drop(columns=num_cols_after)], axis=1)
    Xt.columns = Xt.columns.str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True).str.strip("_")

    scaler_mean = None
    scaler_scale = None
    scaler_cols = None
    if scale_features:
        scaler_cols = Xt.select_dtypes(include=[np.number]).columns.tolist()
        if len(scaler_cols) > 0:
            scaler = StandardScaler().fit(Xt[scaler_cols])
            scaler_mean = scaler.mean_
            scaler_scale = scaler.scale_
            Xt[scaler_cols] = (Xt[scaler_cols].values - scaler_mean) / scaler_scale

    # validação final
    if Xt.isna().any().any():
        raise ValueError("[ERRO] Há valores NaN após o fit_preprocess. Verifique os dados.")

    state = {
        "drop_cols": drop_cols + datetime_cols,
        "num_medians": num_medians,
        "cat_categories": cat_categories,
        "selector_support": selector.get_support(),
        "selector_num_cols_in": num_cols_after,
        "scale_features": scale_features,
        "scaler_mean": None if scaler_mean is None else scaler_mean.tolist(),
        "scaler_scale": None if scaler_scale is None else scaler_scale.tolist(),
        "scaler_cols": scaler_cols,
        "safe_columns": list(Xt.columns),
    }
    return Xt, state

def transform_preprocess(X, state):
    Xt = X.copy()
    Xt = Xt.drop(columns=state["drop_cols"], errors="ignore")

    for col in Xt.select_dtypes(include=["object"]).columns:
        Xt[col] = Xt[col].astype(str)

    for c, cats in state["cat_categories"].items():
        if c in Xt.columns:
            Xt[c] = pd.Categorical(Xt[c].astype("object").where(Xt[c].notna(), "unknown"), categories=cats)
        else:
            Xt[c] = pd.Categorical(["unknown"] * len(Xt), categories=cats)

    for c, med in state["num_medians"].items():
        if c in Xt.columns:
            Xt[c] = pd.to_numeric(Xt[c], errors="coerce").fillna(med)
        else:
            Xt[c] = med

    for col in ["price", "freight_value", "product_weight_g", "product_volume_cm3", "product_density"]:
        logc = f"{col}_log"
        if col in Xt.columns and logc not in Xt.columns:
            Xt[logc] = np.log1p(pd.to_numeric(Xt[col], errors="coerce")).fillna(0)

    num_in = state["selector_num_cols_in"]
    support = state["selector_support"]
    for c in num_in:
        if c not in Xt.columns:
            Xt[c] = 0
    Xt_num = pd.DataFrame(Xt[num_in].values[:, support],
                          columns=np.array(num_in)[support],
                          index=Xt.index)
    Xt = pd.concat([Xt_num, Xt.drop(columns=[c for c in Xt.columns if c in num_in])], axis=1)
    Xt.columns = Xt.columns.str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True).str.strip("_")

    if state.get("scale_features") and state.get("scaler_cols"):
        cols = [c for c in state["scaler_cols"] if c in Xt.columns]
        if cols:
            mean = np.array(state["scaler_mean"])[[state["scaler_cols"].index(c) for c in cols]]
            scale = np.array(state["scaler_scale"])[[state["scaler_cols"].index(c) for c in cols]]
            Xt[cols] = (Xt[cols].values - mean) / scale

    for c in state["safe_columns"]:
        if c not in Xt.columns:
            Xt[c] = 0
    Xt = Xt[state["safe_columns"]]
    return Xt
# ==============================
# BACKTEST ROLANTE
# ==============================
def rolling_backtest_df(df, target, use_tweedie=False, scale_features=False, splits=5):
    df_bt = add_time_features(df, target=target, time_col="order_purchase_timestamp", id_col=None)
    lag_cols = [c for c in df_bt.columns if c.startswith(f"{target}_lag")]
    max_lag = 0
    if lag_cols:
        try:
            max_lag = max(int(c.rsplit("lag", 1)[1]) for c in lag_cols if "lag" in c)
        except Exception:
            max_lag = 0
    if max_lag > 0:
        df_bt = df_bt.iloc[max_lag:].reset_index(drop=True)

    X_all, y_all = df_bt.drop(columns=[target]), df_bt[target]
    y_used = y_all.copy() if use_tweedie else np.log1p(y_all.copy())

    tscv = TimeSeriesSplit(n_splits=splits)
    maes = []
    for tr, te in tscv.split(X_all):
        Xtr_raw, Xte_raw = X_all.iloc[tr], X_all.iloc[te]
        ytr, yte = y_used.iloc[tr], y_used.iloc[te]

        Xtr_prep, state = fit_preprocess(Xtr_raw, target=target, scale_features=scale_features)
        Xte_prep = transform_preprocess(Xte_raw, state)

        params = {
            "objective": "tweedie" if use_tweedie else "regression",
            "learning_rate": 0.03,
            "n_estimators": 1500,
            "num_leaves": 63,
            "random_state": SEED,
            "n_jobs": -1
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(Xtr_prep, ytr, eval_set=[(Xte_prep, yte)],
                  callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
        preds = model.predict(Xte_prep)
        real = y_all.iloc[te]
        if not use_tweedie:
            preds = np.expm1(preds)
        preds = np.clip(preds, 0, None)
        maes.append(mean_absolute_error(real, preds))
    print(f"[BACKTEST] MAE médio {np.mean(maes):.3f} ± {np.std(maes):.3f}")

# ==============================
# TUNING DO ALPHA (blend residual)
# ==============================
def _inverse(arr, target_transform):
    return np.expm1(arr) if target_transform == "log1p" else arr

def tune_alpha(y_val, yhat_full_val, res_val_pred, target_transform, grid=None, ridge_lambda=0.0):
    if grid is None:
        grid = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0])
    maes = []
    y_val_inv = _inverse(y_val, target_transform)
    for a in grid:
        pred = _inverse(yhat_full_val + a * res_val_pred, target_transform)
        maes.append(mean_absolute_error(y_val_inv, np.clip(pred, 0, None)))
    best_idx = int(np.argmin(maes))
    best_alpha = float(grid[best_idx])
    if np.isnan(best_alpha) or np.isinf(best_alpha):
        num = float(((y_val - yhat_full_val) * res_val_pred).sum())
        den = float((res_val_pred ** 2).sum() + ridge_lambda)
        best_alpha = num / den if den != 0 else 0.0
    return np.clip(best_alpha, 0.0, 2.0), float(maes[best_idx])

# ==============================
# PIPELINE PRINCIPAL
# ==============================
def run_refinamento(data_path: Path, out_path: Path, log_path: Path,
                    n_iter=40, objective="rmse", target="payment_value_sum",
                    early_stopping_rounds=100, scale_features=False,
                    reuse_params_full=False, use_tweedie=False,
                    id_col=None, report_path: Path = DEFAULT_REPORT):

    t0 = time.time()
    if not data_path.exists():
        sys.exit(f"[ERRO] Arquivo não encontrado: {data_path}")

    report_path = Path(report_path).with_suffix(".json")
    df = pd.read_parquet(data_path)
    if "order_purchase_timestamp" in df.columns:
        df = df.sort_values("order_purchase_timestamp").reset_index(drop=True)

    if target not in df.columns:
        sys.exit(f"[ERRO] Target {target} não encontrado!")

    df = add_time_features(df, target=target, time_col="order_purchase_timestamp", id_col=id_col)
    lag_cols = [c for c in df.columns if c.startswith(f"{target}_lag")]
    if lag_cols:
        try:
            max_lag = max(int(c.rsplit("lag", 1)[1]) for c in lag_cols)
            df = df.iloc[max_lag:].reset_index(drop=True)
        except Exception:
            pass

    X_raw, y = df.drop(columns=[target]), df[target]

    if use_tweedie:
        y_used = y.copy()
        objective_lgbm = "tweedie"
        target_transform = "identity"
        scoring = "neg_mean_absolute_error"
    else:
        y_used = np.log1p(y.copy())
        objective_lgbm = "regression"
        target_transform = "log1p"
        scoring = "neg_mean_absolute_error"

    cutoff = int(len(df) * 0.8)
    cutoff_date = df["order_purchase_timestamp"].iloc[cutoff] if "order_purchase_timestamp" in df.columns else "N/A"
    X_train_raw, X_test_raw = X_raw.iloc[:cutoff].copy(), X_raw.iloc[cutoff:].copy()
    y_train, y_test = y_used.iloc[:cutoff].copy(), y_used.iloc[cutoff:].copy()

    X_train, prep_state = fit_preprocess(X_train_raw, target=target, scale_features=scale_features)
    X_test = transform_preprocess(X_test_raw, prep_state)

    # Features dinâmicas
    ban_patterns = ("customer", "_id")
    keep_prefixes = ("month", "dow", "price", "freight", "weight", "volume", "density", f"{target}_lag", f"{target}_ma")
    features_dyn = [c for c in X_train.columns
                    if not any(p in c.lower() for p in ban_patterns)
                    and (c.startswith(keep_prefixes) or c.endswith("_log"))]
    if not features_dyn:
        features_dyn = list(X_train.columns)

    print(f"[INFO] Features dinâmicas ({len(features_dyn)}): {features_dyn[:10]}{' ...' if len(features_dyn) > 10 else ''}")

    base_grid_common = {
        "learning_rate": [0.01, 0.02],
        "n_estimators": [3000, 5000],
        "num_leaves": [31, 63],
        "max_depth": [-1, 10],
        "feature_fraction": [0.6, 0.75, 0.85],
        "bagging_fraction": [0.6, 0.75, 0.9],
        "bagging_freq": [1, 5],
        "min_child_samples": [128, 256],
        "min_data_in_leaf": [256, 512],
        "reg_alpha": [0.0, 0.1, 0.3],
        "reg_lambda": [0.1, 0.5, 1.0],
        "min_gain_to_split": [0.0, 0.02],
        "max_bin": [255, 511],
        "extra_trees": [True, False],
        "feature_fraction_bynode": [0.6, 0.8],
        "deterministic": [True],
        "force_col_wise": [True],
    }

    param_grid = {"objective": ["tweedie"], "tweedie_variance_power": [1.3, 1.5, 1.7], **base_grid_common} if use_tweedie else {"objective": ["regression"], **base_grid_common}

    sig_obj = {
        "target": target,
        "use_tweedie": use_tweedie,
        "scale_features": scale_features,
        "features": sorted(list(X_train.columns)),
        "pipeline_version": PIPELINE_VERSION[:8],
    }
    sig = hashlib.md5(json.dumps(sig_obj, sort_keys=True).encode()).hexdigest()[:10]
    cache_file = DEFAULT_PARAMS.with_name(f"best_params_{sig}.json")

    cut_val = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:cut_val], X_train.iloc[cut_val:]
    y_tr, y_val = y_train.iloc[:cut_val], y_train.iloc[cut_val:]
    cat_cols_full = X_tr.select_dtypes(include=["category"]).columns.tolist()

    best_params_full = train_lightgbm(X_train, y_train, param_grid, scoring=scoring, cache_path=cache_file)
    esr_full = min(early_stopping_rounds, int(best_params_full.get("n_estimators", 3000) * 0.5))

    tgt_kind = "revenue" if ("value" in target or "revenue" in target) else "demand"
    mono_full = build_monotone_constraints(list(X_train.columns), target_kind=tgt_kind)

    model_full = lgb.LGBMRegressor(**best_params_full, random_state=SEED, n_jobs=-1, monotone_constraints=mono_full)
    model_full.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], categorical_feature=cat_cols_full,
                   callbacks=[lgb.early_stopping(esr_full), lgb.log_evaluation(100)])

    yhat_full_tr = model_full.predict(X_tr)
    yhat_full_val = model_full.predict(X_val)
    yhat_full_test = model_full.predict(X_test)

    preds_full_eval = _inverse(yhat_full_test, target_transform)
    y_eval_full = y.iloc[cutoff:]
    preds_full_eval = np.clip(preds_full_eval, 0, None)
    metrics_full = regression_metrics(y_eval_full, preds_full_eval)

    # Residual
    res_tr, res_val = y_tr - yhat_full_tr, y_val - yhat_full_val
    if float(np.var(res_tr)) < 1e-10:
        best_alpha, model_dyn, metrics_dyn = 0.0, None, metrics_full.copy()
        res_val_pred, res_var, corr_res_err = np.zeros_like(res_val), 0.0, 0.0
        print("[INFO] Residual desativado: variância quase nula.")
    else:
        base_grid_dyn = {**base_grid_common, "objective": [best_params_full.get("objective", objective_lgbm)]}
        best_params_dyn = train_lightgbm(X_tr[features_dyn], res_tr, base_grid_dyn, scoring="neg_mean_absolute_error", cache_path=None)
        esr_dyn = min(early_stopping_rounds, int(best_params_dyn.get("n_estimators", 3000) * 0.5))
        mono_dyn = build_monotone_constraints(features_dyn, target_kind=tgt_kind)
        cat_cols_dyn = [c for c in X_tr[features_dyn].select_dtypes(include=["category"]).columns.tolist()]
        model_dyn = lgb.LGBMRegressor(**best_params_dyn, random_state=SEED, n_jobs=-1, monotone_constraints=mono_dyn)
        model_dyn.fit(X_tr[features_dyn], res_tr, eval_set=[(X_val[features_dyn], res_val)],
                      categorical_feature=cat_cols_dyn, callbacks=[lgb.early_stopping(esr_dyn), lgb.log_evaluation(100)])
        res_val_pred = model_dyn.predict(X_val[features_dyn])
        err_val_full = y_val - yhat_full_val
        res_var = float(np.var(res_val_pred))
        corr_res_err = float(np.corrcoef(res_val_pred, err_val_full)[0, 1]) if np.std(res_val_pred) > 0 and np.std(err_val_full) > 0 else 0.0
        best_alpha, best_mae_val = tune_alpha(y_val, yhat_full_val, res_val_pred, target_transform)
        full_val_mae = mean_absolute_error(_inverse(y_val, target_transform), np.clip(_inverse(yhat_full_val, target_transform), 0, None))
        if (full_val_mae - best_mae_val) < 0.01:
            best_alpha, model_dyn = 0.0, None
        preds_dyn_eval = preds_full_eval.copy() if model_dyn is None else np.clip(_inverse(yhat_full_test + best_alpha * model_dyn.predict(X_test[features_dyn]), target_transform), 0, None)
        metrics_dyn = regression_metrics(y.iloc[cutoff:], preds_dyn_eval)

    try:
        rolling_backtest_df(df, target=target, use_tweedie=use_tweedie, scale_features=scale_features, splits=5)
    except Exception as e:
        print(f"[WARN] Backtest falhou: {e}")

    drift_mean = float(np.mean(np.abs(preds_full_eval - preds_dyn_eval)))
    corr = float(np.corrcoef(preds_full_eval, preds_dyn_eval)[0, 1])

    print(f"[RESULTADOS - Modelo Completo] {json.dumps(metrics_full, ensure_ascii=False)}")
    print(f"[RESULTADOS - Modelo Dinâmico] {json.dumps(metrics_dyn, ensure_ascii=False)}")
    print(f"[INFO] Drift médio = {drift_mean:.4f}, Correlação = {corr:.4f}")

    # Hash do dataset
    try:
        data_hash = hashlib.md5(pd.util.hash_pandas_object(df, index=False).values).hexdigest()
    except Exception:
        data_hash = "unknown"

    # SHAP
    try:
        explainer_full = shap.TreeExplainer(model_full)
        shap_vals_full = explainer_full.shap_values(X_test)
        if isinstance(shap_vals_full, list): shap_vals_full = shap_vals_full[0]
        top_shap_full = pd.Series(np.abs(shap_vals_full).mean(axis=0), index=X_test.columns).nlargest(20)
    except Exception:
        top_shap_full = pd.Series(dtype=float)

    try:
        if model_dyn is not None:
            explainer_dyn = shap.TreeExplainer(model_dyn)
            shap_vals_dyn = explainer_dyn.shap_values(X_test[features_dyn])
            if isinstance(shap_vals_dyn, list): shap_vals_dyn = shap_vals_dyn[0]
            top_shap_dyn = pd.Series(np.abs(shap_vals_dyn).mean(axis=0), index=X_test[features_dyn].columns).nlargest(20)
        else:
            top_shap_dyn = pd.Series(dtype=float)
    except Exception:
        top_shap_dyn = pd.Series(dtype=float)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "pipeline_version": PIPELINE_VERSION,
        "model": model_full,
        "model_dyn": model_dyn,
        "features": list(X_train.columns),
        "features_dyn": features_dyn,
        "target": target,
        "metrics_full": metrics_full,
        "metrics_dyn": metrics_dyn,
        "params_full": best_params_full,
        "params_dyn": None if model_dyn is None else best_params_dyn,
        "drift_mean": drift_mean,
        "drift_corr": corr,
        "best_alpha": float(best_alpha),
        # 🔥 adições essenciais:
        "prep_state": prep_state,
        "dtypes": X_train.dtypes.apply(lambda x: str(x)).to_dict(),
        "dtypes_dyn": X_train[features_dyn].dtypes.apply(lambda x: str(x)).to_dict(),
    }, out_path)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "target": target,
        "cutoff_date": str(cutoff_date),
        "metrics_full": metrics_full,
        "metrics_dyn": metrics_dyn,
        "data_hash": data_hash,
        "drift": {"mean_abs_diff": drift_mean, "corr": corr},
        "top_shap_full": top_shap_full.to_dict(),
        "top_shap_dyn": top_shap_dyn.to_dict(),
        "best_alpha": float(best_alpha)
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    elapsed = round((time.time() - t0) / 60, 2)
    logging.info(f"Treinamento finalizado. MAE={metrics_full['MAE']:.3f}, R2={metrics_full['R2']:.5f}, Alpha={best_alpha:.3f}")
    logging.info(f"Modelos salvos em {out_path}")
    logging.info(f"Report salvo em {report_path}")

    print(f"[OK] Treinamento finalizado em {elapsed} min")
    print(f"[INFO] Modelos salvos em: {out_path}")
    print(f"[INFO] Report salvo em: {report_path}")

# ==============================
# ENTRY POINT
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--n_iter", type=int, default=40)
    parser.add_argument("--objective", default="rmse")
    parser.add_argument("--target", default="payment_value_sum")
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    parser.add_argument("--scale_features", action="store_true")
    parser.add_argument("--reuse_params_full", action="store_true")
    parser.add_argument("--use_tweedie", action="store_true")
    parser.add_argument("--id_col", default=None)
    args = parser.parse_args()

    try:
        run_refinamento(
            Path(args.data),
            Path(args.out),
            Path(args.log),
            n_iter=args.n_iter,
            objective=args.objective,
            target=args.target,
            early_stopping_rounds=args.early_stopping_rounds,
            scale_features=args.scale_features,
            reuse_params_full=args.reuse_params_full,
            use_tweedie=args.use_tweedie,
            id_col=args.id_col,
        )
    except Exception as e:
        logging.exception(f"Falha no refinamento: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

