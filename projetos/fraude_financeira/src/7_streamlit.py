# 7_streamlit.py — apenas PREVISÃO AUTOMÁTICA (Gold Luxury)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ==============================
# CONFIGURAÇÃO DE PÁGINA
# ==============================
st.set_page_config(page_title="Fraude Financeira - Isaac Araújo", layout="centered")

# ==============================
# CSS - Gold Luxury (inline)
# ==============================
GOLD_LUXURY_CSS = """
<style>
body, .stApp { background-color: #1a1a1a; color: #FFD700; font-family: 'Arial', sans-serif; }
h1, h2, h3 { color: #FFD700; text-align: center; }
.stButton>button { background-color: #FFD700; color: #000; border-radius: 10px; font-weight: 700; border: none; }
.stButton>button:hover { background-color: #000; color: #FFD700; border: 1px solid #FFD700; }
.stDataFrame, .stTable { border: 1px solid #FFD700; }
.block-container { padding-top: 2rem; }
hr { border-top: 1px solid #FFD70066; }
.footer { text-align: center; color: #FFD700; margin-top: 1rem; opacity: .9; }
.sidebar .sidebar-content { background: #141414; }
</style>
"""
st.markdown(GOLD_LUXURY_CSS, unsafe_allow_html=True)

# ==============================
# CAMINHOS DOS ARTEFATOS
# ==============================
MODEL_PATH     = r"D:\github\data-science\projetos\fraude_financeira\models\lightgbm_calibrated.joblib"
THRESHOLD_PATH = r"D:\github\data-science\projetos\fraude_financeira\models\optimal_threshold.txt"
FEATURES_PATH  = r"D:\github\data-science\projetos\fraude_financeira\models\features_usadas.txt"

# ==============================
# LOADERS (cache)
# ==============================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_thresholds():
    with open(THRESHOLD_PATH, "r") as f:
        raw = f.read().strip()
        if raw.startswith("{"):
            return json.loads(raw)        # {"custo": ..., "f1": ...}
        return {"custo": float(raw)}      # número único

@st.cache_resource
def load_features():
    with open(FEATURES_PATH, "r", encoding="utf-8") as f:
        feats = [ln.strip() for ln in f if ln.strip()]
    return feats

# ==============================
# TÍTULO / HEADER
# ==============================
st.title("💰 Sistema de Detecção de Fraude Financeira")
st.markdown("**Autor:** Isaac Araújo | **Modelo:** LightGBM (calibrado)")

# ==============================
# CARREGAR MODELO / THRESHOLDS / FEATURES
# ==============================
model = load_model()
thresholds = load_thresholds()
features = load_features()

# Sidebar: seleção de limiar
st.sidebar.header("⚙️ Configurações")
chosen_thr_key = st.sidebar.selectbox("Threshold", options=list(thresholds.keys()), index=0)
THRESHOLD = thresholds[chosen_thr_key]
st.sidebar.write(f"**Valor do threshold**: `{THRESHOLD:.4f}`")

# ==============================
# PREVISÃO AUTOMÁTICA
# ==============================
st.subheader("📂 Previsão Automática (CSV/Parquet)")
uploaded = st.file_uploader("Envie um arquivo .csv ou .parquet com as features do modelo", type=["csv", "parquet"])

def prepare_X(df_in: pd.DataFrame, feats: list) -> pd.DataFrame:
    """
    Reindexa para as features do modelo, preenche ausentes com 0 e converte para numérico quando possível.
    """
    X = df_in.reindex(columns=feats, fill_value=np.nan)
    # tentar converter para numérico; itens não numéricos viram NaN -> preenche 0
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)
    return X

def predict_batch(dfX: pd.DataFrame, thr: float):
    prob = model.predict_proba(dfX)[:, 1]
    pred = (prob >= thr).astype(int)
    return prob, pred

if uploaded is not None:
    # Ler arquivo
    df_in = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_parquet(uploaded)
    st.write("📊 Prévia dos dados carregados:")
    st.dataframe(df_in.head())

    # Preparar X com as features certas
    missing = [f for f in features if f not in df_in.columns]
    extra   = [c for c in df_in.columns if c not in features and c != "fraude"]
    if missing:
        st.warning(f"Algumas features do modelo não estão no arquivo e serão preenchidas com 0: {missing[:10]}{'...' if len(missing)>10 else ''}")
    if extra:
        st.info(f"Colunas extras serão ignoradas para o modelo: {extra[:10]}{'...' if len(extra)>10 else ''}")

    X = prepare_X(df_in, features)

    if st.button("🚀 Executar Previsões"):
        proba, pred = predict_batch(X, THRESHOLD)

        out = df_in.copy()
        out["prob_fraude"] = proba
        out["prev_class"]  = pred
        out["prev_label"]  = np.where(pred == 1, "Fraude", "Não Fraude")

        st.success("Previsões concluídas!")
        st.dataframe(out.head(30))

        # Acurácia (se houver coluna 'fraude')
        if "fraude" in out.columns:
            acc = (out["prev_class"].astype(int) == out["fraude"].astype(int)).mean()
            st.markdown(f"**Acurácia no arquivo (comparando vs 'fraude')**: `{acc:.4f}`")

        # Download
        st.markdown("---")
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Baixar resultados (.csv)", data=csv_bytes, file_name="previsoes_fraude.csv", mime="text/csv")

# ==============================
# RODAPÉ
# ==============================
st.markdown("---")
st.markdown("<div class='footer'>Estilo: Gold Luxury &nbsp;|&nbsp; Autor: Isaac Araújo</div>", unsafe_allow_html=True)
