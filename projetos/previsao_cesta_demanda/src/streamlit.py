# app/app.py
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import io
import time

# =====================================
# CONFIGURAÇÃO
# =====================================
st.set_page_config(
    page_title="Previsão de Demanda | Isaac Araújo",
    page_icon="📦",
    layout="centered"
)

# CSS customizado
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Caminhos fixos
MODEL_PATH = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/lightgbm_refinado.joblib")
DATA_PATH  = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/data/processed/processed_features.parquet")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_base():
    return pd.read_parquet(DATA_PATH)

saved = load_model()
df_base = load_base()

# =====================================
# FUNÇÕES AUXILIARES
# =====================================
def _recompute_logs(df):
    if "product_density" not in df.columns and {"product_weight_g", "product_volume_cm3"} <= set(df.columns):
        w = pd.to_numeric(df["product_weight_g"], errors="coerce")
        v = pd.to_numeric(df["product_volume_cm3"], errors="coerce")
        df["product_density"] = np.where((v > 0) & (~w.isna()), w / v, np.nan)
    for col in ["price", "freight_value", "product_weight_g", "product_volume_cm3", "product_density"]:
        df[f"{col}_log"] = np.log1p(pd.to_numeric(df[col], errors="coerce")).fillna(0)
    return df

def predict_demand(df_input):
    model = saved["model"]
    prep_state = saved.get("prep_state", {})
    features = saved["features"]

    df_input = _recompute_logs(df_input.copy())
    for col in features:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[features]

    y_pred = model.predict(df_input)
    return np.expm1(y_pred) if saved.get("target_transform") == "log1p" else y_pred


# =====================================
# INTRO (animação de boas-vindas)
# =====================================
if "show_intro" not in st.session_state:
    st.session_state.show_intro = True

if st.session_state.show_intro:
    st.markdown("""
        <div class='intro'>
            <h1 class='intro-title'>Bem-vindo</h1>
            <p class='intro-sub'>Site desenvolvido pelo Cientista/Analista de Dados<br><b>Isaac Araújo</b></p>
        </div>
    """, unsafe_allow_html=True)

    time.sleep(2.8)
    st.session_state.show_intro = False
    st.rerun()

# =====================================
# INTERFACE PRINCIPAL
# =====================================
st.markdown("<h1 class='title'>Previsão de Demanda</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='subtitle'>por Isaac Araújo</h3>", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

abas = st.tabs([
    "🏠 Bem-vindo",
    "🧮 Manual",
    "📂 Automática"
])

# ------------------------------
# 🏠 ABA BEM-VINDO
# ------------------------------
with abas[0]:
    st.markdown("""
    <div class='welcome'>
        <h2><i class="fa-solid fa-chart-line"></i> Bem-vindo ao Sistema de Previsão de Demanda</h2>
        <p>
        Este aplicativo utiliza um modelo <b>LightGBM refinado</b>, desenvolvido por <b>Isaac Araújo</b>,
        para prever a demanda de produtos com base em variáveis como preço, frete, peso e volume.
        </p>
        <p>Você pode escolher entre dois modos de previsão:</p>
        <ul>
            <li><b>Manual:</b> insira os dados diretamente no painel e obtenha a previsão instantânea.</li>
            <li><b>Automático:</b> envie um arquivo com os dados e o sistema calculará tudo para você.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------
# 🧮 ABA MANUAL
# ------------------------------
with abas[1]:
    st.subheader("Previsão Manual")

    col1, col2 = st.columns(2)
    with col1:
        price = st.number_input("Preço do Produto (R$)", min_value=0.0, value=100.0)
        freight = st.number_input("Valor do Frete (R$)", min_value=0.0, value=20.0)
        weight = st.number_input("Peso (g)", min_value=0.0, value=800.0)
    with col2:
        volume = st.number_input("Volume (cm³)", min_value=0.0, value=2000.0)

    if st.button("Prever Demanda"):
        df = pd.DataFrame([{
            "price": price,
            "freight_value": freight,
            "product_weight_g": weight,
            "product_volume_cm3": volume
        }])
        result = predict_demand(df)
        st.markdown(
            f"<div class='result-card'><i class='fa-solid fa-cubes'></i> Demanda prevista: <b>{result[0]:.2f}</b> unidades</div>",
            unsafe_allow_html=True
        )

# ------------------------------
# 📂 ABA AUTOMÁTICA
# ------------------------------
with abas[2]:
    st.subheader("Previsão Automática (Upload de Arquivo)")

    uploaded = st.file_uploader("Envie seu arquivo (.CSV ou .Parquet)", type=["csv", "parquet"])

    if uploaded:
        if uploaded.name.endswith(".csv"):
            user_df = pd.read_csv(uploaded)
        else:
            user_df = pd.read_parquet(io.BytesIO(uploaded.read()))

        st.markdown("<p class='preview-title'><i class='fa-solid fa-table'></i> Prévia dos dados:</p>", unsafe_allow_html=True)
        st.dataframe(user_df.head())

        if st.button("Gerar Previsões"):
            preds = predict_demand(user_df)
            user_df["demanda_prevista"] = preds
            st.success("Previsões geradas com sucesso!")
            st.dataframe(user_df.head())

            csv_buffer = io.StringIO()
            user_df.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Baixar Previsões (CSV)",
                data=csv_buffer.getvalue(),
                file_name="previsoes_demanda.csv",
                mime="text/csv"
            )
