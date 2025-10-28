import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Previsão de Risco de Crédito",
    page_icon="💳",
    layout="wide"
)

# ===== CSS PROFISSIONAL =====
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: #161A23 !important;
        color: #f3f6fd !important;
    }
    h1, h2, h3, h4, .big-title {
        color: #1976d2 !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px;
    }
    .sub, .stMarkdown, .stDataFrame, .stTable { color: #f3f6fd !important; }
    /* Aba */
    .stTabs [role="tablist"] { background: none !important; }
    .stTabs [role="tab"][aria-selected="true"] {
        background: #1976d2 !important;
        color: #fff !important;
        border-radius: 16px 16px 0 0 !important;
        font-weight: 600 !important;
    }
    .stTabs [role="tab"] {
        background: #222634 !important;
        color: #a4b8d3 !important;
        border-radius: 16px 16px 0 0 !important;
        font-weight: 500;
        padding: 10px 32px !important;
        margin-bottom: 0;
    }
    /* Cards das abas */
    [data-testid="stHorizontalBlock"] > div, .stTabs [role="tabpanel"] {
        background: #23272F !important;
        border-radius: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(24,24,49,.25);
        padding: 28px 24px 18px 24px !important;
        margin-top: 8px !important;
    }
    div[data-testid="stForm"] {
        background: #1d2028 !important;
        border-radius: 22px !important;
        box-shadow: 0 4px 24px 0 rgba(24,24,49,.15);
        padding: 26px !important;
        margin-bottom: 32px;
    }
    .stNumberInput, .stTextInput, .stSelectbox, .stSlider, input, select, textarea {
        background: #222634 !important;
        color: #f3f6fd !important;
        border-radius: 12px !important;
    }
    .stSlider > div {
        background: #23272F !important;
    }
    .stButton>button, button[kind="primary"] {
        background: linear-gradient(90deg,#1976d2 30%,#30a7f7 100%) !important;
        color: #fff !important;
        border-radius: 12px !important;
        font-weight: 600;
        border: none;
        margin-top: 8px;
        box-shadow: 0 1px 10px 0 #1976d263;
        transition: 0.15s;
    }
    .stButton>button:hover {
        background: #174c94 !important;
        color: #fff !important;
        transform: translateY(-2px) scale(1.02);
    }
    .stDataFrame, .stTable, .stMarkdown {
        background: #212433 !important;
        color: #f3f6fd !important;
        border-radius: 14px !important;
    }
    .stProgress > div > div > div > div {
        background-image: linear-gradient(90deg,#1976d2,#30a7f7);
        height: 22px !important;
        border-radius: 12px !important;
    }
    .pulse {
        animation: pulse 1s infinite alternate;
    }
    @keyframes pulse {
        0% { text-shadow: 0 0 0px #30a7f7; }
        100% { text-shadow: 0 0 18px #30a7f7; }
    }
    ::-webkit-scrollbar-thumb { background: #222634; border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

MODEL_PATH = "D:/github/data-science/projetos/risco_credito/models/catboost_optimized_calibrated.joblib"
THRESHOLD_PATH = "D:/github/data-science/projetos/risco_credito/models/catboost_optimized_threshold.txt"

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH, 'r') as f:
        threshold = float(f.read().strip())
    return model, threshold

model, threshold = load_model()
CAT_COLS = ['sexo', 'educacao', 'estado_civil']

st.markdown('<h1 class="big-title">💳 Previsão de Risco de Crédito</h1>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub" style="font-size:1.2em;">Faça previsões em lote ou individuais com nosso modelo avançado de risco de crédito.</div>',
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["📂 Previsão em Lote", "📝 Previsão Manual"])

with tab1:
    st.header("📂 Previsão por Arquivo")
    st.write("Faça upload do seu arquivo CSV ou Parquet (com as mesmas colunas do treinamento).")
    file = st.file_uploader("Selecione seu arquivo", type=['csv', 'parquet'])

    if file is not None:
        ext = os.path.splitext(file.name)[1]
        if ext == ".csv":
            df_upload = pd.read_csv(file)
        elif ext == ".parquet":
            df_upload = pd.read_parquet(file)
        else:
            st.error("Formato de arquivo não suportado. Envie .csv ou .parquet")
            st.stop()

        for c in CAT_COLS:
            if c in df_upload.columns:
                df_upload[c] = df_upload[c].astype(int)

        cols_features = [col for col in df_upload.columns if col not in ['inadimplente_mes_seguinte']]
        X_batch = df_upload[cols_features]

        probs = model.predict_proba(X_batch)[:, 1]
        preds = (probs >= threshold).astype(int)

        df_upload['probabilidade'] = probs
        df_upload['previsao'] = preds

        st.success(f"Arquivo processado! Threshold usado: **{threshold:.2f}**")
        st.dataframe(df_upload.head(30), use_container_width=True)

        csv = df_upload.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Baixar resultado (.csv)",
            data=csv,
            file_name="previsoes_risco_credito.csv",
            mime='text/csv',
            use_container_width=True
        )

with tab2:
    st.header("📝 Previsão Manual")
    st.write("Preencha os campos abaixo para obter a previsão de risco de crédito.")

    with st.form(key='form_previsao'):
        col1, col2, col3 = st.columns(3)
        with col1:
            limite_credito = st.number_input("Limite de Crédito", value=50000, min_value=0, step=1000)
            sexo = st.selectbox("Sexo", options=[1, 2], format_func=lambda x: "Masculino" if x==1 else "Feminino")
            idade = st.number_input("Idade", value=30, min_value=18, max_value=100, step=1)
            estado_civil = st.selectbox("Estado Civil", options=[1, 2, 3], format_func=lambda x: {1:"Solteiro", 2:"Casado", 3:"Outro"}[x])
        with col2:
            educacao = st.selectbox("Educação", options=[1, 2, 3, 4], format_func=lambda x: {1:"Pós-graduação", 2:"Graduação", 3:"Ensino Médio", 4:"Outro"}[x])
            pagamento_mes_0 = st.slider("Pagamento Mês 0", min_value=-2, max_value=8, value=0)
            pagamento_mes_2 = st.slider("Pagamento Mês 2", min_value=-2, max_value=8, value=0)
            pagamento_mes_3 = st.slider("Pagamento Mês 3", min_value=-2, max_value=8, value=0)
        with col3:
            pagamento_mes_4 = st.slider("Pagamento Mês 4", min_value=-2, max_value=8, value=0)
            pagamento_mes_5 = st.slider("Pagamento Mês 5", min_value=-2, max_value=8, value=0)
            pagamento_mes_6 = st.slider("Pagamento Mês 6", min_value=-2, max_value=8, value=0)
            pagamento_mes_1 = st.number_input("Pagamento Mês 1", value=0.0, min_value=-2.0, max_value=8.0, step=0.01)

        fatura_mes_1 = st.number_input("Fatura Mês 1", value=0.0)
        fatura_mes_2 = st.number_input("Fatura Mês 2", value=0.0)
        fatura_mes_3 = st.number_input("Fatura Mês 3", value=0.0)
        fatura_mes_4 = st.number_input("Fatura Mês 4", value=0.0)
        fatura_mes_5 = st.number_input("Fatura Mês 5", value=0.0)
        fatura_mes_6 = st.number_input("Fatura Mês 6", value=0.0)

        submitted = st.form_submit_button("Prever")

    if submitted:
        df_manual = pd.DataFrame([{
            'limite_credito': limite_credito,
            'sexo': int(sexo),
            'educacao': int(educacao),
            'estado_civil': int(estado_civil),
            'idade': idade,
            'pagamento_mes_0': pagamento_mes_0,
            'pagamento_mes_2': pagamento_mes_2,
            'pagamento_mes_3': pagamento_mes_3,
            'pagamento_mes_4': pagamento_mes_4,
            'pagamento_mes_5': pagamento_mes_5,
            'pagamento_mes_6': pagamento_mes_6,
            'pagamento_mes_1': pagamento_mes_1,
            'fatura_mes_1': fatura_mes_1,
            'fatura_mes_2': fatura_mes_2,
            'fatura_mes_3': fatura_mes_3,
            'fatura_mes_4': fatura_mes_4,
            'fatura_mes_5': fatura_mes_5,
            'fatura_mes_6': fatura_mes_6
        }])

        for c in CAT_COLS:
            if c in df_manual.columns:
                df_manual[c] = df_manual[c].astype(int)

        prob = model.predict_proba(df_manual)[:, 1][0]
        prev = int(prob >= threshold)
        label = "INADIMPLENTE" if prev == 1 else "APROVADO"
        cor = "#e34a51" if label == "INADIMPLENTE" else "#32a852"

        st.markdown(
            f"""<div style="font-size:1.4em;padding:18px 0 10px 0;font-weight:800;color:{cor};" class="pulse">
                Resultado: <span style="color:{cor}">{label}</span></div>""", 
            unsafe_allow_html=True)
        st.markdown(f"<b>Probabilidade prevista:</b> <span style='color:#1976d2'>{prob:.2%}</span>", unsafe_allow_html=True)
        st.markdown(f"<b>Threshold utilizado:</b> <span style='color:#1976d2'>{threshold:.2f}</span>", unsafe_allow_html=True)
        st.progress(prob)
