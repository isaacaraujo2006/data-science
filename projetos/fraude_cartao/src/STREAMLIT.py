import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configurações básicas ---
st.set_page_config(
    page_title="Previsão de Fraude - Sistema Avançado",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Função para carregar modelo e threshold ---
@st.cache_resource(show_spinner=True)
def load_model_and_threshold(model_path, threshold_path):
    model_output = joblib.load(model_path)
    model = model_output['model']
    threshold = model_output['threshold']
    return model, threshold

# --- Caminhos (ajuste para seu ambiente) ---
MODEL_PATH = "D:/github/data-science/projetos/fraude_cartao/models/histgb_model_calibrated.pkl"

# Carrega modelo e threshold
model_output = joblib.load(MODEL_PATH)
model = model_output['model']
threshold = model_output['threshold']

# Classe índice fraudada para previsões probabilísticas
fraude_index = list(model.classes_).index(1)

# --- Função para prever fraudem dado dataframe ---
def predict_fraude(df_input):
    probs = model.predict_proba(df_input)[:, fraude_index]
    preds = (probs >= threshold).astype(int)
    return probs, preds

# --- Layout Streamlit ---

st.title("🛡️ Previsão de Fraude em Transações")
st.markdown("""
Este sistema utiliza um modelo avançado treinado para identificar fraudes em transações financeiras.  
Você pode enviar um arquivo CSV com dados em lote, ou preencher manualmente para previsão individual.
""")

# Sidebar para escolher modo
modo = st.sidebar.selectbox("Modo de Previsão", ["Previsão Individual", "Previsão em Lote (CSV)"])

# --- Previsão Individual ---
if modo == "Previsão Individual":
    st.header("Previsão Individual de Fraude")

    # Para facilitar, lista das features em ordem (use a do seu dataset)
    FEATURES = [
        'time', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12',
        'v13', 'v14', 'v15', 'v16', 'v17', 'v18', 'v19', 'v20', 'v21', 'v22', 'v23', 'v24',
        'v25', 'v26', 'v27', 'v28', 'amount', 'outlier_amount', 'hour_of_day'
    ]

    input_data = {}
    with st.form("form_individual", clear_on_submit=False):
        for feat in FEATURES:
            # valores numéricos, usar input numérico
            if feat in ['time', 'amount', 'outlier_amount', 'hour_of_day']:
                val = st.number_input(f"{feat}", value=0.0, format="%.6f")
            else:
                val = st.number_input(f"{feat}", value=0.0, format="%.6f")
            input_data[feat] = val
        submitted = st.form_submit_button("Prever Fraude")

    if submitted:
        df_input = pd.DataFrame([input_data])
        probs, preds = predict_fraude(df_input)
        prob = probs[0]
        pred = preds[0]
        st.markdown(f"### Resultado da Previsão:")
        st.write(f"Probabilidade de Fraude: **{prob:.4f}**")
        if pred == 1:
            st.error("🚩 **Atenção: Transação Provavelmente FRAUDE!**")
        else:
            st.success("✅ Transação Provavelmente NÃO FRAUDE")

# --- Previsão em lote ---
else:
    st.header("Previsão em Lote - Upload de CSV")
    uploaded_file = st.file_uploader("Envie um arquivo CSV com as mesmas colunas de atributos (sem coluna 'class')", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # Checar se todas as colunas necessárias estão presentes
            missing_cols = set(FEATURES) - set(df.columns)
            if missing_cols:
                st.error(f"Faltam colunas no arquivo: {missing_cols}")
            else:
                probs, preds = predict_fraude(df[FEATURES])
                df['probabilidade_fraude'] = probs
                df['predicao_fraude'] = preds
                st.success(f"Previsão realizada com sucesso! {len(df)} registros processados.")

                st.dataframe(df[['probabilidade_fraude', 'predicao_fraude'] + FEATURES].head(15))

                # Download do resultado
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Baixar Resultados com Previsões",
                    data=csv,
                    file_name="resultados_previsao.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")

# --- Rodapé ---
st.markdown("---")
st.markdown("**Modelo calibrado e threshold personalizado para melhor decisão.**")
st.markdown("Desenvolvido por Isaac Araújo. 🧠🚀")
