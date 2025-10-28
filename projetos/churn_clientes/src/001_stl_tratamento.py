import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

def tratar_arquivo(df):
    # Verificar o número de colunas no DataFrame original
    print("Colunas reais no dataset:", df.columns)
    
    # Ajuste a lista de colunas de acordo com o número real de colunas
    df.columns = [
        'id_cliente', 'sobrenome', 'pontuacao_credito', 'geografia', 'genero', 'idade',
        'tempo_relacionamento', 'saldo', 'numero_produtos', 'cartao_credito', 'membro_ativo', 
        'salario_estimado', 'coluna_extra1', 'coluna_extra2'  # Ajuste para o número correto de colunas
    ]

    # Continue com o resto do código para tratamento de dados
    df = df.drop_duplicates()

    # Remover colunas de sobrenomes vazias
    df = df.drop(columns=['sobrenome'])

    # Tratar valores faltantes
    for coluna in df.select_dtypes(include=[float, int]).columns:
        df[coluna] = df[coluna].fillna(df[coluna].mean())

    # Verificar consistência dos dados
    intervalos = {
        'pontuacao_credito': (300, 850),
        'idade': (18, 120)
    }

    for coluna, (min_val, max_val) in intervalos.items():
        if pd.api.types.is_numeric_dtype(df[coluna]):
            df.loc[df[coluna] < min_val, coluna] = min_val
            df.loc[df[coluna] > max_val, coluna] = max_val

    # Normalizar e padronizar dados numéricos
    scaler = StandardScaler()
    colunas_numericas = df.select_dtypes(include=[float, int]).columns
    df[colunas_numericas] = scaler.fit_transform(df[colunas_numericas])

    # Criar uma coluna de exemplo para 'classe'
    np.random.seed(42)
    df['classe'] = np.random.choice([0, 1], size=len(df))

    # Aplicar Label Encoding para 'geografia' e 'genero'
    le_geografia = LabelEncoder()
    le_genero = LabelEncoder()

    df['geografia'] = le_geografia.fit_transform(df['geografia'])
    df['genero'] = le_genero.fit_transform(df['genero'])

    # Separar características e rótulos
    coluna_alvo = 'classe'
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]

    # Garantir que X e y não contenham valores nulos
    X = X.fillna(0)
    y = y.fillna(y.mode()[0])

    # Aplicar SMOTE apenas nas colunas numéricas
    X_numericas = X.select_dtypes(include=[float, int])
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_numericas, y)

    # Obter as colunas categóricas restantes
    X_categoricas = X.drop(columns=X_numericas.columns)

    # Concatenar as colunas numéricas balanceadas com as categóricas originais
    X_res_final = pd.concat([X_categoricas.reset_index(drop=True), pd.DataFrame(X_res, columns=X_numericas.columns).reset_index(drop=True)], axis=1)

    # Unir X_res_final com y_res para o DataFrame final
    df_res = pd.concat([X_res_final, pd.DataFrame(y_res, columns=[coluna_alvo])], axis=1)

    # Calcular os outliers com base nas colunas numéricas após o SMOTE
    outliers = {}
    for coluna in colunas_numericas:
        if coluna in df_res.columns:
            Q1 = df_res[coluna].quantile(0.25)
            Q3 = df_res[coluna].quantile(0.75)
            IQR = Q3 - Q1
            lim_inf = Q1 - 1.5 * IQR
            lim_sup = Q3 + 1.5 * IQR
            outliers[coluna] = len(df_res[(df_res[coluna] < lim_inf) | (df_res[coluna] > lim_sup)])

    return df_res, outliers

st.title("Tratamento de Arquivos CSV e Parquet")
st.write("Faça o upload de um arquivo .csv ou .parquet para ser tratado.")

# Upload do arquivo
uploaded_file = st.file_uploader("Escolha um arquivo", type=["csv", "parquet"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_parquet(uploaded_file)

    st.write("Dados brutos:")
    st.write(df.head())

    df_tratado, outliers = tratar_arquivo(df)

    st.write("Número de outliers por coluna após tratamento:")
    st.write(outliers)

    # Salvar o arquivo tratado
    save_path = 'C:/Users/Windows Lite BR/Downloads/dados_tratados.csv'
    df_tratado.to_csv(save_path, index=False)
    st.success(f"Arquivo tratado salvo em {save_path}")

# Para rodar o aplicativo, execute o comando no terminal:
# streamlit run <nome_do_arquivo>.py
