import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import joblib

# ===== Funções auxiliares =====
def flag_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lim_inf = Q1 - 1.5 * IQR
    lim_sup = Q3 + 1.5 * IQR
    flag = ((df[column] < lim_inf) | (df[column] > lim_sup)).astype(int)
    return flag

def extract_time_features(df, time_col='time'):
    df['hour_of_day'] = ((df[time_col] // 3600) % 24).astype(int)
    return df

# ===== Main =====
if __name__ == "__main__":
    # ========== 1. Carregar config.yaml ==========
    with open("D:/github/data-science/projetos/fraude_cartao/config/config.yaml", 'r') as file:
        config = yaml.safe_load(file)

    # ========== 2. Converter CSV bruto em Parquet se necessário ==========
    raw_csv_path = config['data']['raw']
    raw_parquet_path = raw_csv_path.replace('.csv', '.parquet')

    if not os.path.exists(raw_parquet_path):
        print(f"🔄 Convertendo arquivo bruto para Parquet: {raw_parquet_path}")
        df_temp = pd.read_csv(raw_csv_path)
        df_temp.to_parquet(raw_parquet_path, index=False)
        print(f"✅ Arquivo Parquet salvo em: {raw_parquet_path}")
        # Opcional: remover o arquivo CSV raw para manter só Parquet (descomente se quiser)
        # os.remove(raw_csv_path)
        # print(f"🗑️ Arquivo CSV raw removido: {raw_csv_path}")
    else:
        print(f"✅ Arquivo Parquet já existe: {raw_parquet_path}")

    # ========== 3. Ler os dados brutos a partir do Parquet ==========
    df = pd.read_parquet(raw_parquet_path)
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]

    # ========== 4. Relatório inicial ==========
    num_linhas = len(df)
    print(f"✅ Número total de linhas importadas: {num_linhas}")

    # ========== 5. Verificar e tratar duplicados ==========
    num_duplicados = df.duplicated().sum()
    perc_duplicados = num_duplicados / num_linhas * 100
    print(f"Duplicados: {num_duplicados} linhas ({perc_duplicados:.4f}%)")
    if num_duplicados > 0:
        df = df.drop_duplicates()
        print(f"Duplicados removidos. Novo total de linhas: {len(df)}")

    num_linhas = len(df)

    # ========== 6. Verificar dados faltantes ==========
    faltantes = df.isnull().sum()
    faltantes_existem = faltantes.sum() > 0
    print("\nDados faltantes por coluna (apenas >0):")
    print(faltantes[faltantes > 0])
    if faltantes_existem:
        print("Tratando dados faltantes: preenchendo com mediana")
        for col in faltantes.index[faltantes > 0]:
            mediana = df[col].median()
            df[col].fillna(mediana, inplace=True)
    else:
        print("Nenhum dado faltante detectado.")

    # ========== 7. Verificar valores fora do intervalo esperado ==========
    fora_intervalo_amount = df[(df['amount'] < 0)].shape[0]
    fora_intervalo_time = df[(df['time'] < 0)].shape[0]
    print(f"\nValores fora do intervalo esperado:")
    print(f"amount < 0: {fora_intervalo_amount} linhas ({fora_intervalo_amount/num_linhas*100:.4f}%)")
    print(f"time < 0: {fora_intervalo_time} linhas ({fora_intervalo_time/num_linhas*100:.4f}%)")

    if fora_intervalo_amount > 0 or fora_intervalo_time > 0:
        df = df[(df['amount'] >= 0) & (df['time'] >= 0)]
        print(f"Linhas com valores inválidos removidas. Novo total: {len(df)}")

    num_linhas = len(df)

    # ========== 8. Criar flag de outliers na feature 'amount' ==========
    df['outlier_amount'] = flag_outliers_iqr(df, 'amount')
    num_outliers = df['outlier_amount'].sum()
    perc_outliers = num_outliers / num_linhas * 100
    print(f"\nOutliers detectados em 'amount': {num_outliers} linhas ({perc_outliers:.4f}%)")
    print("Outliers mantidos para ajudar o modelo a capturar possíveis fraudes.")

    # ========== 9. Feature engineering temporal ==========
    df = extract_time_features(df, time_col='time')

    # ========== 10. Separar features e target ==========
    X = df.drop(columns=['class'])
    y = df['class']

    # ========== 11. Escalonamento com RobustScaler ==========
    scaler = RobustScaler()
    cols_to_scale = ['amount', 'time']
    X[cols_to_scale] = scaler.fit_transform(X[cols_to_scale])

    # ========== 12. Split treino/teste estratificado ==========
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ========== 13. Salvar dados processados (Somente Parquet) ==========
    processed_df = X.copy()
    processed_df['class'] = y
    # NÃO salvar CSV processado
    # processed_df.to_csv(config['data']['processed_csv'], index=False)  # removido
    processed_df.to_parquet(config['data']['processed_parquet'], index=False)

    # ========== 14. Criar versão para Power BI ==========
    powerbi_df = processed_df.copy()
    powerbi_df['fraude'] = powerbi_df['class'].map({0: 'Não Fraude', 1: 'Fraude'})
    powerbi_df = powerbi_df.drop(columns=['class'])

    cols = [col for col in powerbi_df.columns if col != 'fraude'] + ['fraude']
    powerbi_df = powerbi_df[cols]

    powerbi_path = os.path.join(os.path.dirname(config['data']['processed_parquet']), "powerbi_ready.csv")
    powerbi_df.to_csv(
        powerbi_path,
        index=False,
        encoding='utf-8-sig',
        sep=';',
        decimal=',',
        float_format='%.6f'
    )
    print(f"\n✅ Arquivo para Power BI salvo em: {powerbi_path}")

    # ========== 15. Salvar scaler ==========
    joblib.dump(scaler, config['preprocessors']['scaler_path'])

    # ========== 16. Salvar splits ==========
    split_dir = config['data'].get('split_dir', "D:/github/data-science/projetos/fraude_cartao/data/split/")
    os.makedirs(split_dir, exist_ok=True)

    X_train.to_csv(os.path.join(split_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(split_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(split_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(split_dir, "y_test.csv"), index=False)

    print("\n✅ Pré-processamento avançado concluído com sucesso!")
    print(f"Dados finais com {len(X)} linhas e {X.shape[1]} colunas.")
