import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import yaml
import os
import base64
import io
from sklearn.preprocessing import StandardScaler

# Carregar arquivo de configuração
config_path = 'C:/Github/data-science/projetos/churn_clientes/config/config.yaml'
with open(config_path, 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Função para tratar e limpar os dados carregados
def tratar_dados(df):
    trad_colunas = {
        'CustomerID': 'id_cliente',
        'Age': 'idade',
        'Gender': 'genero',
        'Tenure': 'tempo_de_assinatura',
        'Usage Frequency': 'frequencia_uso',
        'Support Calls': 'chamadas_suporte',
        'Payment Delay': 'atraso_pagamento',
        'Subscription Type': 'tipo_assinatura',
        'Contract Length': 'duracao_contrato',
        'Total Spend': 'gasto_total',
        'Last Interaction': 'ultima_interacao',
        'Churn': 'churn'
    }
    df.rename(columns=trad_colunas, inplace=True)
    df.fillna(df.median(), inplace=True)
    # Verificar dados duplicados
    df.drop_duplicates(inplace=True)
    # Remover outliers (IQR)
    for coluna in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[coluna].quantile(0.25)
        Q3 = df[coluna].quantile(0.75)
        IQR = Q3 - Q1
        outlier_threshold_min = Q1 - 1.5 * IQR
        outlier_threshold_max = Q3 + 1.5 * IQR
        df = df[(df[coluna] >= outlier_threshold_min) & (df[coluna] <= outlier_threshold_max)]
    # Normalizar e padronizar os dados
    scaler = StandardScaler()
    df[df.select_dtypes(include=[np.number]).columns] = scaler.fit_transform(df.select_dtypes(include=[np.number]).values)
    return df

# Inicializar a aplicação Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = 'Churn Dashboard'

app.layout = html.Div([
    dbc.NavbarSimple(
        brand="Churn Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    dbc.Container([
        dcc.Tabs([
            dcc.Tab(label='Importação e Tratamento', children=[
                html.H3('Importação e Tratamento de Dados'),
                dcc.Upload(
                    id='upload-data',
                    children=html.Div(['Arraste e solte ou ', html.A('selecione um arquivo')]),
                    style={
                        'width': '100%',
                        'height': '60px',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin': '10px'
                    },
                    multiple=False
                ),
                html.Div(id='output-data-upload'),
                html.Button('Salvar Dados Tratados', id='save-button', n_clicks=0),
                html.Div(id='save-output'),
            ]),
        ])
    ])
])

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified')
)
def update_output(content, name, date):
    if content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in name:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                df = tratar_dados(df)
                return html.Div([
                    html.H5(name),
                    html.H6('Dados carregados e tratados com sucesso'),
                    dbc.Table.from_dataframe(df.head(), striped=True, bordered=True, hover=True)
                ])
            else:
                return html.Div(['Por favor, faça o upload de um arquivo CSV.'])
        except Exception as e:
            return html.Div([f'Houve um erro ao processar este arquivo: {str(e)}'])
    return None

@app.callback(
    Output('save-output', 'children'),
    Input('save-button', 'n_clicks'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def save_processed_data(n_clicks, content, name):
    if n_clicks > 0 and content is not None:
        content_type, content_string = content.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'csv' in name:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                df = tratar_dados(df)
                processed_dir = r'C:\Users\Windows Lite BR\Downloads'
                if not os.path.exists(processed_dir):
                    os.makedirs(processed_dir)
                processed_path_csv = os.path.join(processed_dir, "processed.csv")
                processed_path_parquet = os.path.join(processed_dir, "processed.parquet")
                df.to_csv(processed_path_csv, index=False)
                df.to_parquet(processed_path_parquet, index=False)
                return html.Div(['Dados tratados e salvos em C:\\Users\\Windows Lite BR\\Downloads com sucesso.'])
            else:
                return html.Div(['Por favor, faça o upload de um arquivo CSV.'])
        except Exception as e:
            return html.Div([f'Houve um erro ao salvar os dados tratados: {str(e)}'])
    return None

if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=False)
