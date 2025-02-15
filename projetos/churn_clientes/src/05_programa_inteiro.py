import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import base64
import io
import plotly.express as px
import joblib
import os

# Inicializar aplicação Dash
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = 'Dashboard de Churn'

# Carregar o modelo e o scaler
modelo = joblib.load('C:\\Github\\data-science\\projetos\\churn_clientes\\models\\logistic_regression_model.joblib')
scaler = joblib.load('C:/Github/data-science/projetos/churn_clientes/preprocessors/scaler.joblib')

# Lista completa das características do modelo
colunas_modelo = ['Age', 'Total Spend', 'Tenure', 'Usage Frequency', 'Gender', 'Device Type', 'Churn', 'Contract Type', 'Payment Method', 'Monthly Spend', 'Number of Products', 'Support Calls', 'Account Length']

# Diretório para salvar os dados processados
save_directory = 'C:\\Users\\Windows Lite BR\\Downloads\\'

# Diretório para salvar as previsões
predictions_directory = 'C:\\Github\\data-science\\projetos\\churn_clientes\\predictions\\'

# Layout
app.layout = dbc.Container([  
    dbc.NavbarSimple(
        brand="Churn Dashboard",
        brand_href="#",
        color="dark",
        dark=True,
    ),
    dcc.Tabs([  
        dcc.Tab(label="Importação e Tratamento", children=[  
            html.H3("Importação e Tratamento de Dados"),  
            dcc.Upload(  
                id="upload-data",  
                children=html.Button("Carregar Arquivo (.csv ou .parquet)"),  
                multiple=False,  
                style={"margin": "20px"}  
            ),  
            html.Div(id="output-data-upload"),  
            html.Button("Salvar Dados Tratados", id="save-button", n_clicks=0, className="btn btn-primary"),  
            html.Div(id="save-output")  
        ]),  
        dcc.Tab(label="Análise Exploratória", children=[  
            html.H3("Análise de Dados"),  
            html.Div([  
                dcc.Upload(  
                    id="upload-exploracao-data",  
                    children=html.Button("Carregar Arquivo Tratado."),  
                    multiple=False,  
                    style={"margin": "20px"}  
                ),  
                html.Div(id="output-data-upload-exploracao"),  
            ]),  

            html.Div([  
                dcc.Dropdown(id='dropdown-grafico-1', placeholder="Selecione o gráfico", options=[  
                    {'label': 'Histograma de Idade', 'value': 'hist_idade'},  
                    {'label': 'Boxplot de Gasto Total', 'value': 'box_gasto_total'},  
                    {'label': 'Heatmap de Correlação', 'value': 'heatmap_correlacao'},  
                    {'label': 'Frequência de Churn', 'value': 'frequencia_churn'},  
                    {'label': 'Boxplot de Tempo de Assinatura', 'value': 'box_tempo_assinatura'},  
                    {'label': 'Histograma de Frequência de Uso', 'value': 'hist_frequencia_uso'},  
                    {'label': 'Scatterplot Gasto x Tempo', 'value': 'scatter_gasto_tempo'}  
                ]),  
                dcc.Graph(id="grafico-1")  
            ], style={"width": "48%", "display": "inline-block"}),  

            html.Div([  
                dcc.Dropdown(id='dropdown-grafico-2', placeholder="Selecione o gráfico", options=[  
                    {'label': 'Histograma de Idade', 'value': 'hist_idade'},  
                    {'label': 'Boxplot de Gasto Total', 'value': 'box_gasto_total'},  
                    {'label': 'Heatmap de Correlação', 'value': 'heatmap_correlacao'},  
                    {'label': 'Frequência de Churn', 'value': 'frequencia_churn'},  
                    {'label': 'Boxplot de Tempo de Assinatura', 'value': 'box_tempo_assinatura'},  
                    {'label': 'Histograma de Frequência de Uso', 'value': 'hist_frequencia_uso'},  
                    {'label': 'Scatterplot Gasto x Tempo', 'value': 'scatter_gasto_tempo'}  
                ]),  
                dcc.Graph(id="grafico-2")  
            ], style={"width": "48%", "display": "inline-block"}),  

            html.Div([  
                dcc.Dropdown(id='dropdown-grafico-3', placeholder="Selecione o gráfico", options=[  
                    {'label': 'Histograma de Idade', 'value': 'hist_idade'},  
                    {'label': 'Boxplot de Gasto Total', 'value': 'box_gasto_total'},  
                    {'label': 'Heatmap de Correlação', 'value': 'heatmap_correlacao'},  
                    {'label': 'Frequência de Churn', 'value': 'frequencia_churn'},  
                    {'label': 'Boxplot de Tempo de Assinatura', 'value': 'box_tempo_assinatura'},  
                    {'label': 'Histograma de Frequência de Uso', 'value': 'hist_frequencia_uso'},  
                    {'label': 'Scatterplot Gasto x Tempo', 'value': 'scatter_gasto_tempo'}  
                ]),  
                dcc.Graph(id="grafico-3")  
            ], style={"width": "48%", "display": "inline-block"}),  

            html.Div([  
                dcc.Dropdown(id='dropdown-grafico-4', placeholder="Selecione o gráfico", options=[  
                    {'label': 'Histograma de Idade', 'value': 'hist_idade'},  
                    {'label': 'Boxplot de Gasto Total', 'value': 'box_gasto_total'},  
                    {'label': 'Heatmap de Correlação', 'value': 'heatmap_correlacao'},  
                    {'label': 'Frequência de Churn', 'value': 'frequencia_churn'},  
                    {'label': 'Boxplot de Tempo de Assinatura', 'value': 'box_tempo_assinatura'},  
                    {'label': 'Histograma de Frequência de Uso', 'value': 'hist_frequencia_uso'},  
                    {'label': 'Scatterplot Gasto x Tempo', 'value': 'scatter_gasto_tempo'}  
                ]),  
                dcc.Graph(id="grafico-4")  
            ], style={"width": "48%", "display": "inline-block"})  
        ]),  
        dcc.Tab(label="Previsão de Churn", children=[  
            html.H3("Previsão de Churn"),  
            dcc.Upload(  
                id="upload-previsao-data",  
                children=html.Button("Carregar Arquivo para Previsão (.csv ou .parquet)"),  
                multiple=False,  
                style={"margin": "20px"}  
            ),  
            html.Div(id="output-data-upload-previsao"),  
            html.Div([  
                dcc.Checklist(  
                    id='checklist-previsao',  
                    options=[{'label': col, 'value': col} for col in colunas_modelo],  
                    value=colunas_modelo[:4],  # Seleção inicial das primeiras 4 colunas  
                    labelStyle={'display': 'block'},  
                    style={"margin": "20px"}  
                ),  
                html.Button("Prever Churn", id="predict-button", n_clicks=0, className="btn btn-primary"),  
                html.Div(id="previsao-output")  
            ])  
        ])  
    ])  
], fluid=True)

# Função para parse de arquivos
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'parquet' in filename:
        df = pd.read_parquet(io.BytesIO(decoded))
    else:
        return None
    return df

# Callback para carregar arquivo e exibir mensagem de sucesso na aba "Importação e Tratamento"
@app.callback(  
    Output('output-data-upload', 'children'),  
    [Input('upload-data', 'contents')],  
    [State('upload-data', 'filename')]  
)  
def upload_file(contents, filename):  
    if contents is not None:  
        df = parse_contents(contents, filename)  
        if df is not None:  
            return f'Arquivo "{filename}" carregado com sucesso!'  
    return 'Nenhum arquivo carregado.'  

# Callback para carregar arquivo e exibir mensagem de sucesso na aba "Análise Exploratória"
@app.callback(  
    Output('output-data-upload-exploracao', 'children'),  
    [Input('upload-exploracao-data', 'contents')],  
    [State('upload-exploracao-data', 'filename')]  
)  
def upload_file_exploracao(contents, filename):  
    if contents is not None:  
        df = parse_contents(contents, filename)  
        if df is not None:  
            return f'Arquivo "{filename}" carregado com sucesso para Análise Exploratória!'  
    return 'Nenhum arquivo carregado na Análise Exploratória.'  

# Callback para salvar arquivo processado
@app.callback(
    Output('save-output', 'children'),
    [Input('save-button', 'n_clicks')],
    [State('upload-data', 'contents'),
     State('upload-data', 'filename')]
)
def save_processed_data(n_clicks, contents, filename):
    if n_clicks > 0 and contents:
        df = parse_contents(contents, filename)
        if df is not None:
            file_path = os.path.join(save_directory, f'processed_{filename}')
            df.to_csv(file_path, index=False)
            return f"Dados salvos em {file_path}"
    return ""

# Callback para atualizar os gráficos com base na seleção do usuário
@app.callback(  
    [Output('grafico-1', 'figure'),  
     Output('grafico-2', 'figure'),  
     Output('grafico-3', 'figure'),  
     Output('grafico-4', 'figure')],  
    [Input('dropdown-grafico-1', 'value'),  
     Input('dropdown-grafico-2', 'value'),  
     Input('dropdown-grafico-3', 'value'),  
     Input('dropdown-grafico-4', 'value')],  
    [State('upload-exploracao-data', 'contents')]  
)  
def update_graphs(graph_1, graph_2, graph_3, graph_4, contents):  
    if contents is None:  
        return {}, {}, {}, {}

    df = parse_contents(contents, 'file.csv')  

    graph_map = {
        'hist_idade': px.histogram(df, x="Age", title="Histograma de Idade"),
        'box_gasto_total': px.box(df, x="Total Spend", title="Boxplot de Gasto Total"),
        'heatmap_correlacao': px.imshow(df.corr(), title="Heatmap de Correlação"),
    }

    return (
        graph_map.get(graph_1, {}),
        graph_map.get(graph_2, {}),
        graph_map.get(graph_3, {}),
        graph_map.get(graph_4, {}),
    )

# Função para prever churn
def predict_churn(df):
    df_model = df[colunas_modelo[:4]]  # As primeiras 4 colunas para a previsão
    df_scaled = scaler.transform(df_model)
    predictions = modelo.predict(df_scaled)
    return predictions

# Callback para previsão de churn
@app.callback(  
    Output('previsao-output', 'children'),  
    [Input('predict-button', 'n_clicks')],  
    [State('upload-previsao-data', 'contents')]  
)  
def predict_churn_callback(n_clicks, contents):  
    if n_clicks > 0 and contents:  
        df = parse_contents(contents, 'file.csv')  
        predictions = predict_churn(df)  
        df['Churn Prediction'] = predictions  
        output_path = os.path.join(predictions_directory, 'previsao_churn.csv')  
        df.to_csv(output_path, index=False)  
        return f'Previsões salvas em {output_path}'  
    return 'Nenhum arquivo carregado para previsão.'

if __name__ == '__main__':
    app.run_server(debug=True)
