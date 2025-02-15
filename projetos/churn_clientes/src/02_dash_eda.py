import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import plotly.express as px
import base64
import io
import yaml

# Carregar o arquivo de configuração
with open(r'C:/Github/data-science/projetos/churn_clientes/config/config.yaml', 'r', encoding='utf-8') as file:
    config = yaml.safe_load(file)

# Traduzir os nomes das colunas
traducao_colunas = {
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

# Criar a aplicação Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Análise de Churn de Clientes"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Carregar Arquivo (.csv ou .parquet)'),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Div([
        dcc.Dropdown(
            id='dropdown-grafico-1',
            options=[],
            placeholder='Selecione o gráfico 1'
        ),
        dcc.Graph(id='grafico-1')
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Dropdown(
            id='dropdown-grafico-2',
            options=[],
            placeholder='Selecione o gráfico 2'
        ),
        dcc.Graph(id='grafico-2')
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Dropdown(
            id='dropdown-grafico-3',
            options=[],
            placeholder='Selecione o gráfico 3'
        ),
        dcc.Graph(id='grafico-3')
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Dropdown(
            id='dropdown-grafico-4',
            options=[],
            placeholder='Selecione o gráfico 4'
        ),
        dcc.Graph(id='grafico-4')
    ], style={'width': '48%', 'display': 'inline-block'})
])

def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    if 'csv' in filename:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    elif 'parquet' in filename:
        df = pd.read_parquet(io.BytesIO(decoded))
    else:
        return None
    df.rename(columns=traducao_colunas, inplace=True)
    return df

@app.callback(
    Output('output-data-upload', 'children'),
    Output('dropdown-grafico-1', 'options'),
    Output('dropdown-grafico-2', 'options'),
    Output('dropdown-grafico-3', 'options'),
    Output('dropdown-grafico-4', 'options'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            options = [
                {'label': 'Histograma de Idade', 'value': 'hist_idade'},
                {'label': 'Boxplot de Gasto Total', 'value': 'box_gasto_total'},
                {'label': 'Heatmap de Correlação', 'value': 'heatmap_correlacao'},
                {'label': 'Frequência de Churn', 'value': 'frequencia_churn'},
                {'label': 'Boxplot de Tempo de Assinatura', 'value': 'box_tempo_assinatura'},
                {'label': 'Histograma de Frequência de Uso', 'value': 'hist_frequencia_uso'},
                {'label': 'Scatterplot Gasto Total x Tempo de Assinatura', 'value': 'scatter_gasto_tempo'}
            ]
            return f'Arquivo {filename} carregado com sucesso!', options, options, options, options
    return 'Falha no carregamento do arquivo.', [], [], [], []

def gerar_grafico(df, tipo_grafico):
    if tipo_grafico == 'hist_idade':
        fig = px.histogram(df, x='idade', nbins=30, title='Histograma de Idade', height=350)
    elif tipo_grafico == 'box_gasto_total':
        fig = px.box(df, y='gasto_total', title='Boxplot de Gasto Total', height=350)
    elif tipo_grafico == 'heatmap_correlacao':
        correlation_matrix = df.corr(numeric_only=True)
        fig = px.imshow(correlation_matrix, text_auto=True, title='Heatmap de Correlação', height=350)
    elif tipo_grafico == 'frequencia_churn':
        churn_counts = df['churn'].value_counts(normalize=True) * 100
        churn_counts = churn_counts.reset_index()
        churn_counts.columns = ['churn', 'percentual']
        fig = px.bar(churn_counts, x='churn', y='percentual', title='Frequência de Churn', height=350)
    elif tipo_grafico == 'box_tempo_assinatura':
        fig = px.box(df, y='tempo_de_assinatura', title='Boxplot de Tempo de Assinatura', height=350)
    elif tipo_grafico == 'hist_frequencia_uso':
        fig = px.histogram(df, x='frequencia_uso', nbins=30, title='Histograma de Frequência de Uso', height=350)
    elif tipo_grafico == 'scatter_gasto_tempo':
        fig = px.scatter(df, x='tempo_de_assinatura', y='gasto_total', title='Scatterplot Gasto Total x Tempo de Assinatura', height=350)
    return fig

@app.callback(
    Output('grafico-1', 'figure'),
    Output('grafico-2', 'figure'),
    Output('grafico-3', 'figure'),
    Output('grafico-4', 'figure'),
    Input('dropdown-grafico-1', 'value'),
    Input('dropdown-grafico-2', 'value'),
    Input('dropdown-grafico-3', 'value'),
    Input('dropdown-grafico-4', 'value'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_graphs(grafico1, grafico2, grafico3, grafico4, contents, filename):
    if contents is not None and filename is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            fig1 = gerar_grafico(df, grafico1) if grafico1 else {}
            fig2 = gerar_grafico(df, grafico2) if grafico2 else {}
            fig3 = gerar_grafico(df, grafico3) if grafico3 else {}
            fig4 = gerar_grafico(df, grafico4) if grafico4 else {}
            return fig1, fig2, fig3, fig4
    return {}, {}, {}, {}

if __name__ == '__main__':
    app.run_server(debug=True)
