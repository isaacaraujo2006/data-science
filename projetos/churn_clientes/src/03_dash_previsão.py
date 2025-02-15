import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import pandas as pd
import joblib
import base64
import io
import os

# Definindo os caminhos
models_dir = 'C:/Github/data-science/projetos/churn_clientes/models'
preprocessors_dir = 'C:/Github/data-science/projetos/churn_clientes/preprocessors'
predictions_dir = 'C:/Github/data-science/projetos/churn_clientes/predictions'

# Caminhos dos arquivos de modelo e scaler
model_path = os.path.join(models_dir, 'logistic_regression_model.joblib')
scaler_path = os.path.join(preprocessors_dir, 'scaler.joblib')

# Verificar se o arquivo do scaler existe
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {scaler_path}")

# Carregar o modelo treinado e o scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Criar a aplicação Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Previsão de Churn de Clientes"),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Carregar Arquivo (.csv ou .parquet)'),
        multiple=False
    ),
    html.Div(id='output-data-upload'),
    html.Button('Realizar Previsão', id='predict-button', n_clicks=0),
    dcc.Loading(
        id='loading',
        type='default',
        children=html.Div(id='output-prediction')
    )
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
    return df

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('output-prediction', 'children')],
    [Input('upload-data', 'contents'),
     Input('predict-button', 'n_clicks')],
    [State('upload-data', 'filename'),
     State('upload-data', 'contents')]
)
def update_output(contents, n_clicks, filename, contents_state):
    if contents is not None:
        df = parse_contents(contents, filename)
        if df is not None:
            children = f'Arquivo {filename} carregado com sucesso!'
            if n_clicks > 0:
                # Preprocessar os dados
                df_encoded = pd.get_dummies(df, drop_first=True)
                
                # Ajustar colunas ausentes devido à codificação One-Hot
                missing_cols = set(scaler.feature_names_in_) - set(df_encoded.columns)
                for col in missing_cols:
                    df_encoded[col] = 0
                df_encoded = df_encoded[scaler.feature_names_in_]
                
                df_encoded = scaler.transform(df_encoded)
                
                # Fazer previsões
                predictions = model.predict(df_encoded)
                df['previsao_churn'] = predictions
                
                # Salvar previsões
                if not os.path.exists(predictions_dir):
                    os.makedirs(predictions_dir)
                predictions_path = os.path.join(predictions_dir, 'predictions.csv')
                df.to_csv(predictions_path, index=False)
                
                # Gerar relatório de previsão
                num_total = len(df)
                num_churn = df['previsao_churn'].sum()
                perc_churn = (num_churn / num_total) * 100
                
                relatorio = (f"### Relatório de Previsão ###\n\n"
                             f"Total de registros: {num_total}\n"
                             f"Total de churn previstos: {num_churn}\n"
                             f"Percentual de churn previstos: {perc_churn:.2f}%\n\n"
                             f"Previsões realizadas e salvas em: {predictions_path}")
                
                return children, html.Pre(relatorio)
            return children, ''
    return 'Falha no carregamento do arquivo.', ''

if __name__ == '__main__':
    app.run_server(debug=True)
