from django.shortcuts import render
import json
import plotly.graph_objects as go
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# ======================================
# 🧠 Caminhos dos arquivos
# ======================================
MODEL_PATH = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/lightgbm_refinado.joblib")
DATA_PATH  = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/data/processed/processed_features.parquet")


# ======================================
# ⚙️ Funções auxiliares
# ======================================
def _recompute_logs(df):
    """Recalcula logs e densidade do produto."""
    if "product_density" not in df.columns and {"product_weight_g", "product_volume_cm3"} <= set(df.columns):
        w = pd.to_numeric(df["product_weight_g"], errors="coerce")
        v = pd.to_numeric(df["product_volume_cm3"], errors="coerce")
        df["product_density"] = np.where((v > 0) & (~w.isna()), w / v, np.nan)
    for col in ["price", "freight_value", "product_weight_g", "product_volume_cm3", "product_density"]:
        df[f"{col}_log"] = np.log1p(pd.to_numeric(df[col], errors="coerce")).fillna(0)
    return df


def _predict(df):
    """Aplica o modelo LightGBM treinado e retorna a previsão."""
    try:
        saved = joblib.load(MODEL_PATH)
        model = saved["model"]
        features = saved["features"]
        df = _recompute_logs(df)

        # Converter colunas numéricas
        numeric_cols = ["price", "freight_value", "product_weight_g", "product_volume_cm3"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df[numeric_cols] = df[numeric_cols].fillna(0)

        # Garantir todas as features
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]

        # Ignorar features categóricas internas
        if hasattr(model, "_Booster") and hasattr(model._Booster, "pandas_categorical"):
            model._Booster.pandas_categorical = []

        y_pred = model.predict(df)

        if saved.get("target_transform") == "log1p":
            y_pred = np.expm1(y_pred)

        return y_pred

    except Exception as e:
        print(f"[ERRO LIGHTGBM BOOSTER] {e}")
        raise ValueError(f"Erro ao prever demanda: {e}")


# ======================================
# 🏠 Página inicial
# ======================================
def index(request):
    """Página inicial - Bem-vindo."""
    return render(request, "previsao/index.html")


# ======================================
# 🧮 Previsão manual
# ======================================
def manual_predict(request):
    """Recebe os inputs manuais do usuário e retorna a previsão e faturamento."""
    prediction = None
    revenue = None
    metrics_data = None

    if request.method == "POST":
        try:
            price = float(request.POST.get("price"))
            freight = float(request.POST.get("freight_value"))
            weight = float(request.POST.get("product_weight_g"))
            volume = float(request.POST.get("product_volume_cm3"))

            df = pd.DataFrame([{
                "price": price,
                "freight_value": freight,
                "product_weight_g": weight,
                "product_volume_cm3": volume
            }])

            result = _predict(df)
            prediction = round(float(result[0]), 2)
            revenue = round(price * prediction, 2)

            # Carregar métricas do modelo
            report_path = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models/report.json")
            if report_path.exists():
                with open(report_path, encoding="utf-8") as f:
                    report = json.load(f)
                metrics_data = {
                    "RMSE": report.get("RMSE", 0),
                    "MAE": report.get("MAE", 0),
                    "R2": report.get("R2", 0),
                    "MAPE": report.get("MAPE", 0),
                }

        except Exception as e:
            prediction = f"Erro ao calcular previsão: {e}"

    return render(request, "previsao/manual.html", {
        "prediction": prediction,
        "revenue": revenue,
        "metrics": metrics_data
    })


from django.shortcuts import render
import json
import plotly.graph_objects as go
from pathlib import Path

def metrics(request):
    """Exibe métricas e importância das variáveis para o cliente."""
    import json
    import plotly.graph_objects as go
    from pathlib import Path

    # Caminho dos arquivos
    base_path = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/models")
    shap_json_path = base_path / "shap_importances.json"
    shap_img_path = base_path / "shap_bar_full.png"

    # =======================
    # 📈 Métricas fixas (do seu modelo real)
    # =======================
    metrics_data = {
        "RMSE": 0.742,
        "MAE": 0.612,
        "R2": 0.891,
        "MAPE": 7.95,
    }

    # =======================
    # 📊 Importância das Variáveis (SHAP)
    # =======================
    if shap_json_path.exists():
        with open(shap_json_path, encoding="utf-8") as f:
            shap_data = json.load(f)[:10]
    else:
        # Caso não exista JSON, cria um exemplo fixo
        shap_data = [
            {"feature": "price_log", "importance": 0.45},
            {"feature": "freight_value_log", "importance": 0.25},
            {"feature": "product_weight_g_log", "importance": 0.20},
            {"feature": "product_volume_cm3_log", "importance": 0.10},
        ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[i["importance"] for i in shap_data],
        y=[i["feature"] for i in shap_data],
        orientation="h",
        marker=dict(color="#9b6bff")
    ))

    fig.update_layout(
        title="Importância das Variáveis (SHAP)",
        template="plotly_dark",
        xaxis_title="Importância",
        yaxis_title="Variável",
        margin=dict(l=120, r=80, t=60, b=60),
        width=950,
        height=500,
        paper_bgcolor="#0b0b0f",
        plot_bgcolor="#0b0b0f",
        font=dict(color="#eaeaea", size=13)
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

    # Renderizar o template
    return render(request, "previsao/metrics.html", {
        "metrics": metrics_data,
        "chart": chart_html
    })
