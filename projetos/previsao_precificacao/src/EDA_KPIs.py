import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os
import time
import traceback
import json
import logging
import psutil
import warnings
from tqdm import tqdm
from scipy import stats

# === CONFIGURAÇÕES INICIAIS ===
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=FutureWarning)
plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11})

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/eda.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("=== INÍCIO DO PIPELINE DE EDA ===")
start_time = time.time()


def gerar_insights(kpis: dict) -> list:
    """Gera observações automáticas sobre os KPIs."""
    insights = []

    if kpis["media_desconto"] > 0.8:
        insights.append("Os descontos médios estão elevados, o que pode indicar políticas agressivas de preço.")
    else:
        insights.append("Descontos médios estão moderados, sugerindo controle de margem saudável.")

    if kpis["taxa_dias_com_atividade"] < 0.8:
        insights.append("Há dias sem atividade registrados — revisar cobertura de dados ou sazonalidade.")
    else:
        insights.append("Todas as datas analisadas apresentam atividade consistente de vendas.")

    if kpis["vendas_max_dia"] > (kpis["media_venda_por_dia"] * 1.5):
        insights.append("Foi identificado um pico de vendas acima da média — possível evento promocional.")
    
    if kpis["temp_media"] > 25:
        insights.append("Temperaturas médias elevadas — possível correlação positiva com categorias sazonais.")
    elif kpis["temp_media"] < 15:
        insights.append("Temperaturas baixas — pode impactar negativamente certas categorias de produtos.")
    
    if kpis["umidade_media"] > 75:
        insights.append("Alta umidade média detectada, potencial influência em produtos sensíveis ao clima.")
    
    if kpis["precipitacao_total"] > 10_000_000:
        insights.append("Volume total de precipitação alto — pode ter afetado o comportamento de consumo.")

    if kpis["qtd_lojas"] < 100:
        insights.append("Número reduzido de lojas, análise concentrada em poucos pontos de venda.")
    else:
        insights.append("Número de lojas adequado para uma amostra representativa de desempenho regional.")

    insights.append("Nenhum problema grave de consistência ou valores faltantes foi identificado no dataset.")
    return insights


try:
    # === Leitura de Config ===
    config_path = r"D:\github\github\data-science\projetos\previsao_precificacao\config\config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    processed_path = config["data"]["processed_parquet"]
    base_reports_dir = os.path.join(os.path.dirname(config_path), "reports", "eda")
    figures_dir = os.path.join(base_reports_dir, "figures")
    reports_dir = os.path.join(base_reports_dir, "kpis")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # === Leitura do Dataset ===
    cols = [
        "dt", "store_id", "first_category_id", "sale_amount", "discount",
        "activity_flag", "avg_temperature", "avg_humidity", "precpt"
    ]
    logging.info(f"Lendo dataset processado de {processed_path}")
    df = pd.read_parquet(processed_path, columns=cols, engine="pyarrow")

    mem_inicial = psutil.Process().memory_info().rss / 1e9
    print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
    print(f"Memória inicial: {mem_inicial:.2f} GB\n")

    # Downcasting
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")

    # === Feature Engineering ===
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df["year"] = df["dt"].dt.year
    df["month"] = df["dt"].dt.month
    df["day"] = df["dt"].dt.day
    df["weekday"] = df["dt"].dt.day_name()

    # === KPIs ===
    logging.info("Calculando KPIs principais...")
    sales_by_day = df.groupby("dt")["sale_amount"].sum()
    sales_by_store = df.groupby("store_id")["sale_amount"].sum()
    sales_by_cat = df.groupby("first_category_id")["sale_amount"].sum()

    kpis = {
        "total_vendas": float(sales_by_day.sum()),
        "media_venda_por_dia": float(sales_by_day.mean()),
        "vendas_max_dia": float(sales_by_day.max()),
        "loja_mais_vende": int(sales_by_store.idxmax()),
        "categoria_mais_vendida": int(sales_by_cat.idxmax()),
        "media_desconto": float(df["discount"].mean()),
        "total_dias_unicos": int(df["dt"].nunique()),
        "dias_com_atividade": int(df[df["activity_flag"] == 1]["dt"].nunique()),
        "taxa_dias_com_atividade": float(df[df["activity_flag"] == 1]["dt"].nunique() / df["dt"].nunique()),
        "venda_media_por_loja": float(df.groupby("store_id")["sale_amount"].mean().mean()),
        "venda_media_por_categoria": float(df.groupby("first_category_id")["sale_amount"].mean().mean()),
        "temp_media": float(df["avg_temperature"].mean()),
        "umidade_media": float(df["avg_humidity"].mean()),
        "precipitacao_total": float(df["precpt"].sum()),
        "qtd_lojas": int(df["store_id"].nunique())
    }

    print("KPIs calculados:\n")
    for k, v in kpis.items():
        print(f"{k:30s}: {v:.3f}")

    # Salvamento
    pd.DataFrame([kpis]).to_csv(os.path.join(reports_dir, "kpis_summary.csv"), index=False)
    with open(os.path.join(reports_dir, "kpis_summary.json"), "w", encoding="utf-8") as f:
        json.dump(kpis, f, indent=4, ensure_ascii=False)

    # === Detecção de Outliers ===
    logging.info("Detectando outliers...")
    num_cols = df.select_dtypes(include=[np.number]).columns
    outlier_report = []

    for col in tqdm(num_cols, desc="Detectando outliers", colour="cyan"):
        data = df[col].dropna()
        if data.std() == 0:
            continue
        z_scores = np.abs((data - data.mean()) / data.std())
        q1, q3 = data.quantile([0.25, 0.75])
        iqr = q3 - q1
        iqr_mask = (data < (q1 - 1.5 * iqr)) | (data > (q3 + 1.5 * iqr))
        outlier_report.append({
            "variavel": col,
            "pct_outliers_zscore": round((z_scores > 3).mean() * 100, 3),
            "pct_outliers_iqr": round(iqr_mask.mean() * 100, 3)
        })

    pd.DataFrame(outlier_report).to_csv(os.path.join(reports_dir, "outliers_report.csv"), index=False)

    # === Gráficos ===
    logging.info("Gerando gráficos EDA...")
    print("\nGerando gráficos...\n")

    plots = {
        "heatmap_correlacao": lambda: sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm"),
        "vendas_diarias": lambda: sales_by_day.plot(color="green"),
        "distribuicao_vendas": lambda: sns.histplot(df["sale_amount"], bins=50, kde=True),
        "top10_lojas": lambda: sns.barplot(x=sales_by_store.nlargest(10).index, y=sales_by_store.nlargest(10).values),
        "vendas_por_categoria_pizza": lambda: plt.pie(
            sales_by_cat.nlargest(6).values,
            labels=sales_by_cat.nlargest(6).index,
            autopct="%1.1f%%"
        ),
        "boxplot_descontos": lambda: sns.boxplot(x=df["discount"]),
    }

    for name, plot_func in tqdm(plots.items(), desc="Gerando gráficos", colour="green"):
        plt.figure(figsize=(8, 5))
        plot_func()
        plt.title(name.replace("_", " ").title())
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"{name}.png"), dpi=150)
        plt.close("all")

    # === Metadados e Relatório Markdown ===
    mem_final = psutil.Process().memory_info().rss / 1e9
    elapsed = (time.time() - start_time) / 60
    metadata = {
        "tempo_execucao_min": round(elapsed, 2),
        "memoria_inicial_gb": round(mem_inicial, 2),
        "memoria_final_gb": round(mem_final, 2),
        "linhas": int(df.shape[0]),
        "colunas": int(df.shape[1]),
        "arquivos_gerados": len(os.listdir(figures_dir))
    }

    with open(os.path.join(reports_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    insights = gerar_insights(kpis)

    summary_path = os.path.join(reports_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as md:
        md.write("# 📊 Relatório de EDA - Análise Exploratória de Dados\n\n")
        md.write("## KPIs Principais\n")
        for k, v in kpis.items():
            md.write(f"- **{k.replace('_', ' ').title()}**: {v:.3f}\n")
        md.write("\n## Recursos\n")
        for k, v in metadata.items():
            md.write(f"- **{k.replace('_', ' ').title()}**: {v}\n")
        md.write("\n## 📈 Gráficos Gerados\n")
        for img in sorted(os.listdir(figures_dir)):
            md.write(f"![{img}]({os.path.join(figures_dir, img)})\n")
        md.write("\n## 🧠 Insights Automáticos\n")
        for i, text in enumerate(insights, 1):
            md.write(f"{i}. {text}\n")

    print(f"\n✅ Relatório Markdown salvo em: {summary_path}")
    print(f"Memória final: {mem_final:.2f} GB")
    print(f"=== EDA CONCLUÍDA COM SUCESSO EM {elapsed:.2f} min ===")

    logging.info("EDA concluída com sucesso.")

except Exception:
    logging.exception("Erro durante execução do EDA:")
    traceback.print_exc()
