# EDA.py
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
import argparse
import sys

# =====================================================
# CONFIG
# =====================================================
DEFAULT_DATA = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/data/processed/processed.parquet")
DEFAULT_FIGURES_DIR = Path(r"D:/github/data-science/projetos/previsao_cesta_demanda/reports/figures")

# =====================================================
# HELPERS
# =====================================================
def save_fig(fig, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

def save_plotly(fig, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(filename))

# =====================================================
# MAIN EDA PIPELINE
# =====================================================
def run_eda(data_path: Path, figures_dir: Path):
    if not data_path.exists():
        print(f"[ERRO] Arquivo processado não encontrado: {data_path}")
        sys.exit(1)

    print(f"[OK] Carregando dataset: {data_path}")
    df = pd.read_parquet(data_path)
    print(f"[INFO] Linhas: {df.shape[0]:,} | Colunas: {df.shape[1]:,}")

    # =====================================================
    # Estatísticas gerais
    # =====================================================
    print("\n[INFO] Estatísticas descritivas:")
    print(df.describe(include="all").transpose().head(20))

    # =====================================================
    # Distribuição numéricas
    # =====================================================
    num_cols = ["price", "freight_value", "product_weight_g", "product_volume_cm3"]

    for col in num_cols:
        if col in df.columns:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df[col], bins=50, kde=True, ax=ax)
            ax.set_title(f"Distribuição - {col}")
            save_fig(fig, figures_dir / f"hist_{col}.png")

            fig_int = px.histogram(df, x=col, nbins=50, title=f"Distribuição - {col}")
            save_plotly(fig_int, figures_dir / f"hist_{col}.html")
            print(f"[OK] Histograma salvo: {col}")

    # =====================================================
    # Top categorias
    # =====================================================
    if "product_category_name_english" in df.columns:
        top_cat = df["product_category_name_english"].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x=top_cat.values, y=top_cat.index, palette="viridis", ax=ax)
        ax.set_title("Top 15 categorias mais vendidas")
        save_fig(fig, figures_dir / "top_categories.png")

        fig_int = px.bar(top_cat[::-1], x=top_cat[::-1].values, y=top_cat[::-1].index,
                         orientation="h", title="Top 15 categorias mais vendidas")
        save_plotly(fig_int, figures_dir / "top_categories.html")
        print("[OK] Gráfico salvo: top_categories")

    # =====================================================
    # Evolução temporal (mensal, semanal e diária)
    # =====================================================
    if "order_purchase_timestamp" in df.columns:
        df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])

        # Mensal
        sales_month = df.groupby([df["order_purchase_timestamp"].dt.to_period("M")]).size().reset_index(name="sales")
        sales_month["date"] = sales_month["order_purchase_timestamp"].dt.to_timestamp()

        fig, ax = plt.subplots(figsize=(12,6))
        sns.lineplot(data=sales_month, x="date", y="sales", ax=ax)
        ax.set_title("Evolução das vendas (mensal)")
        save_fig(fig, figures_dir / "sales_trend_month.png")
        save_plotly(px.line(sales_month, x="date", y="sales", title="Evolução das vendas (mensal)"),
                    figures_dir / "sales_trend_month.html")

        # Semanal
        sales_week = df.groupby([df["order_purchase_timestamp"].dt.to_period("W")]).size().reset_index(name="sales")
        sales_week["date"] = sales_week["order_purchase_timestamp"].dt.to_timestamp()

        fig, ax = plt.subplots(figsize=(12,6))
        sns.lineplot(data=sales_week, x="date", y="sales", ax=ax)
        ax.set_title("Evolução das vendas (semanal)")
        save_fig(fig, figures_dir / "sales_trend_week.png")
        save_plotly(px.line(sales_week, x="date", y="sales", title="Evolução das vendas (semanal)"),
                    figures_dir / "sales_trend_week.html")

        # Diária
        sales_day = df.groupby([df["order_purchase_timestamp"].dt.to_period("D")]).size().reset_index(name="sales")
        sales_day["date"] = sales_day["order_purchase_timestamp"].dt.to_timestamp()

        fig, ax = plt.subplots(figsize=(14,6))
        sns.lineplot(data=sales_day, x="date", y="sales", ax=ax)
        ax.set_title("Evolução das vendas (diária)")
        save_fig(fig, figures_dir / "sales_trend_day.png")
        save_plotly(px.line(sales_day, x="date", y="sales", title="Evolução das vendas (diária)"),
                    figures_dir / "sales_trend_day.html")

        print("[OK] Gráficos de vendas salvos (mensal, semanal, diária)")

    # =====================================================
    # Review score + por categoria
    # =====================================================
    if "review_score" in df.columns:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(x="review_score", data=df, palette="magma", ax=ax)
        ax.set_title("Distribuição das notas de review")
        save_fig(fig, figures_dir / "review_score.png")
        save_plotly(px.histogram(df, x="review_score", title="Distribuição das notas de review"),
                    figures_dir / "review_score.html")

        if "product_category_name_english" in df.columns:
            rev_cat = df.groupby("product_category_name_english")["review_score"].mean().sort_values(ascending=False).head(15)
            fig, ax = plt.subplots(figsize=(12,6))
            sns.barplot(x=rev_cat.values, y=rev_cat.index, palette="coolwarm", ax=ax)
            ax.set_title("Top 15 categorias por média de review")
            save_fig(fig, figures_dir / "review_score_category.png")
            save_plotly(px.bar(rev_cat[::-1], x=rev_cat[::-1].values, y=rev_cat[::-1].index,
                               orientation="h", title="Top 15 categorias por média de review"),
                        figures_dir / "review_score_category.html")

            print("[OK] Review score por categoria salvo")

    # =====================================================
    # SLA por estado
    # =====================================================
    if "sla_diff_days" in df.columns and "customer_state" in df.columns:
        sla_state = df.groupby("customer_state")["sla_diff_days"].mean().sort_values(ascending=False).reset_index()

        fig, ax = plt.subplots(figsize=(12,6))
        sns.barplot(x="sla_diff_days", y="customer_state", data=sla_state, palette="rocket", ax=ax)
        ax.set_title("Média do SLA por estado")
        save_fig(fig, figures_dir / "sla_by_state.png")
        save_plotly(px.bar(sla_state, x="sla_diff_days", y="customer_state",
                           orientation="h", title="Média do SLA por estado"),
                    figures_dir / "sla_by_state.html")
        print("[OK] SLA por estado salvo")

    # =====================================================
    # Mapa de clientes (lat/lng)
    # =====================================================
    if {"lat_mean_customer", "lng_mean_customer"}.issubset(df.columns):
        sample_map = df.dropna(subset=["lat_mean_customer", "lng_mean_customer"]).sample(5000, random_state=42)
        fig_int = px.scatter_mapbox(
            sample_map, lat="lat_mean_customer", lon="lng_mean_customer",
            zoom=3, height=600, opacity=0.5,
            title="Distribuição geográfica de clientes"
        )
        fig_int.update_layout(mapbox_style="open-street-map")
        save_plotly(fig_int, figures_dir / "map_customers.html")
        print("[OK] Mapa de clientes salvo (HTML interativo)")

    # =====================================================
    # Pagamentos
    # =====================================================
    if "payment_types" in df.columns:
        pay_types = df["payment_types"].value_counts().reset_index()
        pay_types.columns = ["payment_type", "count"]

        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x="payment_type", y="count", data=pay_types, palette="Set2", ax=ax)
        ax.set_title("Distribuição dos tipos de pagamento")
        save_fig(fig, figures_dir / "payment_types.png")
        save_plotly(px.bar(pay_types, x="payment_type", y="count", title="Distribuição dos tipos de pagamento"),
                    figures_dir / "payment_types.html")

        if "payment_installments_max" in df.columns:
            fig, ax = plt.subplots(figsize=(8,4))
            sns.histplot(df["payment_installments_max"], bins=20, ax=ax)
            ax.set_title("Distribuição do nº de parcelas")
            save_fig(fig, figures_dir / "payment_installments.png")
            save_plotly(px.histogram(df, x="payment_installments_max", nbins=20,
                                     title="Distribuição do nº de parcelas"),
                        figures_dir / "payment_installments.html")

        print("[OK] Gráficos de pagamentos salvos")

    print("\n[FINALIZADO] EDA concluído. Gráficos salvos em:", figures_dir)

# =====================================================
# ENTRY POINT
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA, help="Caminho para o parquet processado")
    parser.add_argument("--figures", default=DEFAULT_FIGURES_DIR, help="Diretório para salvar figuras")
    args = parser.parse_args()

    run_eda(Path(args.data), Path(args.figures))

if __name__ == "__main__":
    main()
