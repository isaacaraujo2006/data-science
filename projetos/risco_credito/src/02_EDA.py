import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import mannwhitneyu

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Diretório para salvar figuras
FIGURES_DIR = "D:/github/data-science/projetos/risco_credito/reports/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

def test_statistical_diff(df, col, target='inadimplente_mes_seguinte'):
    group0 = df[df[target] == 0][col]
    group1 = df[df[target] == 1][col]
    stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
    return p

# Carregar dados processados
df = pd.read_parquet("D:/github/data-science/projetos/risco_credito/data/processed/processed.parquet")

print("=== INÍCIO DO RELATÓRIO DE EDA - INSIGHTS ===\n")

# 1) Proporção das classes inadimplentes
inadimplencia_counts = df['inadimplente_mes_seguinte'].value_counts()
total = inadimplencia_counts.sum()
inadimplencia_pct = inadimplencia_counts / total * 100
print(f"1) Proporção das classes:\n"
      f"   Não inadimplente: {inadimplencia_counts.get(0,0)} ({inadimplencia_pct.get(0,0):.2f}%)\n"
      f"   Inadimplente: {inadimplencia_counts.get(1,0)} ({inadimplencia_pct.get(1,0):.2f}%)\n")

plt.figure(figsize=(7,7))
plt.pie(inadimplencia_counts, labels=['Não Inadimplente', 'Inadimplente'],
        autopct='%1.2f%%', startangle=140, colors=['#66b3ff','#ff6666'])
plt.title("Proporção de Inadimplência")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "proporcao_inadimplencia_pizza.png"))
plt.close()

# 2) Histograma e violin plot de limite_credito por inadimplência
plt.figure()
sns.histplot(df, x='limite_credito', hue='inadimplente_mes_seguinte', bins=50, kde=True, multiple='stack', palette=['#66b3ff','#ff6666'])
plt.title("Distribuição do Limite de Crédito por Inadimplência")
plt.xlim(df['limite_credito'].quantile(0.01), df['limite_credito'].quantile(0.99))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "hist_limite_credito_inadimplencia.png"))
plt.close()

plt.figure()
sns.violinplot(x='inadimplente_mes_seguinte', y='limite_credito', data=df, palette=['#66b3ff','#ff6666'])
plt.title("Violin Plot do Limite de Crédito por Inadimplência")
plt.ylim(df['limite_credito'].quantile(0.01), df['limite_credito'].quantile(0.99))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "violin_limite_credito_inadimplencia.png"))
plt.close()

# 3) Análise de idade por inadimplência
plt.figure()
sns.histplot(df, x='idade', hue='inadimplente_mes_seguinte', bins=30, kde=True, multiple='stack', palette=['#66b3ff','#ff6666'])
plt.title("Distribuição da Idade por Inadimplência")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "hist_idade_inadimplencia.png"))
plt.close()

plt.figure()
sns.violinplot(x='inadimplente_mes_seguinte', y='idade', data=df, palette=['#66b3ff','#ff6666'])
plt.title("Violin Plot da Idade por Inadimplência")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "violin_idade_inadimplencia.png"))
plt.close()

# 4) Boxplots para variáveis numéricas selecionadas por classe
features_box = ['limite_credito', 'idade', 'pagamento_mes_0', 'pagamento_mes_1', 'pagamento_mes_2',
                'fatura_mes_1', 'fatura_mes_2']

for f in features_box:
    plt.figure()
    sns.boxplot(x='inadimplente_mes_seguinte', y=f, data=df, palette=['#66b3ff','#ff6666'])
    plt.title(f'Boxplot da feature {f} por Inadimplência')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'boxplot_{f}_inadimplencia.png'))
    plt.close()

# 5) Matriz de correlação heatmap
plt.figure(figsize=(14,12))
corr = df.corr(numeric_only=True)  # evitar FutureWarning
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "heatmap_correlacao.png"))
plt.close()

# 6) Testes estatísticos Mann-Whitney para features numéricas
print("\n6) Testes estatísticos Mann-Whitney para diferenças entre classes (inadimplente=0 vs 1):")
for col in features_box:
    p_val = test_statistical_diff(df, col)
    print(f"   - {col}: p-value = {p_val:.4g}")

# 7) Gráfico interativo treemap (com correção para tipos)
df['sexo_str'] = df['sexo'].astype(str).copy()
df['inadimplente_str'] = df['inadimplente_mes_seguinte'].astype(str).copy()

fig = px.treemap(
    df.copy(),
    path=['sexo_str', 'inadimplente_str'],
    values='limite_credito',
    color='inadimplente_str',
    color_discrete_map={'0': '#66b3ff', '1': '#ff6666'},
    title='Treemap: Limite de Crédito por Sexo e Inadimplência'
)
fig.write_html(os.path.join(FIGURES_DIR, "treemap_limite_credito_sexo_inadimplencia.html"))

# 8) Scatter plot interativo idade x limite_credito com cor por inadimplência (amostra 5k)
fig = px.scatter(df.sample(5000), x='idade', y='limite_credito', color='inadimplente_mes_seguinte',
                 title='Scatter Plot: Idade vs Limite de Crédito (amostra 5k)',
                 color_continuous_scale=px.colors.sequential.RdBu,
                 opacity=0.6)
fig.write_html(os.path.join(FIGURES_DIR, "scatter_idade_limite_credito.html"))

print("\n✅ Todos os gráficos foram salvos na pasta 'reports/figures/' sem abrir janelas ou navegador.")
