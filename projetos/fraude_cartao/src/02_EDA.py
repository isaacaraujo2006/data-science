import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import mannwhitneyu

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Diretório para salvar figuras
FIGURES_DIR = "D:/github/data-science/projetos/fraude_cartao/reports/figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

def test_statistical_diff(df, col, target='class'):
    group0 = df[df[target] == 0][col]
    group1 = df[df[target] == 1][col]
    stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
    return p

df = pd.read_csv("D:/github/data-science/projetos/fraude_cartao/data/processed/processed.csv")

print("=== INÍCIO DO RELATÓRIO DE EDA - INSIGHTS ===\n")

# 1) Proporção das classes
fraude_counts = df['class'].value_counts()
total = fraude_counts.sum()
fraude_pct = fraude_counts / total * 100
print(f"1) Proporção das classes:\n   Não fraude: {fraude_counts[0]} ({fraude_pct[0]:.2f}%)\n   Fraude: {fraude_counts[1]} ({fraude_pct[1]:.4f}%)\n")

plt.figure(figsize=(7,7))
plt.pie(fraude_counts, labels=['Não Fraude', 'Fraude'], autopct='%1.2f%%', startangle=140, colors=['#66b3ff','#ff6666'])
plt.title("Proporção de Fraude vs Não Fraude")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "proporcao_classe_pizza.png"))
plt.close()

# 2) Histograma e violin plot do 'amount' por classe
plt.figure()
sns.histplot(df, x='amount', hue='class', bins=50, kde=True, multiple='stack', palette=['#66b3ff','#ff6666'])
plt.title("Distribuição do Valor das Transações (Amount) por Classe")
plt.xlim(0, df['amount'].quantile(0.99))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "hist_amount_classe.png"))
plt.close()

plt.figure()
sns.violinplot(x='class', y='amount', data=df, palette=['#66b3ff','#ff6666'])
plt.title("Violin Plot do Valor das Transações (Amount) por Classe")
plt.ylim(0, df['amount'].quantile(0.99))
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "violin_amount_classe.png"))
plt.close()

# 3) Contagem de transações por hora do dia, classe e outlier
plt.figure(figsize=(14,6))
sns.countplot(x='hour_of_day', hue='class', data=df, palette=['#66b3ff','#ff6666'])
plt.title("Volume de Transações por Hora do Dia (Fraude x Não Fraude)")
plt.xlabel("Hora do Dia")
plt.ylabel("Número de Transações")
plt.legend(['Não Fraude', 'Fraude'])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "transacoes_por_hora_classe.png"))
plt.close()

plt.figure(figsize=(14,6))
sns.countplot(x='hour_of_day', hue='outlier_amount', data=df, palette=['#66b3ff','#ff6666'])
plt.title("Volume de Transações por Hora do Dia (Outlier Amount)")
plt.xlabel("Hora do Dia")
plt.ylabel("Número de Transações")
plt.legend(['Normal', 'Outlier'])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "transacoes_por_hora_outlier.png"))
plt.close()

# 4) Heatmap da correlação
plt.figure(figsize=(14,12))
corr = df.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
plt.title('Matriz de Correlação')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "heatmap_correlacao.png"))
plt.close()

# 5) Treemap interativo das transações por hora do dia e classe (salvo como HTML, não abre navegador)
fig = px.treemap(df, path=['hour_of_day', 'class'], values='amount',
                 color='class', color_continuous_scale=['#66b3ff', '#ff6666'],
                 title='Treemap: Volume de Transações por Hora do Dia e Classe')
fig.write_html(os.path.join(FIGURES_DIR, "treemap_hora_classe.html"))

# 6) Scatter plot do 'amount' vs 'time' colorido por classe (amostra 5k)
fig = px.scatter(df.sample(5000), x='time', y='amount', color='class',
                 title='Scatter Plot de Amount vs Time (amostra 5k)',
                 color_discrete_map={0:'#66b3ff', 1:'#ff6666'}, opacity=0.6)
fig.write_html(os.path.join(FIGURES_DIR, "scatter_amount_time.html"))

# 7) Distribuição de 'outlier_amount' por classe
plt.figure()
sns.countplot(x='outlier_amount', hue='class', data=df, palette=['#66b3ff','#ff6666'])
plt.title('Contagem de Outliers em Amount por Classe')
plt.xlabel('Outlier Amount (0=Não, 1=Sim)')
plt.ylabel('Número de Transações')
plt.legend(['Não Fraude', 'Fraude'])
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "outliers_amount_por_classe.png"))
plt.close()

# 8) Boxplots para algumas features numéricas (exemplo: 'v1', 'v2', 'v3')
features_box = ['v1','v2','v3']
for f in features_box:
    plt.figure()
    sns.boxplot(x='class', y=f, data=df, palette=['#66b3ff','#ff6666'])
    plt.title(f'Boxplot da feature {f} por Classe')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, f'boxplot_{f}_classe.png'))
    plt.close()

print("✅ Todos os gráficos foram salvos na pasta 'reports/figures/' sem abrir janelas ou navegador.")
