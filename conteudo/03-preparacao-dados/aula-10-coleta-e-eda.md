# Aula 10 — Coleta de Dados e Análise Exploratória (EDA)

> **Módulo 03 · Aula 10 · Carga horária: 1 hora-aula**

---

## Objetivos da Aula

- Conhecer as principais fontes de dados para projetos de ML.
- Classificar tipos de variáveis e entender suas implicações.
- Conduzir uma EDA sistemática com estatísticas descritivas e visualizações.
- Detectar problemas iniciais: ausentes, outliers, desbalanceamento.

---

## 1. Fontes de Dados

### 1.1 APIs e Serviços Web

```python
import requests
import pandas as pd

# Exemplo: API pública (dados de clima)
resp = requests.get("https://api.open-meteo.com/v1/forecast",
    params={"latitude": -23.55, "longitude": -46.63,
            "hourly": "temperature_2m", "forecast_days": 3})
dados = resp.json()
df_clima = pd.DataFrame({
    "hora": dados["hourly"]["time"],
    "temperatura": dados["hourly"]["temperature_2m"]
})
df_clima["hora"] = pd.to_datetime(df_clima["hora"])
print(df_clima.head())
```

### 1.2 Web Scraping

```python
# pip install requests beautifulsoup4
import requests
from bs4 import BeautifulSoup

url = "https://books.toscrape.com/"
soup = BeautifulSoup(requests.get(url).text, "html.parser")
livros = []
for article in soup.select("article.product_pod"):
    livros.append({
        "titulo": article.h3.a["title"],
        "preco": article.select_one(".price_color").text.strip(),
        "rating": article.p["class"][1]
    })
df_livros = pd.DataFrame(livros)
print(df_livros.head())
```

### 1.3 Datasets Públicos

| Fonte | URL | Características |
|-------|-----|----------------|
| Kaggle | kaggle.com/datasets | +50k datasets, competições |
| UCI ML Repository | archive.ics.uci.edu | Clássicos do ML |
| HuggingFace Datasets | huggingface.co/datasets | NLP, visão, áudio |
| IBGE | ibge.gov.br/microdados | Dados brasileiros oficiais |
| dados.gov.br | dados.gov.br | Dados abertos do governo BR |
| Google Dataset Search | datasetsearch.research.google.com | Busca universal |

```python
# HuggingFace Datasets
from datasets import load_dataset
ds = load_dataset("imdb", split="train")
df_imdb = ds.to_pandas()
print(df_imdb.head())
print(f"Shape: {df_imdb.shape}")

# Kaggle via API
# kaggle datasets download -d titanic
df_titanic = pd.read_csv("train.csv")
```

### 1.4 Bancos de Dados

```python
import sqlite3
import sqlalchemy as sa

# SQLite
conn = sqlite3.connect("dados.db")
df_sql = pd.read_sql("SELECT * FROM clientes WHERE ativo = 1", conn)

# PostgreSQL via SQLAlchemy
engine = sa.create_engine("postgresql://user:senha@localhost:5432/meubanco")
df_pg = pd.read_sql("SELECT * FROM vendas LIMIT 10000", engine)
```

---

## 2. Tipos de Variáveis

| Tipo | Subtipo | Exemplos | Tratamento |
|------|---------|---------|-----------|
| **Numérica** | Contínua | Altura, salário, temperatura | StandardScaler, normalização |
| **Numérica** | Discreta | Número de filhos, contagem | Tratar como numérica ou ordinal |
| **Categórica** | Nominal | Estado, cor, país | One-Hot Encoding |
| **Categórica** | Ordinal | Grau de escolaridade, rating | Ordinal Encoding |
| **Texto** | — | Comentários, artigos | TF-IDF, embeddings |
| **Temporal** | — | Data de nascimento, timestamp | Extração de features temporais |
| **Binária** | — | Sim/Não, 0/1 | Direto ou LabelEncoding |

```python
# Identificar tipos automaticamente
df = pd.read_csv("dataset.csv")

numericas   = df.select_dtypes(include=['number']).columns.tolist()
categoricas = df.select_dtypes(include=['object', 'category']).columns.tolist()
temporais   = df.select_dtypes(include=['datetime']).columns.tolist()

print(f"Numéricas ({len(numericas)}):   {numericas}")
print(f"Categóricas ({len(categoricas)}): {categoricas}")
print(f"Temporais ({len(temporais)}):   {temporais}")

# Converter tipos
df['data_nascimento'] = pd.to_datetime(df['data_nascimento'])
df['categoria'] = df['categoria'].astype('category')
```

---

## 3. EDA Sistemática com o Dataset Titanic

### 3.1 Carregamento e Inspeção Inicial

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Carregar Titanic
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
# Alternativamente: df = pd.read_csv("titanic.csv")

print("=== INSPEÇÃO INICIAL ===")
print(f"Shape: {df.shape}")
print(f"\nTipos de dados:\n{df.dtypes}")
print(f"\nPrimeiras linhas:\n{df.head()}")
print(f"\nÚltimas linhas:\n{df.tail()}")

# Resumo rápido
print(f"\n=== RESUMO RÁPIDO ===")
print(f"Duplicatas: {df.duplicated().sum()}")
print(f"\nValores ausentes:\n{df.isnull().sum()}")
print(f"\nPorcentagem ausente:\n{(df.isnull().mean() * 100).round(2)}")
```

### 3.2 Estatísticas Descritivas

```python
print("\n=== ESTATÍSTICAS DESCRITIVAS ===")

# Variáveis numéricas
desc_num = df.describe().T
desc_num['cv'] = desc_num['std'] / desc_num['mean']  # Coeficiente de variação
print("\nNuméricas:")
print(desc_num[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'cv']].round(3))

# Distribuição do target
print(f"\n=== TARGET: Survived ===")
print(df['Survived'].value_counts())
print(f"Taxa de sobrevivência: {df['Survived'].mean():.1%}")

# Estatísticas por grupo (target)
print("\n=== ESTATÍSTICAS POR CLASSE DE SOBREVIVÊNCIA ===")
print(df.groupby('Survived')[['Age', 'Fare', 'SibSp', 'Parch']].agg(['mean','median','std']).round(2))

# Skewness e Kurtosis
print("\n=== ASSIMETRIA E CURTOSE ===")
for col in ['Age', 'Fare', 'SibSp']:
    s = df[col].dropna()
    print(f"{col:12s} — Skewness: {s.skew():.3f} | Kurtosis: {s.kurtosis():.3f}")
    if abs(s.skew()) > 1:
        print(f"  ⚠️  Alta assimetria: considerar transformação logarítmica")
```

### 3.3 Análise de Variáveis Categóricas

```python
print("\n=== VARIÁVEIS CATEGÓRICAS ===")

categoricas_titanic = ['Survived', 'Pclass', 'Sex', 'Embarked']
for col in categoricas_titanic:
    print(f"\n{col}:")
    freq = df[col].value_counts(normalize=True) * 100
    print(freq.round(1).to_string())
```

### 3.4 Visualizações EDA

```python
fig = plt.figure(figsize=(16, 14))
fig.suptitle('EDA — Dataset Titanic', fontsize=16, y=1.01)

# 1. Distribuição de idade por sobrevivência
ax1 = fig.add_subplot(3, 3, 1)
for sobrev, cor, label in [(0,'coral','Não sobreviveu'), (1,'steelblue','Sobreviveu')]:
    df[df['Survived']==sobrev]['Age'].dropna().plot.hist(
        ax=ax1, bins=25, alpha=0.6, color=cor, label=label, density=True
    )
ax1.set(title='Distribuição de Idade', xlabel='Idade', ylabel='Densidade')
ax1.legend(fontsize=8)

# 2. Taxa de sobrevivência por sexo
ax2 = fig.add_subplot(3, 3, 2)
taxa_sexo = df.groupby('Sex')['Survived'].mean()
taxa_sexo.plot(kind='bar', ax=ax2, color=['coral','steelblue'], rot=0)
ax2.set(title='Taxa de Sobrevivência por Sexo', ylabel='Taxa', ylim=(0,1))
for i, v in enumerate(taxa_sexo):
    ax2.text(i, v+0.02, f'{v:.1%}', ha='center', fontsize=10)

# 3. Taxa de sobrevivência por classe
ax3 = fig.add_subplot(3, 3, 3)
taxa_classe = df.groupby('Pclass')['Survived'].mean()
taxa_classe.plot(kind='bar', ax=ax3, color=['gold','silver','saddlebrown'], rot=0)
ax3.set(title='Sobrevivência por Classe', xlabel='Classe', ylabel='Taxa')
for i, v in enumerate(taxa_classe):
    ax3.text(i, v+0.02, f'{v:.1%}', ha='center', fontsize=10)

# 4. Boxplot Fare por sobrevivência
ax4 = fig.add_subplot(3, 3, 4)
df.boxplot(column='Fare', by='Survived', ax=ax4)
ax4.set(title='Tarifa vs Sobrevivência', xlabel='Survived', ylabel='Fare')
plt.sca(ax4)
plt.title('Tarifa vs Sobrevivência')

# 5. Correlação (numéricas)
ax5 = fig.add_subplot(3, 3, 5)
num_cols = ['Survived','Pclass','Age','SibSp','Parch','Fare']
corr = df[num_cols].corr()
sns.heatmap(corr, ax=ax5, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1)
ax5.set_title('Correlação (Pearson)')

# 6. Distribuição de Fare (log scale)
ax6 = fig.add_subplot(3, 3, 6)
fare_log = np.log1p(df['Fare'])
ax6.hist(df['Fare'].dropna(), bins=50, alpha=0.5, label='Original', color='coral')
ax6_twin = ax6.twinx()
ax6_twin.hist(fare_log.dropna(), bins=50, alpha=0.5, label='Log(1+Fare)', color='steelblue')
ax6.set(title='Fare: Original vs Log', xlabel='Valor')
ax6.legend(loc='upper right')
ax6_twin.legend(loc='upper center')

# 7. Missing values heatmap
ax7 = fig.add_subplot(3, 3, 7)
missing = df.isnull()
sns.heatmap(missing.T, ax=ax7, cbar=False, cmap='viridis', yticklabels=True)
ax7.set_title('Mapa de Valores Ausentes')
ax7.set_xlabel('Amostras')

# 8. Violin plot: Idade por Classe e Sexo
ax8 = fig.add_subplot(3, 3, 8)
df_nonan = df.dropna(subset=['Age'])
sns.violinplot(data=df_nonan, x='Pclass', y='Age', hue='Sex',
               split=True, ax=ax8, palette=['coral','steelblue'])
ax8.set(title='Idade por Classe e Sexo')

# 9. Taxa de sobrevivência por faixa de idade
ax9 = fig.add_subplot(3, 3, 9)
df['faixa_etaria'] = pd.cut(df['Age'], bins=[0,12,18,35,60,100],
                             labels=['Criança','Adolescente','Adulto','Meia-idade','Idoso'])
taxa_faixa = df.groupby('faixa_etaria', observed=True)['Survived'].mean()
taxa_faixa.plot(kind='bar', ax=ax9, color='steelblue', rot=30)
ax9.set(title='Sobrevivência por Faixa Etária', ylabel='Taxa')

plt.tight_layout()
plt.savefig("titanic_eda.png", dpi=150, bbox_inches='tight')
plt.show()
print("EDA salvo: titanic_eda.png")
```

### 3.5 Análise de Correlação Avançada

```python
from scipy.stats import pointbiserialr, chi2_contingency

print("\n=== CORRELAÇÃO COM O TARGET ===")

# Variáveis numéricas: correlação ponto-biserial com target binário
print("\nCorrelação Ponto-Biserial (numéricas vs Survived):")
for col in ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']:
    dados_limpos = df[[col, 'Survived']].dropna()
    r, p = pointbiserialr(dados_limpos['Survived'], dados_limpos[col])
    sig = "✅" if p < 0.05 else "❌"
    print(f"  {col:10s}: r={r:+.4f}, p={p:.4f} {sig}")

# Variáveis categóricas: qui-quadrado com target
print("\nTeste Qui-Quadrado (categóricas vs Survived):")
for col in ['Sex', 'Pclass', 'Embarked']:
    tabela = pd.crosstab(df[col], df['Survived'])
    chi2, p, dof, _ = chi2_contingency(tabela)
    sig = "✅" if p < 0.05 else "❌"
    print(f"  {col:10s}: chi2={chi2:.2f}, p={p:.6f}, dof={dof} {sig}")
```

### 3.6 Pair Plot e Análise Multivariada

```python
# Pair plot com coloração por target
colunas_pair = ['Age', 'Fare', 'Pclass', 'Survived']
df_pair = df[colunas_pair].dropna()
df_pair['Survived_label'] = df_pair['Survived'].map({0: 'Não sobreviveu', 1: 'Sobreviveu'})

g = sns.pairplot(df_pair, hue='Survived_label', diag_kind='kde',
                  plot_kws={'alpha': 0.5, 's': 20},
                  palette={'Não sobreviveu': 'coral', 'Sobreviveu': 'steelblue'})
g.fig.suptitle('Pair Plot — Titanic', y=1.02)
plt.savefig("titanic_pairplot.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

## 4. Análise Rápida com ydata-profiling

```python
# pip install ydata-profiling
from ydata_profiling import ProfileReport

# Gera relatório HTML interativo completo em 1 linha
perfil = ProfileReport(df, title="EDA — Titanic", explorative=True)
perfil.to_file("relatorio_titanic.html")
print("Relatório gerado: relatorio_titanic.html")

# Configurações úteis
perfil_minimo = ProfileReport(df, minimal=True)  # mais rápido
perfil_minimo.to_file("relatorio_minimo.html")
```

---

## 5. Checklist de EDA

| Passo | Verificar |
|-------|-----------|
| ✅ Shape e tipos | `df.shape`, `df.dtypes`, `df.info()` |
| ✅ Valores ausentes | `df.isnull().sum()`, heatmap |
| ✅ Duplicatas | `df.duplicated().sum()` |
| ✅ Distribuição do target | `df[target].value_counts()`, desbalanceamento? |
| ✅ Estatísticas descritivas | `df.describe()`, skewness, kurtosis |
| ✅ Distribuições das features | histogramas, boxplots |
| ✅ Correlações | Pearson, Spearman, qui-quadrado, ponto-biserial |
| ✅ Outliers | boxplots, z-score, IQR |
| ✅ Relações com target | boxplot/violin por classe, taxa por categoria |
| ✅ Pair plot | relações entre pares de variáveis |

---

## 6. Exercícios

1. **EDA Completa**: Baixe o dataset `Heart Disease` do UCI ML Repository. Realize a EDA completa seguindo o checklist acima.
2. **ydata-profiling**: Gere relatórios para o Titanic com `explorative=True` e `minimal=True`. Compare as diferenças.
3. **Correlação**: No Titanic, qual variável tem maior correlação com `Survived`? Justifique com métricas.
4. **Desbalanceamento**: Pesquise técnicas para tratar desbalanceamento de classes (SMOTE, class_weight, undersampling). Quando cada uma é mais adequada?

---

## 7. Referências

- GÉRON, Aurélien. *Mãos à Obra*. 3. ed. Alta Books, 2023. Cap. 2.
- FACELI et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. 2. ed. LTC, 2021. Cap. 3.
- ydata-profiling Documentation. [https://docs.profiling.ydata.ai](https://docs.profiling.ydata.ai)

---

*← [Módulo 02](../02-paradigmas-aprendizado/README.md) | [Aula 11 — Qualidade e Limpeza →](aula-11-qualidade-limpeza.md)*
