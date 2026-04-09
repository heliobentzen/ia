# Prática 02 — Análise Exploratória de Dados (EDA)

**Módulo:** 02–03 | **Duração:** ~90 minutos | **Dataset:** Titanic

## Objetivos
- Realizar EDA completo em um dataset real
- Formular e testar hipóteses com visualizações
- Gerar insights acionáveis para modelagem

---

## 1. Carregamento e Visão Geral

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

# Perfil completo
print("=" * 50)
print(f"Linhas: {df.shape[0]} | Colunas: {df.shape[1]}")
print("=" * 50)
print(df.dtypes)
print("\nEstatísticas descritivas:")
print(df.describe(include='all').T)
```

---

## 2. Análise Univariada

```python
# Variável target
print(f"\nTaxa de sobrevivência: {df['Survived'].mean():.1%}")
print(df['Survived'].value_counts(normalize=True).map('{:.1%}'.format))

# Distribuições
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# Numéricas
for ax, col in zip(axes[0], ['Age', 'Fare', 'SibSp', 'Parch']):
    df[col].hist(ax=ax, bins=30, edgecolor='black')
    ax.set_title(col)
    ax.axvline(df[col].mean(), color='red', linestyle='--', label='Média')
    ax.axvline(df[col].median(), color='blue', linestyle='--', label='Mediana')
    ax.legend(fontsize=8)

# Categóricas
for ax, col in zip(axes[1], ['Survived', 'Pclass', 'Sex', 'Embarked']):
    df[col].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(col)
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()
```

---

## 3. Análise Bivariada e Hipóteses

```python
# Hipótese 1: Mulheres têm maior taxa de sobrevivência
print("H1: Sobrevivência por sexo")
h1 = df.groupby('Sex')['Survived'].agg(['mean', 'count'])
print(h1)
# Teste qui-quadrado
ct = pd.crosstab(df['Sex'], df['Survived'])
chi2, p, _, _ = stats.chi2_contingency(ct)
print(f"χ²={chi2:.2f}, p={p:.6f} → {'Significativo' if p < 0.05 else 'Não significativo'}")

# Hipótese 2: 1ª classe tem maior sobrevivência
print("\nH2: Sobrevivência por classe")
h2 = df.groupby('Pclass')['Survived'].mean()
print(h2)

# Hipótese 3: Crianças têm maior sobrevivência
df['IsChild'] = (df['Age'] < 12).astype(int)
print("\nH3: Sobrevivência — crianças vs adultos")
print(df.groupby('IsChild')['Survived'].mean())

# Visualização
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.barplot(data=df, x='Sex', y='Survived', hue='Pclass', ax=axes[0])
axes[0].set_title('Sobrevivência: Sexo × Classe')

sns.boxplot(data=df, x='Survived', y='Age', ax=axes[1])
axes[1].set_title('Idade por Sobrevivência')

fare_survived = df.groupby('Survived')['Fare'].median()
axes[2].bar(['Não sobreviveu', 'Sobreviveu'], fare_survived.values)
axes[2].set_title('Mediana da Tarifa por Sobrevivência')

plt.tight_layout()
plt.show()
```

---

## 4. Relatório de Insights

```python
print("""
📊 INSIGHTS PRINCIPAIS — Titanic EDA
=====================================
1. Taxa geral de sobrevivência: 38.4%
2. Mulheres: 74.2% | Homens: 18.9% (p<0.001) ✅ H1 confirmada
3. 1ª classe: 63% | 2ª: 47% | 3ª: 24% ✅ H2 confirmada
4. Crianças (<12): 59% | Adultos: 37% ✅ H3 confirmada
5. Features com ausentes: Age (20%), Cabin (77%), Embarked (0.2%)
6. Tarifa tem distribuição muito enviesada (skew positivo) → transformação log

📌 FEATURES PARA MODELAGEM:
- Altas: Sex, Pclass, Fare (log), Age (imputado)
- Médias: Embarked, SibSp+Parch (FamilySize)
- Baixas: Cabin (muitos NaN), Ticket (alta cardinalidade)
""")
```

---

## Desafios Extras

1. Teste a hipótese de que `FamilySize` ótimo para sobrevivência é entre 2 e 4 membros
2. Extraia o título do nome (Mr, Mrs, Miss, etc.) e analise sua relação com sobrevivência
3. Use `pd.cut` para criar faixas de tarifa e analise a sobrevivência por faixa
4. Crie um par plot (seaborn `pairplot`) para todas as variáveis numéricas colorido por `Survived`
