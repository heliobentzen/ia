# Prática 01 — Python para Data Science

**Módulo:** 01 — Fundamentos de IA | **Duração:** ~90 minutos

## Objetivos
- Configurar o ambiente Python para ML
- Dominar operações essenciais com NumPy e Pandas
- Criar visualizações com Matplotlib e Seaborn

## Pré-requisitos
- Python instalado (>= 3.10)
- Jupyter Lab ou Google Colab

---

## Parte 1 — NumPy

```python
import numpy as np

# Criação de arrays
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 4))
c = np.ones((2, 3)) * 5
d = np.linspace(0, 1, 11)
e = np.random.randn(100)

print(f"Forma: {a.shape} | Tipo: {a.dtype}")
print(f"Média: {e.mean():.4f} | Std: {e.std():.4f}")

# Álgebra linear
A = np.random.randn(4, 3)
B = np.random.randn(3, 5)
C = A @ B  # multiplicação matricial
print(f"A @ B shape: {C.shape}")

# Broadcasting
X = np.random.randn(100, 5)
X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
print(f"Médias após normalização: {X_normalized.mean(axis=0).round(10)}")

# Indexação avançada
idx = np.where(e > 0)
print(f"Valores positivos: {len(idx[0])}")
```

---

## Parte 2 — Pandas

```python
import pandas as pd
import numpy as np

# Carregar dataset Titanic
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

print(f"Shape: {df.shape}")
print(df.head())
print(df.info())
print(df.describe())

# Seleção e filtragem
sobreviventes = df[df['Survived'] == 1]
mulheres_1classe = df[(df['Sex'] == 'female') & (df['Pclass'] == 1)]

# GroupBy
taxa_sobrevivencia = df.groupby(['Pclass', 'Sex'])['Survived'].mean().unstack()
print("\nTaxa de sobrevivência por classe e sexo:")
print(taxa_sobrevivencia.round(3))

# Criação de features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')

# Valores ausentes
print("\nValores ausentes:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Preenchimento
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
```

---

## Parte 3 — Visualização

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', palette='muted')
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Taxa de sobrevivência por classe
sns.barplot(data=df, x='Pclass', y='Survived', ax=axes[0,0])
axes[0,0].set_title('Sobrevivência por Classe')

# 2. Distribuição da idade por sobrevivência
sns.histplot(data=df, x='Age', hue='Survived', bins=30, ax=axes[0,1])
axes[0,1].set_title('Distribuição da Idade')

# 3. Boxplot: tarifa por classe
sns.boxplot(data=df, x='Pclass', y='Fare', ax=axes[0,2])
axes[0,2].set_yscale('log')
axes[0,2].set_title('Tarifa por Classe (log)')

# 4. Heatmap de correlação
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=axes[1,0])
axes[1,0].set_title('Mapa de Correlação')

# 5. Countplot: sobrevivência por sexo
sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[1,1])
axes[1,1].set_title('Sobrevivência por Sexo')

# 6. Scatter: Idade vs Tarifa
sns.scatterplot(data=df, x='Age', y='Fare', hue='Survived',
                alpha=0.6, ax=axes[1,2])
axes[1,2].set_title('Idade vs Tarifa')

plt.tight_layout()
plt.savefig('visualizacoes.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Desafios Extras

1. **Calcule** a mediana da tarifa para cada combinação de sexo e classe
2. **Crie** uma coluna `AgeGroup` com categorias: Criança (<12), Adolescente (12-17), Adulto (18-60), Idoso (>60)
3. **Plote** a taxa de sobrevivência por `AgeGroup` e `Sex`
4. **Calcule** a correlação de Pearson entre `Fare` e `Survived` — interprete o resultado
5. **Exporte** o DataFrame limpo para CSV: `df.to_csv('titanic_limpo.csv', index=False)`

## Referências
- Documentação NumPy: numpy.org
- Documentação Pandas: pandas.pydata.org
- Seaborn Gallery: seaborn.pydata.org/examples
