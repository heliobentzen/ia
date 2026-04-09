# Aula 05 — Python para IA e Ciência de Dados

> **Módulo:** 01 — Fundamentos de Inteligência Artificial  
> **Duração:** 45 minutos  
> **Pré-requisitos:** Python básico (funções, listas, dicionários); Aulas 01–04

---

## Objetivos de Aprendizagem

Ao final desta aula, o estudante será capaz de:

1. **Configurar** um ambiente Python reproduzível para projetos de IA com Anaconda/conda.
2. **Manipular** arrays N-dimensionais com NumPy, incluindo broadcasting e vetorização.
3. **Analisar** dados tabulares com Pandas: leitura, filtragem, agregação e limpeza.
4. **Criar** visualizações informativas com Matplotlib e Seaborn.
5. **Utilizar** a API do Scikit-learn para treinar e avaliar modelos de ML.
6. **Estruturar** projetos de ciência de dados com Jupyter Notebooks de forma organizada.

---

## 1. Por que Python para IA?

### 1.1 O Ecossistema Python

Python tornou-se a **língua franca da IA e ciência de dados** por razões concretas:

| Fator | Detalhe |
|-------|---------|
| **Sintaxe legível** | Código próximo a pseudocódigo; prototipagem rápida |
| **Ecossistema rico** | NumPy, Pandas, Scikit-learn, TensorFlow, PyTorch, Hugging Face |
| **Comunidade enorme** | 2ª linguagem mais popular no GitHub; maior comunidade de DS/ML |
| **Interoperabilidade** | Bindings para C/C++ (NumPy), CUDA (CuPy), R (rpy2), Java (JPype) |
| **REPL e Notebooks** | Jupyter permite ciclos rápidos de exploração |
| **Gratuito e open-source** | Sem licenças; fácil de distribuir |

### 1.2 As Linguagens Concorrentes

| Linguagem | Onde é usada | Por que não domina IA? |
|-----------|-------------|------------------------|
| **R** | Estatística, bioinformática | Menor ecossistema de DL; menos geral |
| **Julia** | Computação científica de alto desempenho | Ecossistema menor; curva de adoção |
| **C++** | Produção de sistemas embarcados, jogos | Muito verboso para prototipagem |
| **Scala/Java** | Spark, sistemas distribuídos | Verbosidade; menos bibliotecas de ML |
| **MATLAB** | Engenharia, universidades | Proprietário; caro |

**Conclusão**: Python domina porque combina produtividade (alta abstração) com desempenho via extensões em C/C++ (NumPy, pandas, PyTorch usam código nativo internamente).

---

## 2. Configuração de Ambiente

### 2.1 Anaconda vs. Miniconda vs. pip + venv

```
Anaconda:  distribuição completa (~3GB), inclui Jupyter, Pandas, NumPy, etc.
           → Ideal para iniciantes, tudo pré-instalado

Miniconda: apenas o conda (~400MB), você instala o que precisa
           → Ideal para quem quer controle total e espaço em disco

pip + venv: ferramenta nativa do Python
           → Sem conda; menor compatibilidade com pacotes não-Python (BLAS, CUDA)
```

### 2.2 Criando o Ambiente do Curso

```bash
# Instalar Miniconda (baixe de: https://docs.conda.io/en/latest/miniconda.html)

# Criar ambiente para o curso
conda create -n ia-curso python=3.11 -y

# Ativar o ambiente
conda activate ia-curso

# Instalar pacotes científicos essenciais via conda-forge (mais estável)
conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn -y

# Instalar Jupyter e ferramentas de desenvolvimento
pip install jupyterlab notebook ipywidgets

# Para Deep Learning (Módulos 7-10):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# pip install tensorflow

# Verificar instalação
python -c "
import numpy as np
import pandas as pd
import matplotlib
import sklearn
print(f'NumPy:       {np.__version__}')
print(f'Pandas:      {pd.__version__}')
print(f'Matplotlib:  {matplotlib.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print('✓ Todas as bibliotecas instaladas com sucesso!')
"
```

### 2.3 Gerenciando Dependências com requirements.txt

```bash
# Exportar o ambiente atual
pip freeze > requirements.txt

# Ou via conda (inclui dependências do sistema)
conda env export > environment.yml

# Recrear o ambiente em outra máquina
# pip:
pip install -r requirements.txt

# conda:
conda env create -f environment.yml
```

**Arquivo `requirements.txt` típico para este curso:**
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyterlab>=4.0.0
notebook>=7.0.0
ipywidgets>=8.0.0
```

---

## 3. NumPy — Computação Numérica de Alto Desempenho

### 3.1 Por que NumPy?

```python
import numpy as np
import time

# Python puro vs NumPy — comparação de velocidade
tamanho = 1_000_000

# Python puro
lista = list(range(tamanho))
inicio = time.time()
resultado_python = [x * 2 for x in lista]
tempo_python = time.time() - inicio

# NumPy
array = np.arange(tamanho)
inicio = time.time()
resultado_numpy = array * 2
tempo_numpy = time.time() - inicio

print(f"Python puro: {tempo_python*1000:.2f}ms")
print(f"NumPy:       {tempo_numpy*1000:.2f}ms")
print(f"Speedup:     {tempo_python/tempo_numpy:.0f}x mais rápido!")
# Resultado típico: NumPy é ~50-200x mais rápido
```

**Por que NumPy é tão rápido?**
1. Arrays armazenados contiguamente na memória (cache-friendly).
2. Operações implementadas em C/Fortran compilado.
3. Suporte a SIMD (Single Instruction, Multiple Data) via BLAS/LAPACK.
4. Sem overhead de boxing/unboxing de objetos Python.

### 3.2 Arrays NumPy — Fundamentos

```python
import numpy as np

# ─── Criação de arrays ────────────────────────────────────────────────────────

# A partir de listas Python
a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3],
              [4, 5, 6]])           # array 2D (matriz)

print(f"a: {a}")           # [1 2 3 4 5]
print(f"b:\n{b}")
print(f"a.shape: {a.shape}")    # (5,)   — vetor de 5 elementos
print(f"b.shape: {b.shape}")    # (2, 3) — 2 linhas, 3 colunas
print(f"b.ndim:  {b.ndim}")     # 2      — número de dimensões
print(f"b.dtype: {b.dtype}")    # int64  — tipo dos elementos
print(f"b.size:  {b.size}")     # 6      — total de elementos

# Funções de criação especializadas
zeros  = np.zeros((3, 4))            # Matriz de zeros
uns    = np.ones((2, 3))             # Matriz de uns
identidade = np.eye(4)               # Matriz identidade 4×4
aleatorio = np.random.rand(3, 3)     # Valores aleatórios em [0, 1)
normal = np.random.randn(1000)       # Distribuição normal padrão
seq = np.arange(0, 10, 2)           # [0, 2, 4, 6, 8]
lin = np.linspace(0, 1, 11)         # 11 valores uniformes de 0 a 1

print(f"\nIdentidade 3×3:\n{np.eye(3)}")
print(f"seq: {seq}")
print(f"lin: {lin}")
```

### 3.3 Operações com Arrays

```python
import numpy as np

# ─── Aritmética elementar (element-wise) ─────────────────────────────────────
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([2.0, 2.0, 2.0, 2.0])

print("Operações element-wise:")
print(f"  a + b  = {a + b}")         # [3. 4. 5. 6.]
print(f"  a * b  = {a * b}")         # [2. 4. 6. 8.]
print(f"  a ** 2 = {a ** 2}")        # [1. 4. 9. 16.]
print(f"  a / b  = {a / b}")         # [0.5 1. 1.5 2.]
print(f"  np.sqrt(a) = {np.sqrt(a)}")  # [1. 1.41 1.73 2.]

# ─── Funções universais (ufuncs) ─────────────────────────────────────────────
angulos = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"\nSeno: {np.sin(angulos).round(4)}")
print(f"Cosseno: {np.cos(angulos).round(4)}")
print(f"Exponencial: {np.exp(np.array([0, 1, 2]))}")
print(f"Log natural: {np.log(np.array([1, np.e, np.e**2]))}")

# ─── Álgebra linear ───────────────────────────────────────────────────────────
A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(f"\nA:\n{A}")
print(f"\nB:\n{B}")
print(f"\nA @ B (multiplicação matricial):\n{A @ B}")  # [[19,22],[43,50]]
print(f"\nA.T (transposta):\n{A.T}")
print(f"\nnp.linalg.det(A) = {np.linalg.det(A):.1f}")  # -2.0
print(f"\nnp.linalg.inv(A):\n{np.linalg.inv(A)}")

# Decomposição de valores singulares (usada em PCA, recomendação)
U, S, Vt = np.linalg.svd(A)
print(f"\nSVD — Valores singulares: {S}")

# ─── Estatísticas ─────────────────────────────────────────────────────────────
dados = np.random.randn(1000)  # 1000 amostras de N(0,1)
print(f"\nEstatísticas de 1000 amostras N(0,1):")
print(f"  Média:      {dados.mean():.4f}  (esperado: 0)")
print(f"  Desvio pad. {dados.std():.4f}   (esperado: 1)")
print(f"  Mínimo:     {dados.min():.4f}")
print(f"  Máximo:     {dados.max():.4f}")
print(f"  Mediana:    {np.median(dados):.4f}")
print(f"  Percentil 95: {np.percentile(dados, 95):.4f}")
```

### 3.4 Broadcasting — O Superpoder do NumPy

```python
import numpy as np

# ─── Broadcasting: operações entre arrays de formas diferentes ───────────────
#
# Regras de broadcasting:
# 1. Arrays são alinhados a partir da dimensão mais à direita
# 2. Dimensões de tamanho 1 são "esticadas" para combinar com a outra
# 3. Se as formas são incompatíveis, há erro

# Exemplo 1: vetor + escalar (o mais simples)
v = np.array([1, 2, 3, 4, 5])
print(f"v + 10 = {v + 10}")  # [11 12 13 14 15] — escalar é "broadcast" para todos

# Exemplo 2: matriz + vetor linha (shapes: (3,4) + (4,) → (3,4))
matriz = np.ones((3, 4))
linha  = np.array([1, 2, 3, 4])
print(f"\nmatriz + linha =\n{matriz + linha}")
# [[2. 3. 4. 5.]
#  [2. 3. 4. 5.]
#  [2. 3. 4. 5.]]

# Exemplo 3: matriz + vetor coluna (shapes: (3,4) + (3,1) → (3,4))
coluna = np.array([[10], [20], [30]])  # shape: (3, 1)
print(f"\nmatriz + coluna =\n{matriz + coluna}")
# [[11. 11. 11. 11.]
#  [21. 21. 21. 21.]
#  [31. 31. 31. 31.]]

# Aplicação em ML: normalização Z-score via broadcasting
X = np.random.randn(100, 5)  # 100 amostras, 5 features
media = X.mean(axis=0)       # shape: (5,)
desvio = X.std(axis=0)       # shape: (5,)

# Broadcasting: (100,5) - (5,) e (100,5) / (5,)
X_normalizado = (X - media) / desvio

print(f"\nApós normalização Z-score:")
print(f"  Média das colunas: {X_normalizado.mean(axis=0).round(4)}")
print(f"  Desvio das colunas: {X_normalizado.std(axis=0).round(4)}")
```

### 3.5 Indexação e Fatiamento

```python
import numpy as np

A = np.arange(12).reshape(3, 4)
print(f"A:\n{A}")

# ─── Indexação básica ─────────────────────────────────────────────────────────
print(f"\nA[0, 0] = {A[0, 0]}")      # 0  (primeiro elemento)
print(f"A[2, 3] = {A[2, 3]}")        # 11 (último elemento)
print(f"A[-1, -1] = {A[-1, -1]}")    # 11 (mesmo elemento, índice negativo)

# ─── Fatiamento ───────────────────────────────────────────────────────────────
print(f"\nPrimeira linha:    {A[0, :]}")     # [0 1 2 3]
print(f"Última coluna:     {A[:, -1]}")      # [3 7 11]
print(f"Sub-matriz 2×2:   \n{A[:2, :2]}")   # [[0,1],[4,5]]
print(f"Passo 2:           {A[::2, ::2]}")  # [[0,2],[8,10]]

# ─── Indexação booleana (muito usada em ML!) ──────────────────────────────────
numeros = np.array([1, -2, 3, -4, 5, -6, 7])
positivos = numeros > 0
print(f"\nnumeros: {numeros}")
print(f"máscara: {positivos}")          # [T F T F T F T]
print(f"filtrado: {numeros[positivos]}") # [1 3 5 7]

# Substituição condicional
numeros_abs = numeros.copy()
numeros_abs[numeros_abs < 0] = 0
print(f"negativos → 0: {numeros_abs}")  # [1 0 3 0 5 0 7]

# ─── Indexação sofisticada (fancy indexing) ───────────────────────────────────
indices = np.array([0, 2, 4])
print(f"\nElementos nos índices {indices}: {numeros[indices]}")  # [1 3 5]
```

---

## 4. Pandas — Análise de Dados Tabulares

### 4.1 Series e DataFrames

```python
import pandas as pd
import numpy as np

# ─── Series: array unidimensional com rótulos ─────────────────────────────────
notas = pd.Series(
    [7.5, 8.0, 6.5, 9.0, 5.5],
    index=["Ana", "Bruno", "Carlos", "Diana", "Eduardo"],
    name="Notas_IA"
)
print("Series notas:")
print(notas)
print(f"\nMédia: {notas.mean():.2f}")
print(f"Aprovados (≥7): {(notas >= 7).sum()} alunos")

# ─── DataFrame: tabela 2D com rótulos ────────────────────────────────────────
dados = {
    "nome":     ["Ana", "Bruno", "Carlos", "Diana", "Eduardo"],
    "idade":    [22, 25, 23, 28, 21],
    "nota_ia":  [7.5, 8.0, 6.5, 9.0, 5.5],
    "nota_ml":  [8.0, 7.5, 7.0, 9.5, 6.0],
    "turma":    ["A", "B", "A", "B", "A"],
    "aprovado": [True, True, False, True, False],
}

df = pd.DataFrame(dados)
print("\nDataFrame de alunos:")
print(df.to_string(index=False))
print(f"\nShape: {df.shape}")  # (5, 6)
print(f"Dtypes:\n{df.dtypes}")
```

### 4.2 Leitura de Dados

```python
import pandas as pd

# ─── Leitura de diferentes formatos ──────────────────────────────────────────

# CSV
# df = pd.read_csv("dados.csv", sep=",", encoding="utf-8")
# df = pd.read_csv("dados.csv", sep=";", decimal=",",  # formato brasileiro
#                   encoding="latin-1", na_values=["N/A", "?", "-"])

# Excel
# df = pd.read_excel("dados.xlsx", sheet_name="Dados", header=0)

# JSON
# df = pd.read_json("dados.json", orient="records")

# SQL
# import sqlalchemy as sa
# engine = sa.create_engine("postgresql://user:pass@localhost/db")
# df = pd.read_sql("SELECT * FROM tabela", engine)

# Datasets embutidos do Seaborn (úteis para prática)
import seaborn as sns

iris     = sns.load_dataset("iris")
titanic  = sns.load_dataset("titanic")
penguins = sns.load_dataset("penguins")
tips     = sns.load_dataset("tips")

print("Dataset Iris:")
print(iris.head())
print(f"\nShape: {iris.shape}")
print(f"\nDescrição estatística:")
print(iris.describe().round(2))
print(f"\nValores nulos por coluna:\n{iris.isnull().sum()}")
```

### 4.3 Manipulação e Limpeza de Dados

```python
import pandas as pd
import numpy as np
import seaborn as sns

# Usamos o Titanic — dataset clássico com muitos valores nulos e tipos mistos
df = sns.load_dataset("titanic")

print(f"Shape original: {df.shape}")
print(f"\nValores nulos por coluna:")
print(df.isnull().sum().sort_values(ascending=False))

# ─── Seleção de colunas e linhas ──────────────────────────────────────────────
# Selecionar colunas
cols_interesse = df[["survived", "pclass", "sex", "age", "fare"]]

# Filtrar linhas com condição
sobreviventes = df[df["survived"] == 1]
mulheres = df[df["sex"] == "female"]
classe_1_sobreviventes = df[(df["pclass"] == 1) & (df["survived"] == 1)]

print(f"\nTotal de passageiros: {len(df)}")
print(f"Sobreviventes: {len(sobreviventes)} ({len(sobreviventes)/len(df):.1%})")
print(f"Mulheres: {len(mulheres)}")

# ─── Tratamento de valores nulos ─────────────────────────────────────────────
df_limpo = df.copy()

# Estratégia 1: Preencher com mediana (boa para numéricas assimétricas)
mediana_age = df_limpo["age"].median()
df_limpo["age"].fillna(mediana_age, inplace=True)

# Estratégia 2: Preencher com moda (boa para categóricas)
moda_embarked = df_limpo["embarked"].mode()[0]
df_limpo["embarked"].fillna(moda_embarked, inplace=True)

# Estratégia 3: Dropar coluna com muitos nulos (>50%)
pct_nulos = df_limpo.isnull().sum() / len(df_limpo)
colunas_remover = pct_nulos[pct_nulos > 0.5].index.tolist()
df_limpo.drop(columns=colunas_remover, inplace=True)
print(f"\nColunas removidas (>50% nulos): {colunas_remover}")

# ─── Engenharia de features básica ───────────────────────────────────────────
# Converter categórica para binária
df_limpo["is_female"] = (df_limpo["sex"] == "female").astype(int)

# Criar faixas etárias
df_limpo["faixa_etaria"] = pd.cut(
    df_limpo["age"],
    bins=[0, 12, 18, 35, 60, 100],
    labels=["criança", "adolescente", "adulto_jovem", "adulto", "idoso"]
)

print(f"\nDistribuição por faixa etária:")
print(df_limpo["faixa_etaria"].value_counts())

# ─── Agrupamentos e agregações ───────────────────────────────────────────────
taxa_sobrev_por_classe = (df.groupby("pclass")["survived"]
                           .agg(["sum", "count", "mean"])
                           .rename(columns={"sum": "sobreviveu",
                                            "count": "total",
                                            "mean": "taxa_sobrevivencia"}))
taxa_sobrev_por_classe["taxa_sobrevivencia"] = (
    taxa_sobrev_por_classe["taxa_sobrevivencia"].map("{:.1%}".format)
)

print("\nTaxa de sobrevivência por classe:")
print(taxa_sobrev_por_classe.to_string())

# Análise cruzada: sexo × classe × sobrevivência
crosstab = pd.crosstab(
    df["pclass"], df["sex"],
    values=df["survived"], aggfunc="mean"
).round(3)
print("\nTaxa de sobrevivência por classe × sexo:")
print(crosstab)

# ─── Merge/Join entre DataFrames ─────────────────────────────────────────────
# Criar tabelas separadas para demonstrar merge
alunos = pd.DataFrame({
    "id": [1, 2, 3, 4],
    "nome": ["Ana", "Bruno", "Carlos", "Diana"],
    "turma_id": [1, 2, 1, 3]
})

turmas = pd.DataFrame({
    "turma_id": [1, 2, 3],
    "nome_turma": ["ML Básico", "Deep Learning", "IA para Negócios"],
    "professor": ["Prof. Silva", "Prof. Santos", "Prof. Costa"]
})

# Inner join (apenas correspondências de ambos os lados)
merge_inner = pd.merge(alunos, turmas, on="turma_id", how="inner")
print("\nMerge alunos × turmas:")
print(merge_inner.to_string(index=False))
```

### 4.4 apply, map e operações vetorizadas

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    "nome": ["Ana Costa", "Bruno Lima", "carlos melo", "Diana Pires"],
    "nota1": [7.5, 8.0, 6.5, 9.0],
    "nota2": [8.5, 7.0, 7.5, 9.5],
})

# apply: aplica função a colunas/linhas
df["media"] = df[["nota1", "nota2"]].mean(axis=1)
df["aprovado"] = df["media"].apply(lambda x: "Aprovado" if x >= 7 else "Reprovado")
df["nome_titulo"] = df["nome"].apply(str.title)  # Capitaliza corretamente

# Vetorizado (mais eficiente que apply para operações simples)
df["media_norm"] = (df["media"] - df["media"].min()) / (df["media"].max() - df["media"].min())

print(df.to_string(index=False))
```

---

## 5. Matplotlib e Seaborn — Visualizações

```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd

# Configurações de estilo
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")

# Carrega dados para exemplos
iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")

# ─── Figura 1: Gráficos básicos do Matplotlib ────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Visualizações Básicas com Matplotlib", fontsize=16, fontweight="bold")

# 1. Gráfico de linha
x = np.linspace(0, 4 * np.pi, 200)
axes[0, 0].plot(x, np.sin(x), label="sin(x)", color="blue", linewidth=2)
axes[0, 0].plot(x, np.cos(x), label="cos(x)", color="red", linewidth=2, linestyle="--")
axes[0, 0].set_title("Funções Trigonométricas")
axes[0, 0].set_xlabel("x")
axes[0, 0].set_ylabel("f(x)")
axes[0, 0].legend()
axes[0, 0].axhline(y=0, color="k", linewidth=0.5)

# 2. Gráfico de dispersão
np.random.seed(42)
x_pts = np.random.randn(100)
y_pts = 2 * x_pts + np.random.randn(100) * 0.5
cores = np.abs(x_pts)  # cor baseada no valor de x
scatter = axes[0, 1].scatter(x_pts, y_pts, c=cores, cmap="viridis",
                               alpha=0.7, edgecolors="white", linewidth=0.5)
plt.colorbar(scatter, ax=axes[0, 1])
axes[0, 1].set_title("Dispersão com Correlação")
axes[0, 1].set_xlabel("X")
axes[0, 1].set_ylabel("Y = 2X + ruído")

# 3. Histograma
dados_norm = np.random.randn(1000)
axes[0, 2].hist(dados_norm, bins=40, color="steelblue", alpha=0.7,
                edgecolor="white", density=True)
x_norm = np.linspace(-4, 4, 200)
from scipy import stats as sp_stats
axes[0, 2].plot(x_norm, sp_stats.norm.pdf(x_norm), "r-", linewidth=2,
                label="N(0,1) teórica")
axes[0, 2].set_title("Histograma com PDF Teórica")
axes[0, 2].set_xlabel("Valor")
axes[0, 2].set_ylabel("Densidade")
axes[0, 2].legend()

# 4. Gráfico de barras
classes = ["1ª Classe", "2ª Classe", "3ª Classe"]
taxas = [titanic[titanic["pclass"] == i]["survived"].mean() for i in [1, 2, 3]]
cores_barras = ["gold", "silver", "#cd7f32"]
bars = axes[1, 0].bar(classes, taxas, color=cores_barras, alpha=0.85, edgecolor="gray")
axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_title("Taxa de Sobrevivência por Classe — Titanic")
axes[1, 0].set_ylabel("Taxa de Sobrevivência")
for bar, taxa in zip(bars, taxas):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, taxa + 0.02,
                    f"{taxa:.1%}", ha="center", va="bottom", fontweight="bold")

# 5. Box plot
espécies = iris["species"].unique()
dados_box = [iris[iris["species"] == esp]["sepal_length"].values
             for esp in espécies]
bp = axes[1, 1].boxplot(dados_box, patch_artist=True, notch=True,
                         labels=espécies)
cores_box = ["#ff6b6b", "#4ecdc4", "#45b7d1"]
for patch, cor in zip(bp["boxes"], cores_box):
    patch.set_facecolor(cor)
    patch.set_alpha(0.7)
axes[1, 1].set_title("Comprimento Sépala por Espécie — Iris")
axes[1, 1].set_xlabel("Espécie")
axes[1, 1].set_ylabel("Comprimento da Sépala (cm)")

# 6. Pie chart
generos = titanic["sex"].value_counts()
axes[1, 2].pie(generos, labels=["Homens", "Mulheres"],
               autopct="%1.1f%%", startangle=90,
               colors=["#5b8ff9", "#ff85c0"],
               explode=[0.05, 0], shadow=True)
axes[1, 2].set_title("Distribuição por Gênero — Titanic")

plt.tight_layout()
plt.savefig("visualizacoes_matplotlib.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Figura salva como 'visualizacoes_matplotlib.png'")


# ─── Figura 2: Visualizações Avançadas com Seaborn ───────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Visualizações com Seaborn — Dataset Iris e Titanic",
             fontsize=16, fontweight="bold")

# 1. Pairplot simplificado (scatter matrix)
sns.scatterplot(data=iris, x="sepal_length", y="petal_length",
                hue="species", style="species", s=80, ax=axes[0, 0])
axes[0, 0].set_title("Sépala × Pétala por Espécie")

# 2. Violin plot
sns.violinplot(data=iris, x="species", y="sepal_length",
               palette="muted", inner="box", ax=axes[0, 1])
axes[0, 1].set_title("Distribuição Sépala por Espécie")

# 3. Heatmap de correlação
corr_matrix = iris.drop("species", axis=1).corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=axes[0, 2], square=True,
            cbar_kws={"shrink": 0.8})
axes[0, 2].set_title("Matriz de Correlação — Iris")

# 4. Count plot
sns.countplot(data=titanic, x="pclass", hue="survived",
              palette={0: "#e74c3c", 1: "#2ecc71"}, ax=axes[1, 0])
axes[1, 0].set_title("Sobrevivência por Classe")
axes[1, 0].set_xlabel("Classe")
axes[1, 0].legend(["Não sobreviveu", "Sobreviveu"])

# 5. KDE (Kernel Density Estimate)
for sexo, cor in [("male", "blue"), ("female", "red")]:
    subset = titanic[titanic["sex"] == sexo]["age"].dropna()
    sns.kdeplot(subset, ax=axes[1, 1], label=sexo, color=cor,
                fill=True, alpha=0.3)
axes[1, 1].set_title("Distribuição de Idade por Sexo")
axes[1, 1].set_xlabel("Idade")
axes[1, 1].legend()

# 6. Regplot (regressão com intervalo de confiança)
sns.regplot(data=tips, x="total_bill", y="tip",
            scatter_kws={"alpha": 0.4}, line_kws={"color": "red"},
            ax=axes[1, 2])
axes[1, 2].set_title("Gorjeta vs. Conta Total")
axes[1, 2].set_xlabel("Conta Total ($)")
axes[1, 2].set_ylabel("Gorjeta ($)")

plt.tight_layout()
plt.savefig("visualizacoes_seaborn.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Figura salva como 'visualizacoes_seaborn.png'")
```

---

## 6. Scikit-learn — Machine Learning na Prática

### 6.1 A API Unificada do Scikit-learn

```
Scikit-learn tem uma API consistente para todos os modelos:

Estimadores (modelos):
  .fit(X, y)         — treina o modelo
  .predict(X)        — faz predições
  .score(X, y)       — avalia desempenho (acurácia, R², etc.)
  .predict_proba(X)  — probabilidades (classificadores)

Transformadores (pré-processamento):
  .fit(X)            — aprende parâmetros de transformação
  .transform(X)      — aplica a transformação
  .fit_transform(X)  — combina fit + transform (mais eficiente)

Pipeline:
  Encadeia transformadores + estimador em um único objeto
```

### 6.2 Primeiro Modelo Completo

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline

# ─── 1. Carregar e explorar os dados ─────────────────────────────────────────
iris = load_iris()
X = iris.data          # shape: (150, 4)
y = iris.target        # shape: (150,) — 0, 1, 2

print("Dataset Iris:")
print(f"  Features: {iris.feature_names}")
print(f"  Classes:  {iris.target_names}")
print(f"  Shape X:  {X.shape}")
print(f"  Shape y:  {y.shape}")
print(f"  Distribuição de classes: {np.bincount(y)}")

# ─── 2. Divisão treino/teste ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% para teste
    random_state=42,      # reprodutibilidade
    stratify=y            # mantém proporção das classes
)

print(f"\nDivisão treino/teste:")
print(f"  Treino: {X_train.shape[0]} amostras")
print(f"  Teste:  {X_test.shape[0]} amostras")

# ─── 3. Pipeline: pré-processamento + modelo ─────────────────────────────────
# Pipeline garante que o scaler é ajustado APENAS nos dados de treino
# (evita data leakage — um erro comum de iniciantes!)
pipeline = Pipeline([
    ("scaler", StandardScaler()),           # Normalização Z-score
    ("modelo", KNeighborsClassifier(n_neighbors=5))  # KNN
])

# ─── 4. Treinamento ──────────────────────────────────────────────────────────
pipeline.fit(X_train, y_train)

# ─── 5. Avaliação ────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

print(f"\nResultados — KNN (k=5):")
print(f"  Acurácia no teste: {acuracia:.2%}")
print(f"\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ─── 6. Validação cruzada (melhor estimativa de desempenho) ──────────────────
scores_cv = cross_val_score(pipeline, X, y, cv=10, scoring="accuracy")
print(f"\nValidação cruzada 10-fold:")
print(f"  Acurácias: {scores_cv.round(3)}")
print(f"  Média:     {scores_cv.mean():.3f} ± {scores_cv.std():.3f}")

# ─── 7. Comparação de modelos ─────────────────────────────────────────────────
modelos = {
    "KNN (k=5)":         KNeighborsClassifier(n_neighbors=5),
    "KNN (k=3)":         KNeighborsClassifier(n_neighbors=3),
    "Árvore de Decisão": DecisionTreeClassifier(random_state=42),
    "Regressão Logística": LogisticRegression(max_iter=1000, random_state=42),
}

print(f"\n{'Modelo':<30} {'Média CV':>10} {'Desvio':>8}")
print("-" * 50)

for nome, modelo in modelos.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("modelo", modelo)])
    scores = cross_val_score(pipe, X, y, cv=10, scoring="accuracy")
    print(f"{nome:<30} {scores.mean():>10.3f} {scores.std():>8.3f}")

# ─── 8. Matriz de confusão ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=True)
ax.set_title(f"Matriz de Confusão — KNN (k=5)\nAcurácia: {acuracia:.1%}",
             fontsize=14)
plt.tight_layout()
plt.savefig("matriz_confusao_iris.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✓ Matriz de confusão salva.")
```

### 6.3 Pré-processamento com Scikit-learn

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                    RobustScaler, OneHotEncoder,
                                    OrdinalEncoder, LabelEncoder)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Dataset sintético com tipos mistos
dados = pd.DataFrame({
    "idade":   [25, np.nan, 35, 28, 45, 32],
    "salario": [3000, 5000, np.nan, 4500, 8000, 6000],
    "sexo":    ["M", "F", "M", "F", "M", np.nan],
    "cargo":   ["Junior", "Senior", "Pleno", "Junior", "Senior", "Pleno"],
    "churn":   [0, 0, 1, 0, 1, 1]
})

print("Dados brutos:")
print(dados.to_string())

# ─── Definir colunas por tipo ────────────────────────────────────────────────
colunas_numericas = ["idade", "salario"]
colunas_nominais  = ["sexo"]       # sem ordem
colunas_ordinais  = ["cargo"]      # com ordem (Junior < Pleno < Senior)

# ─── Pipeline para numéricas: imputação + normalização ───────────────────────
pipe_numerico = Pipeline([
    ("imputar", SimpleImputer(strategy="median")),  # preenche NaN com mediana
    ("escalar", StandardScaler()),                  # Z-score normalization
])

# ─── Pipeline para nominais: imputação + one-hot encoding ────────────────────
pipe_nominal = Pipeline([
    ("imputar", SimpleImputer(strategy="most_frequent")),  # preenche com moda
    ("onehot", OneHotEncoder(handle_unknown="ignore",      # codificação OHE
                              sparse_output=False)),
])

# ─── Pipeline para ordinais: codificação com ordem ───────────────────────────
pipe_ordinal = Pipeline([
    ("ordinal", OrdinalEncoder(categories=[["Junior", "Pleno", "Senior"]])),
])

# ─── ColumnTransformer: aplica cada pipeline às colunas corretas ─────────────
preprocessador = ColumnTransformer(transformers=[
    ("num", pipe_numerico, colunas_numericas),
    ("nom", pipe_nominal,  colunas_nominais),
    ("ord", pipe_ordinal,  colunas_ordinais),
])

X = dados.drop("churn", axis=1)
y = dados["churn"]

X_transformado = preprocessador.fit_transform(X)

nomes_features = (
    colunas_numericas
    + list(preprocessador.named_transformers_["nom"]
           .named_steps["onehot"].get_feature_names_out(colunas_nominais))
    + colunas_ordinais
)

df_transformado = pd.DataFrame(X_transformado, columns=nomes_features)
print("\nDados transformados:")
print(df_transformado.round(3).to_string())
```

---

## 7. Boas Práticas com Jupyter Notebooks

### 7.1 Estrutura Recomendada de um Notebook

```
projeto_analise.ipynb
├── Célula 1: Título + Descrição (Markdown)
│   # Análise de Churn de Clientes
│   **Objetivo**: Prever quais clientes cancelarão o serviço.
│   **Data**: 2024-06-01
│   **Autor**: Nome do Aluno
│
├── Célula 2: Imports (Python)
│   import numpy as np
│   import pandas as pd
│   ...
│
├── Célula 3: Constantes e Configurações
│   RANDOM_STATE = 42
│   TEST_SIZE = 0.2
│   DATA_PATH = "dados/clientes.csv"
│
├── Célula 4+: Análise exploratória
├── Célula N+: Modelagem
└── Célula final: Conclusões (Markdown)
```

### 7.2 Comandos Mágicos Úteis

```python
# Tempo de execução de uma célula
# %time minha_funcao()
# %timeit -n 100 operacao_rapida()

# Exibir gráficos inline (padrão no Jupyter Lab)
# %matplotlib inline

# Recarregar módulos automaticamente (útil durante desenvolvimento)
# %load_ext autoreload
# %autoreload 2

# Listar variáveis
# %whos

# Perfiling de desempenho
# %prun funcao_lenta()
```

### 7.3 Dicas de Produtividade

```python
# ─── Exibir múltiplos outputs em uma célula ──────────────────────────────────
from IPython.display import display

import pandas as pd
df1 = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
df2 = pd.DataFrame({"X": [5, 6], "Y": [7, 8]})
display(df1, df2)

# ─── Barra de progresso para loops longos ────────────────────────────────────
# from tqdm.notebook import tqdm
# for i in tqdm(range(1000)):
#     pass

# ─── Warnings: suprimir durante análise ──────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Reprodutibilidade: definir todas as seeds ───────────────────────────────
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
# Para PyTorch: torch.manual_seed(SEED)
# Para TensorFlow: tf.random.set_seed(SEED)

print("✓ Seeds definidas para reprodutibilidade")
```

---

## 8. Projeto Prático: Análise Exploratória Completa

```python
"""
Análise Exploratória de Dados (EDA) completa do dataset Titanic.
Este é o tipo de análise que precede qualquer projeto de ML.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Configuração de visualização
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("Set2")
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:.2f}".format)

print("="*60)
print("EDA COMPLETA — DATASET TITANIC")
print("="*60)

# ─── 1. Carregamento ─────────────────────────────────────────────────────────
df = sns.load_dataset("titanic")
print(f"\n1. Shape: {df.shape}")
print(f"   Linhas: {df.shape[0]}, Colunas: {df.shape[1]}")

# ─── 2. Visão geral ───────────────────────────────────────────────────────────
print("\n2. Primeiras linhas:")
print(df.head(3).to_string())

print("\n3. Tipos de dados:")
print(df.dtypes.to_string())

print("\n4. Estatísticas descritivas (numéricas):")
print(df.describe().round(2).to_string())

print("\n5. Estatísticas descritivas (categóricas):")
print(df.describe(include="object").to_string())

# ─── 3. Valores nulos ────────────────────────────────────────────────────────
nulos = df.isnull().sum()
pct_nulos = nulos / len(df) * 100
tabela_nulos = pd.DataFrame({
    "Nulos": nulos[nulos > 0],
    "Percentual": pct_nulos[nulos > 0].map("{:.1f}%".format)
})
print("\n6. Valores nulos:")
print(tabela_nulos.to_string())

# ─── 4. Análise da variável alvo ─────────────────────────────────────────────
print("\n7. Distribuição da variável alvo:")
print(df["survived"].value_counts(normalize=True).map("{:.1%}".format).to_string())

# ─── 5. Análise de correlação ─────────────────────────────────────────────────
num_cols = df.select_dtypes(include=np.number).columns.tolist()
corr_com_alvo = df[num_cols].corr()["survived"].drop("survived").sort_values()
print("\n8. Correlação com 'survived':")
print(corr_com_alvo.map("{:.3f}".format).to_string())

print("\n" + "="*60)
print("✓ EDA concluída! Próximos passos:")
print("  1. Tratar valores nulos (mediana para age, moda para embarked)")
print("  2. Encodar variáveis categóricas (sex, embarked, pclass)")
print("  3. Criar features: title (do nome), family_size (sibsp + parch)")
print("  4. Treinar modelos: LogisticRegression, RandomForest, XGBoost")
print("  5. Validar com cross-validation e ajustar hiperparâmetros")
```

---

## 9. Questões para Reflexão

1. **NumPy vs. Python puro**: Por que operações vetorizadas com NumPy são muito mais rápidas do que loops Python? Quando você **não** deveria usar NumPy?

2. **Broadcasting**: Explique com suas palavras o que acontece quando somamos um array de shape `(100, 3)` com um array de shape `(3,)`. Por que isso funciona? Como o broadcasting evita que você precise escrever loops explícitos?

3. **Data Leakage**: Por que é errado calcular a média e o desvio padrão de todos os dados (treino + teste) antes de fazer StandardScaler? Como o uso de Pipeline no Scikit-learn previne esse problema?

4. **Pandas vs. SQL**: Para que tipo de análise você preferiria usar Pandas? E SQL? Em quais situações Pandas tem vantagem sobre SQL (e vice-versa)?

5. **Visualização**: Escolha 3 tipos de gráfico da aula (histograma, box plot, violin plot, scatter plot, etc.) e explique: em que situações cada um é mais informativo? Quando um gráfico pode ser enganoso?

6. **Scikit-learn API**: A API unificada do Scikit-learn (`fit`, `transform`, `predict`) é um exemplo de design pattern. Qual é o benefício dessa consistência para um desenvolvedor que usa muitos modelos diferentes?

---

## Referências

**[1]** GÉRON, A. **Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow**. 3. ed. Alta Books, 2023. Cap. 2 (EDA e Projeto Completo), Cap. 3 (Classificação).

**[2]** MCKINNEY, W. **Python para Análise de Dados**. 3. ed. O'Reilly / Novatec, 2023.
> *O livro definitivo sobre Pandas, pelo criador da biblioteca.*

**[3]** VANDERPLAS, J. **Python Data Science Handbook**. O'Reilly, 2016.
> *Disponível gratuitamente em: https://jakevdp.github.io/PythonDataScienceHandbook/*

**[4]** NUMPY COMMUNITY. **NumPy Documentation**. Disponível em: https://numpy.org/doc/stable/

**[5]** PANDAS COMMUNITY. **Pandas Documentation**. Disponível em: https://pandas.pydata.org/docs/

**[6]** SCIKIT-LEARN COMMUNITY. **Scikit-learn User Guide**. Disponível em: https://scikit-learn.org/stable/user_guide.html

**[7]** HUNTER, J. D. Matplotlib: A 2D Graphics Environment. *Computing in Science & Engineering*, v. 9, n. 3, p. 90–95, 2007.

**[8]** WASKOM, M. Seaborn: Statistical Data Visualization. *Journal of Open Source Software*, v. 6, n. 60, 2021.

---

*Aula anterior: [Aula 04 — Representação do Conhecimento](./aula-04-representacao-conhecimento.md)*  
*Próximo módulo: [Módulo 02 — Paradigmas de Aprendizado de Máquina](../02-paradigmas-aprendizado/README.md)*
