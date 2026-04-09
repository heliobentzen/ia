# Aula 11 — Qualidade e Limpeza de Dados

> **Módulo 03 · Aula 11 · Carga horária: 1 hora-aula**

---

## Objetivos da Aula

- Identificar e classificar problemas de qualidade de dados.
- Tratar valores ausentes com estratégias adequadas ao mecanismo de ausência.
- Detectar e tratar outliers com múltiplas abordagens.
- Remover duplicatas e corrigir inconsistências.

---

## 1. Problemas de Qualidade de Dados

| Problema | Descrição | Exemplo |
|----------|-----------|---------|
| **Valores ausentes** | NaN, NULL, vazio | Idade não preenchida |
| **Outliers** | Valores extremos atípicos | Salário de R$1.000.000 em uma base de R$3.000–R$20.000 |
| **Duplicatas** | Registros repetidos | Mesmo CPF cadastrado duas vezes |
| **Inconsistências** | Mesma info em formatos diferentes | "São Paulo" / "sp" / "S.Paulo" |
| **Erros de digitação** | Typos em texto | "Brasl" em vez de "Brasil" |
| **Valores inválidos** | Fora do domínio possível | Idade = -5, nota = 15/10 |

---

## 2. Valores Ausentes

### 2.1 Mecanismos de Ausência

| Mecanismo | Nome | Descrição | Impacto na Imputação |
|-----------|------|-----------|---------------------|
| **MCAR** | Missing Completely At Random | Ausência aleatória, sem padrão | Imputação simples funciona |
| **MAR** | Missing At Random | Ausência depende de outras variáveis observadas | Imputação condicional recomendada |
| **MNAR** | Missing Not At Random | Ausência depende do próprio valor ausente | Mais difícil; requer atenção especial |

**Exemplos:**
- **MCAR**: sensor de temperatura falhou aleatoriamente → ausência de leituras.
- **MAR**: pacientes mais velhos não respondem à pergunta de renda → ausência depende da idade (observável).
- **MNAR**: pacientes com alto nível de stress se recusam a reportar o estresse → ausência depende do valor oculto.

### 2.2 Diagnóstico de Valores Ausentes

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
n = 1000

# Criar dataset com padrões realistas de ausência
idade    = np.random.normal(40, 15, n).clip(18, 90)
salario  = np.exp(np.random.normal(9, 0.5, n))  # log-normal
satisf   = np.random.choice([1,2,3,4,5], n)
regiao   = np.random.choice(['Sul','Sudeste','Norte','Nordeste','Centro-Oeste'], n)
churn    = (np.random.random(n) < 0.25).astype(int)

df = pd.DataFrame({
    'idade': idade, 'salario': salario,
    'satisfacao': satisf.astype(float),
    'regiao': regiao, 'churn': churn
})

# Introduzir ausências
df.loc[np.random.choice(n, 50, replace=False), 'idade'] = np.nan           # MCAR ~5%
df.loc[df['idade'] > 60, 'salario'] = np.where(                             # MAR (depende da idade)
    np.random.random((df['idade'] > 60).sum()) < 0.4, np.nan,
    df.loc[df['idade'] > 60, 'salario']
)
df.loc[df['satisfacao'] <= 2, 'satisfacao'] = np.where(                     # MNAR (insatisfeitos não respondem)
    np.random.random((df['satisfacao'] <= 2).sum()) < 0.6, np.nan,
    df.loc[df['satisfacao'] <= 2, 'satisfacao']
)

print("=== DIAGNÓSTICO DE VALORES AUSENTES ===")
print(f"\nContagem:\n{df.isnull().sum()}")
print(f"\nPorcentagem:\n{(df.isnull().mean()*100).round(2)}")

# Padrões de ausência
print("\nCo-ocorrência de ausências:")
print(df.isnull().any(axis=1).sum(), "linhas com pelo menos 1 ausente")

# Teste Little's MCAR (simplificado: t-test entre grupos)
mask_salario_ausente = df['salario'].isnull()
t, p = stats.ttest_ind(
    df.loc[~mask_salario_ausente, 'idade'].dropna(),
    df.loc[mask_salario_ausente, 'idade'].dropna()
)
print(f"\nTeste MCAR (salário ausente vs idade): t={t:.3f}, p={p:.4f}")
if p < 0.05:
    print("  → Ausência de salário NÃO é aleatória em relação à idade (MAR)")
else:
    print("  → Ausência de salário parece aleatória em relação à idade")
```

### 2.3 Estratégias de Imputação

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Criar versão com valores reais para comparação
df_real = df.copy()
mask_ausentes = df.isnull()

# --- 1. Remoção (Listwise) ---
df_removido = df.dropna()
print(f"Listwise deletion: {len(df_removido)}/{len(df)} linhas restantes ({len(df_removido)/len(df):.0%})")

# --- 2. Imputação Simples ---
imp_media   = SimpleImputer(strategy='mean')
imp_mediana = SimpleImputer(strategy='median')
imp_moda    = SimpleImputer(strategy='most_frequent')

df_num = df[['idade', 'salario', 'satisfacao']].copy()
X_media   = imp_media.fit_transform(df_num)
X_mediana = imp_mediana.fit_transform(df_num)
print(f"\nImputação por média  — salário imputado: {imp_media.statistics_[1]:.2f}")
print(f"Imputação por mediana — salário imputado: {imp_mediana.statistics_[1]:.2f}")

# --- 3. KNN Imputer ---
knn_imp = KNNImputer(n_neighbors=5, weights='distance')
X_knn = knn_imp.fit_transform(df_num)
print(f"\nKNN Imputer: imputação baseada em {knn_imp.n_neighbors} vizinhos mais próximos")

# --- 4. Imputação Iterativa (MICE) ---
mice_imp = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10, random_state=42
)
X_mice = mice_imp.fit_transform(df_num)
print(f"MICE (IterativeImputer): {mice_imp.n_iter_} iterações realizadas")

# --- Comparação visual ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metodos = [
    ('Original (sem ausentes)', df_num['salario'].dropna()),
    ('Imputação por Mediana', pd.Series(X_mediana[:, 1])),
    ('KNN Imputer', pd.Series(X_knn[:, 1])),
]
for ax, (nome, dados) in zip(axes, metodos):
    ax.hist(dados, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
    ax.axvline(dados.median(), color='red', linestyle='--', label=f'Mediana: {dados.median():.0f}')
    ax.set(title=nome, xlabel='Salário', ylabel='Frequência')
    ax.legend()
plt.suptitle('Comparação de Estratégias de Imputação', fontsize=13)
plt.tight_layout()
plt.savefig("imputacao_comparacao.png", dpi=150, bbox_inches='tight')
plt.show()

# --- 5. Imputação com indicador de ausência (flag) ---
df_com_flag = df.copy()
for col in ['idade', 'salario', 'satisfacao']:
    df_com_flag[f'{col}_era_ausente'] = df[col].isnull().astype(int)
df_com_flag = df_com_flag.fillna(df_com_flag.median(numeric_only=True))
print("\nColunas após imputação com flag:")
print([c for c in df_com_flag.columns if 'era_ausente' in c])
```

---

## 3. Detecção e Tratamento de Outliers

### 3.1 Métodos de Detecção

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Gerar dataset com outliers
np.random.seed(42)
X_normal   = np.random.normal(0, 1, (950, 2))
X_outliers = np.random.uniform(-5, 5, (50, 2)) * 3
X_all = np.vstack([X_normal, X_outliers])
y_true = np.array([1]*950 + [-1]*50)  # 1=normal, -1=outlier

print("=== DETECÇÃO DE OUTLIERS ===\n")

# --- 1. IQR (Interquartile Range) ---
col = X_all[:, 0]
Q1, Q3 = np.percentile(col, 25), np.percentile(col, 75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR
outliers_iqr = (col < limite_inf) | (col > limite_sup)
print(f"IQR: [{limite_inf:.2f}, {limite_sup:.2f}]")
print(f"  Outliers detectados: {outliers_iqr.sum()} ({outliers_iqr.mean():.1%})")

# --- 2. Z-Score ---
z_scores = np.abs(stats.zscore(col))
outliers_zscore = z_scores > 3
print(f"\nZ-Score (|z| > 3):")
print(f"  Outliers detectados: {outliers_zscore.sum()} ({outliers_zscore.mean():.1%})")

# --- 3. Isolation Forest ---
iso_forest = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
preds_iso = iso_forest.fit_predict(X_all)
outliers_iso = preds_iso == -1
print(f"\nIsolation Forest (contamination=0.05):")
print(f"  Outliers detectados: {outliers_iso.sum()} ({outliers_iso.mean():.1%})")
acuracia_iso = (preds_iso == y_true).mean()
print(f"  Acurácia vs verdade: {acuracia_iso:.2%}")

# --- 4. LOF (Local Outlier Factor) ---
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
preds_lof = lof.fit_predict(X_all)
outliers_lof = preds_lof == -1
print(f"\nLOF (n_neighbors=20, contamination=0.05):")
print(f"  Outliers detectados: {outliers_lof.sum()} ({outliers_lof.mean():.1%})")
acuracia_lof = (preds_lof == y_true).mean()
print(f"  Acurácia vs verdade: {acuracia_lof:.2%}")

# Visualização comparativa
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
metodos_det = [
    ("IQR (col 0)", outliers_iqr),
    ("Z-Score (col 0)", outliers_zscore),
    ("Isolation Forest", outliers_iso),
    ("LOF", outliers_lof),
]
for ax, (nome, mask) in zip(axes.flat, metodos_det):
    ax.scatter(X_all[~mask, 0], X_all[~mask, 1], c='steelblue', s=10, alpha=0.5, label='Normal')
    ax.scatter(X_all[mask, 0],  X_all[mask, 1],  c='red', s=30, alpha=0.8, label=f'Outlier ({mask.sum()})')
    ax.set(title=nome); ax.legend(fontsize=8)
plt.suptitle('Comparação de Métodos de Detecção de Outliers', fontsize=13)
plt.tight_layout()
plt.savefig("outliers_comparacao.png", dpi=150, bbox_inches='tight')
plt.show()
```

### 3.2 Tratamento de Outliers

```python
import pandas as pd
import numpy as np

salarios = pd.Series([3000, 4500, 3200, 5000, 4800, 3700, 4100, 95000, 3300, 4600])
print("Salários originais:", salarios.values)

# --- 1. Remoção ---
Q1, Q3 = salarios.quantile(0.25), salarios.quantile(0.75)
IQR = Q3 - Q1
sem_outliers = salarios[(salarios >= Q1 - 1.5*IQR) & (salarios <= Q3 + 1.5*IQR)]
print(f"\n1. Remoção: {len(sem_outliers)} valores restantes")

# --- 2. Capping (Winsorization) ---
limite_sup = salarios.quantile(0.95)
limite_inf = salarios.quantile(0.05)
capped = salarios.clip(lower=limite_inf, upper=limite_sup)
print(f"2. Capping (p5-p95): {capped.values}")

# --- 3. Winsorization com scipy ---
from scipy.stats.mstats import winsorize
winsorizado = winsorize(salarios, limits=[0.05, 0.05])
print(f"3. Winsorize (5%): {np.array(winsorizado)}")

# --- 4. Transformação logarítmica ---
log_salarios = np.log1p(salarios)
print(f"4. Log(1+x): {log_salarios.round(2).values}")
print(f"   Skewness original: {salarios.skew():.2f} → após log: {log_salarios.skew():.2f}")
```

---

## 4. Duplicatas e Inconsistências

```python
# --- DUPLICATAS ---
df_dup = pd.DataFrame({
    'cpf': ['123.456.789-00', '987.654.321-00', '123.456.789-00', '111.111.111-11'],
    'nome': ['Ana Silva', 'Bruno Costa', 'Ana Silva', 'Carla Souza'],
    'valor': [1500, 2300, 1500, 800]
})

print("=== DUPLICATAS ===")
print(f"Total: {len(df_dup)} | Duplicatas: {df_dup.duplicated().sum()}")
print(f"Duplicatas por CPF: {df_dup['cpf'].duplicated().sum()}")

df_sem_dup = df_dup.drop_duplicates()
df_sem_dup_cpf = df_dup.drop_duplicates(subset=['cpf'], keep='last')
print(f"Após remoção: {len(df_sem_dup)} registros")

# --- INCONSISTÊNCIAS ---
df_inc = pd.DataFrame({
    'estado': ['SP', 'Sao Paulo', 'são paulo', 'S.Paulo', 'sp', 'RJ', 'Rio de Janeiro'],
    'cep': ['01310-100', '01310100', '01310.100', '01310-100', '01310-100', '20040-020', '20040020'],
    'valor': ['R$ 1.500,00', '2300.50', '1,200', 'R$800', '1500', '2100.00', '900,50']
})

print("\n=== INCONSISTÊNCIAS ===")

# Padronização de strings
mapa_estados = {
    'sp': 'SP', 'são paulo': 'SP', 'sao paulo': 'SP', 's.paulo': 'SP',
    'rj': 'RJ', 'rio de janeiro': 'RJ'
}
df_inc['estado_padrao'] = df_inc['estado'].str.lower().str.strip().map(
    lambda x: mapa_estados.get(x, x.upper())
)

# Padronização de CEP
df_inc['cep_padrao'] = df_inc['cep'].str.replace(r'[.\-\s]', '', regex=True)
df_inc['cep_padrao'] = df_inc['cep_padrao'].apply(
    lambda x: f"{x[:5]}-{x[5:]}" if len(x) == 8 else x
)

# Padronização de valores monetários
def limpar_valor(v):
    v = str(v).replace('R$', '').replace(' ', '').strip()
    if ',' in v and '.' in v:
        v = v.replace('.', '').replace(',', '.')
    elif ',' in v:
        v = v.replace(',', '.')
    try:
        return float(v)
    except:
        return np.nan

df_inc['valor_numerico'] = df_inc['valor'].apply(limpar_valor)

print(df_inc[['estado', 'estado_padrao', 'cep', 'cep_padrao', 'valor', 'valor_numerico']])
```

---

## 5. Tabela Resumo: Estratégias de Tratamento

| Problema | Quando usar | Estratégia recomendada |
|----------|------------|----------------------|
| Ausentes < 5% (MCAR) | Baixo impacto | Imputação por mediana/moda |
| Ausentes 5–30% (MAR) | Depende de outras vars | KNN Imputer ou MICE |
| Ausentes > 30% | Alta % ausente | Considerar remover a coluna ou criar flag |
| Outliers isolados | Poucos extremos | Capping (p1/p99) ou remoção |
| Outliers padrão (anomalia) | Detecção específica | Isolation Forest, LOF |
| Distribuição muito assimétrica | Skew > 1 | Transformação log/sqrt/Box-Cox |

---

## 6. Exercícios

1. **Diagnóstico**: No dataset Titanic, identifique o mecanismo de ausência de `Age`, `Cabin` e `Embarked`. Justifique sua hipótese com análises.
2. **Imputação**: Compare imputação por mediana vs. KNN para `Age` no Titanic medindo o impacto na acurácia de um modelo de Random Forest.
3. **Outliers**: No dataset de salários (crie com `np.random.lognormal`), compare IQR, Z-score e Isolation Forest. Qual detecta mais? Qual é mais preciso?
4. **Limpeza Real**: Baixe um dataset do Kaggle e aplique o pipeline completo: duplicatas → valores ausentes → outliers → inconsistências.

---

## 7. Referências

- GÉRON, Aurélien. *Mãos à Obra*. 3. ed. Alta Books, 2023. Cap. 2.
- FACELI et al. *Inteligência Artificial*. 2. ed. LTC, 2021. Cap. 3.
- Scikit-learn: [sklearn.impute](https://scikit-learn.org/stable/modules/impute.html)

---

*← [Aula 10 — EDA](aula-10-coleta-e-eda.md) | [Aula 12 — Feature Engineering →](aula-12-feature-engineering.md)*
