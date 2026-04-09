# Aula 12 — Feature Engineering

> **Módulo 03 · Aula 12 · Carga horária: 1 hora-aula**

---

## Objetivos da Aula

- Compreender o conceito e a importância de Feature Engineering.
- Aplicar encoding para variáveis categóricas com a técnica adequada.
- Criar features temporais, de interação e de texto.
- Implementar feature engineering específico de domínio.

---

## 1. O Que é Feature Engineering?

Feature Engineering é o processo de usar **conhecimento de domínio** para criar, transformar ou combinar variáveis brutas em representações mais informativas para os modelos de ML.

> *"Feature engineering is the most important step in applied machine learning."* — Andrew Ng

**Por que importa?**
- Modelos como árvores de decisão e regressão linear não conseguem descobrir automaticamente relações não-lineares ou combinações de features.
- Uma boa feature pode melhorar o modelo mais do que um algoritmo mais sofisticado.
- Reduz a necessidade de modelos complexos.

---

## 2. Encoding de Variáveis Categóricas

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, OneHotEncoder
)
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import category_encoders as ce   # pip install category_encoders

np.random.seed(42)
n = 2000

df = pd.DataFrame({
    'pais':     np.random.choice(['Brasil','EUA','Alemanha','França','Japão'], n),
    'educacao': np.random.choice(['Fundamental','Médio','Superior','Pós-Grad'], n),
    'regiao':   np.random.choice(['Norte','Sul','Leste','Oeste'], n),
    'valor':    np.random.randn(n)
})
df['target'] = (
    (df['educacao'].isin(['Superior','Pós-Grad'])).astype(int)
    + (df['pais'] == 'Japão').astype(int)
    + (np.random.randn(n) > 0.5).astype(int)
) >= 2

print("=== ENCODING DE VARIÁVEIS CATEGÓRICAS ===\n")

# --- 1. Label Encoding ---
# Uso: target encoding, features ordinais com semântica
le = LabelEncoder()
df['pais_label'] = le.fit_transform(df['pais'])
print(f"Label Encoding (pais):")
for orig, enc in zip(le.classes_, range(len(le.classes_))):
    print(f"  {orig} → {enc}")
print("  ⚠️  Cria ordem artificial! Usar apenas com árvores ou para target.")

# --- 2. Ordinal Encoding ---
# Uso: variáveis com ordem natural
ordem_educacao = [['Fundamental','Médio','Superior','Pós-Grad']]
oe = OrdinalEncoder(categories=ordem_educacao)
df['educacao_ordinal'] = oe.fit_transform(df[['educacao']])
print(f"\nOrdinal Encoding (educação): {dict(zip(ordem_educacao[0], range(4)))}")

# --- 3. One-Hot Encoding ---
# Uso: variáveis nominais com poucas categorias
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
X_ohe = ohe.fit_transform(df[['pais']])
df_ohe = pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(['pais']))
print(f"\nOne-Hot Encoding (pais):")
print(f"  {ohe.get_feature_names_out(['pais']).tolist()}")
print(f"  Shape: {df_ohe.shape}")
print("  ⚠️  Alta cardinalidade → explosão de dimensionalidade!")

# --- 4. Target Encoding ---
# Uso: alta cardinalidade; substitui categoria pela média do target
target_enc = ce.TargetEncoder(cols=['pais', 'regiao'])
df_target_enc = target_enc.fit_transform(df[['pais','regiao']], df['target'])
print(f"\nTarget Encoding (pais):")
print(df_target_enc['pais'].describe().round(4))
print("  ⚠️  Requer cuidado com data leakage — usar apenas no treino!")

# --- 5. Binary Encoding ---
# Uso: alta cardinalidade; mais compacto que OHE
bin_enc = ce.BinaryEncoder(cols=['pais'])
df_binary = bin_enc.fit_transform(df[['pais']])
print(f"\nBinary Encoding (pais): {df_binary.shape[1]} colunas (vs {len(df['pais'].unique())-1} no OHE com drop='first')")

# --- 6. Frequência Encoding ---
# Uso: alta cardinalidade; substitui categoria pela frequência
freq_map = df['pais'].value_counts(normalize=True).to_dict()
df['pais_freq'] = df['pais'].map(freq_map)
print(f"\nFrequência Encoding (pais):")
print(df.groupby('pais')['pais_freq'].first().round(4))

# --- Comparação de encodings em modelo ---
print("\n=== COMPARAÇÃO DE ENCODINGS EM GRADIENT BOOSTING ===")
configs = {
    'Label Enc.':  df[['educacao_ordinal', 'pais_label']].values,
    'Target Enc.': df_target_enc.values,
    'Freq Enc.':   df[['pais_freq', 'educacao_ordinal']].values,
}
clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
for nome, X in configs.items():
    scores = cross_val_score(clf, X, df['target'], cv=5, scoring='roc_auc')
    print(f"  {nome:15s} AUC: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 3. Features de Data e Hora

```python
import pandas as pd
import numpy as np

# Criar série temporal simulada
datas = pd.date_range('2022-01-01', '2023-12-31', freq='H')
df_ts = pd.DataFrame({'timestamp': datas, 'vendas': np.random.poisson(50, len(datas))})

def extrair_features_temporais(df, col_data):
    d = df[col_data]
    df = df.copy()

    # Componentes básicos
    df['ano']           = d.dt.year
    df['mes']           = d.dt.month
    df['dia']           = d.dt.day
    df['hora']          = d.dt.hour
    df['dia_semana']    = d.dt.dayofweek       # 0=segunda, 6=domingo
    df['dia_ano']       = d.dt.dayofyear
    df['semana_ano']    = d.dt.isocalendar().week.astype(int)
    df['trimestre']     = d.dt.quarter

    # Features booleanas
    df['is_fim_semana'] = d.dt.dayofweek.isin([5, 6]).astype(int)
    df['is_feriado']    = d.dt.date.isin([
        pd.Timestamp('2022-01-01').date(),
        pd.Timestamp('2022-04-21').date(),
        pd.Timestamp('2022-09-07').date(),
    ]).astype(int)

    # Codificação cíclica (para preservar periodicidade)
    # Ex.: hora 23 e hora 0 são próximas — usar sin/cos
    df['hora_sin']     = np.sin(2 * np.pi * d.dt.hour / 24)
    df['hora_cos']     = np.cos(2 * np.pi * d.dt.hour / 24)
    df['mes_sin']      = np.sin(2 * np.pi * d.dt.month / 12)
    df['mes_cos']      = np.cos(2 * np.pi * d.dt.month / 12)
    df['diasem_sin']   = np.sin(2 * np.pi * d.dt.dayofweek / 7)
    df['diasem_cos']   = np.cos(2 * np.pi * d.dt.dayofweek / 7)

    return df

df_ts = extrair_features_temporais(df_ts, 'timestamp')
print(f"Features temporais criadas: {df_ts.shape[1] - 2} novas colunas")
print(df_ts[['timestamp','mes','hora','is_fim_semana','hora_sin','hora_cos']].head(5).to_string())
```

---

## 4. Features de Séries Temporais: Lags e Rolling Statistics

```python
def criar_features_series_temporais(serie, lags=[1,7,14,28], windows=[7,14,30]):
    """Cria lags e estatísticas em janela deslizante."""
    df_feat = pd.DataFrame({'valor': serie.values}, index=serie.index)

    # Lags
    for lag in lags:
        df_feat[f'lag_{lag}'] = serie.shift(lag)

    # Rolling statistics
    for w in windows:
        df_feat[f'media_movel_{w}']  = serie.shift(1).rolling(w).mean()
        df_feat[f'desvio_movel_{w}'] = serie.shift(1).rolling(w).std()
        df_feat[f'min_movel_{w}']    = serie.shift(1).rolling(w).min()
        df_feat[f'max_movel_{w}']    = serie.shift(1).rolling(w).max()

    # Diferenciação
    df_feat['diff_1']  = serie.diff(1)
    df_feat['diff_7']  = serie.diff(7)

    # Percentual de variação
    df_feat['pct_change_1'] = serie.pct_change(1)
    df_feat['pct_change_7'] = serie.pct_change(7)

    return df_feat.dropna()

# Simular série de vendas diária
np.random.seed(42)
datas_diarias = pd.date_range('2022-01-01', '2023-12-31', freq='D')
vendas_diarias = pd.Series(
    50 + 10 * np.sin(2*np.pi*np.arange(len(datas_diarias))/365)
    + np.random.randn(len(datas_diarias)) * 5,
    index=datas_diarias, name='vendas'
)

df_lags = criar_features_series_temporais(vendas_diarias)
print(f"Features de séries temporais: {df_lags.shape}")
print(df_lags.iloc[0].to_string())
```

---

## 5. Features de Texto

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import re

textos = [
    "O produto chegou no prazo e está em ótimo estado!",
    "Péssimo atendimento, produto com defeito.",
    "Entrega rápida, qualidade excelente. Recomendo!",
    "Produto não correspondeu às expectativas. Decepcionante.",
    "Ótimo custo-benefício. Comprarei novamente!"
]

# --- 1. Features básicas de texto ---
def features_basicas_texto(textos):
    features = []
    for txt in textos:
        features.append({
            'n_caracteres':    len(txt),
            'n_palavras':      len(txt.split()),
            'n_frases':        txt.count('.') + txt.count('!') + txt.count('?'),
            'n_exclamacoes':   txt.count('!'),
            'n_maiusculas':    sum(1 for c in txt if c.isupper()),
            'comprimento_med_palavras': np.mean([len(p) for p in txt.split()]),
            'tem_negacao':     int(any(n in txt.lower() for n in ['não', 'nunca', 'jamais', 'péssimo'])),
        })
    return pd.DataFrame(features)

df_texto_feat = features_basicas_texto(textos)
print("=== FEATURES BÁSICAS DE TEXTO ===")
print(df_texto_feat)

# --- 2. Bag of Words ---
cv = CountVectorizer(max_features=20, stop_words=None, ngram_range=(1, 2))
X_bow = cv.fit_transform(textos)
print(f"\nBag of Words: shape={X_bow.shape}")
print("Vocabulário:", cv.get_feature_names_out()[:10])

# --- 3. TF-IDF ---
tfidf = TfidfVectorizer(max_features=20, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(textos)
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())
print(f"\nTF-IDF: shape={X_tfidf.shape}")
# Features mais importantes para o primeiro texto
top_features = df_tfidf.iloc[0].nlargest(5)
print(f"Top features (texto 1): {top_features.to_dict()}")
```

---

## 6. Features Geográficas e de Interação

```python
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures

# --- Features Geográficas ---
def distancia_haversine(lat1, lon1, lat2, lon2):
    """Distância em km entre dois pontos geográficos."""
    R = 6371  # raio da Terra em km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

np.random.seed(42)
n = 100
df_geo = pd.DataFrame({
    'lat_cliente': np.random.uniform(-23.8, -23.3, n),
    'lon_cliente': np.random.uniform(-46.9, -46.4, n),
})
# Loja central (São Paulo)
lat_loja, lon_loja = -23.5505, -46.6333
df_geo['dist_loja_km'] = distancia_haversine(
    df_geo['lat_cliente'], df_geo['lon_cliente'], lat_loja, lon_loja
)
print(f"Distância até a loja (primeiros 5): {df_geo['dist_loja_km'].head().round(2).values}")

# --- Features de Interação ---
np.random.seed(42)
df_int = pd.DataFrame({
    'salario': np.random.normal(5000, 2000, 500).clip(1000),
    'despesas': np.random.normal(3000, 1500, 500).clip(500),
    'filhos': np.random.poisson(1, 500),
    'anos_emprego': np.random.exponential(5, 500).clip(0.5, 30),
})

# Interações manuais com semântica
df_int['comprometimento_renda'] = df_int['despesas'] / df_int['salario']
df_int['saldo_mensal'] = df_int['salario'] - df_int['despesas']
df_int['custo_por_filho'] = df_int['despesas'] / (df_int['filhos'] + 1)
df_int['salario_por_ano_exp'] = df_int['salario'] / df_int['anos_emprego']

# Polynomial Features (interações automáticas)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(df_int[['salario','despesas','filhos']])
nomes_poly = poly.get_feature_names_out(['salario','despesas','filhos'])
print(f"\nPolynomial Features (degree=2, interaction_only): {X_poly.shape[1]} features")
print("Nomes:", nomes_poly)
```

---

## 7. Feature Engineering por Domínio

```python
# --- E-COMMERCE ---
df_ecomm = pd.DataFrame({
    'data_cadastro': pd.date_range('2020-01-01', periods=500, freq='3D'),
    'data_ultima_compra': pd.date_range('2021-01-01', periods=500, freq='2D'),
    'valor_total_compras': np.random.exponential(500, 500),
    'num_pedidos': np.random.poisson(5, 500) + 1,
    'num_itens_devolvidos': np.random.poisson(0.5, 500),
    'categorias_compradas': np.random.poisson(2, 500) + 1,
})
hoje = pd.Timestamp('2023-06-01')

df_ecomm['dias_desde_cadastro']     = (hoje - df_ecomm['data_cadastro']).dt.days
df_ecomm['dias_desde_ultima_compra']= (hoje - df_ecomm['data_ultima_compra']).dt.days
df_ecomm['ticket_medio']            = df_ecomm['valor_total_compras'] / df_ecomm['num_pedidos']
df_ecomm['taxa_devolucao']          = df_ecomm['num_itens_devolvidos'] / df_ecomm['num_pedidos']
df_ecomm['diversidade_categorias']  = df_ecomm['categorias_compradas']
df_ecomm['frequencia_compras']      = df_ecomm['num_pedidos'] / (df_ecomm['dias_desde_cadastro'] / 30)
# RFM score simplificado
df_ecomm['rfm_recencia'] = pd.qcut(df_ecomm['dias_desde_ultima_compra'], q=5, labels=[5,4,3,2,1])
df_ecomm['rfm_frequencia'] = pd.qcut(df_ecomm['num_pedidos'].rank(method='first'), q=5, labels=[1,2,3,4,5])
df_ecomm['rfm_monetario'] = pd.qcut(df_ecomm['valor_total_compras'].rank(method='first'), q=5, labels=[1,2,3,4,5])
df_ecomm['rfm_score'] = (df_ecomm['rfm_recencia'].astype(int)
                        + df_ecomm['rfm_frequencia'].astype(int)
                        + df_ecomm['rfm_monetario'].astype(int))

print("=== FEATURES E-COMMERCE (RFM) ===")
print(df_ecomm[['ticket_medio','taxa_devolucao','frequencia_compras','rfm_score']].describe().round(3))
```

---

## 8. Guia de Escolha de Encoding

| Situação | Técnica recomendada |
|----------|-------------------|
| Poucas categorias (< 10), nominal | One-Hot Encoding |
| Ordem natural clara | Ordinal Encoding |
| Alta cardinalidade (> 20) + target disponível | Target Encoding (com CV) |
| Alta cardinalidade + sem target | Frequency Encoding ou Binary Encoding |
| Modelo baseado em árvores | Label Encoding é suficiente |
| Modelo linear / KNN / SVM | One-Hot ou Target Encoding |

---

## 9. Exercícios

1. **Encoding**: No dataset Titanic, aplique one-hot para `Pclass` e `Embarked`, ordinal para `Pclass` tratado como ordinal. Compare o impacto na acurácia de um modelo.
2. **Features Temporais**: Baixe um dataset de vendas com datas. Crie features cíclicas (sin/cos) para hora, dia da semana e mês. Avalie se melhoram um modelo de regressão.
3. **RFM**: Implemente o score RFM completo (Recência, Frequência, Monetário) para um dataset de e-commerce. Segmente os clientes em campeões, em risco e perdidos.
4. **Texto**: Crie features de texto para um dataset de reviews (ex.: Amazon reviews). Compare BOW vs TF-IDF vs features básicas num classificador de sentimento.

---

## 10. Referências

- GÉRON, Aurélien. *Mãos à Obra*. 3. ed. Alta Books, 2023. Cap. 2.
- FACELI et al. *Inteligência Artificial*. 2. ed. LTC, 2021. Cap. 4.
- Category Encoders: [https://contrib.scikit-learn.org/category_encoders/](https://contrib.scikit-learn.org/category_encoders/)

---

*← [Aula 11 — Limpeza](aula-11-qualidade-limpeza.md) | [Aula 13 — Normalização →](aula-13-transformacoes-normalizacao.md)*
