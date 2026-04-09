# Aula 13 — Transformações e Normalização de Dados

> **Módulo 03 · Preparação e Análise de Dados** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender por que escalar e transformar dados é essencial
- Aplicar MinMaxScaler, StandardScaler, RobustScaler e PowerTransformer
- Escolher a transformação adequada para cada situação

---

## 1. Por que Escalar?

Algoritmos sensíveis a distância (KNN, SVM, redes neurais) ou que usam gradiente sofrem quando as features têm escalas muito diferentes. Feature com valores em milhares domina sobre feature com valores em frações.

---

## 2. Principais Técnicas

### 2.1 Normalização Min-Max
$$X' = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

Resultado entre **[0, 1]**. Sensível a outliers.

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 2.2 Padronização (Z-score)
$$X' = \frac{X - \mu}{\sigma}$$

Média 0, desvio padrão 1. Mais robusto que Min-Max para outliers moderados.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 2.3 RobustScaler
Usa mediana e IQR. Ideal quando há outliers fortes.

```python
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)
```

### 2.4 PowerTransformer (Box-Cox / Yeo-Johnson)
Transforma dados para aproximar distribuição normal. Útil para regressão linear.

```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
X_transformed = pt.fit_transform(X_train)
```

---

## 3. Regra de Ouro: Fit no Treino, Transform no Teste

```python
scaler.fit(X_train)          # aprende parâmetros SOMENTE no treino
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)   # aplica os MESMOS parâmetros
```

> ⚠️ Nunca faça `fit` no conjunto de teste — isso causa *data leakage*.

---

## 4. Quando Usar Cada Scaler?

| Scaler | Recomendado para |
|--------|-----------------|
| MinMaxScaler | Redes neurais, dados sem outliers |
| StandardScaler | Maioria dos algoritmos (SVM, Logística, PCA) |
| RobustScaler | Dados com outliers significativos |
| PowerTransformer | Regressão com suposição de normalidade |

---

## Questões para Reflexão
1. Por que aplicar o scaler antes do split treino/teste é um erro?
2. Em árvores de decisão e Random Forest, escalar features é necessário?
3. Como você verificaria visualmente se uma transformação aproximou os dados de uma normal?

## Referências
- Géron, cap. 2 (Pipeline de Preparação de Dados)
- Faceli et al., cap. 3

---
*Próxima aula → [Aula 14: Seleção de Atributos](aula-14-selecao-atributos.md)*
