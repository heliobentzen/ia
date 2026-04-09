# Aula 14 — Seleção de Atributos (Feature Selection)

> **Módulo 03 · Preparação e Análise de Dados** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender a maldição da dimensionalidade
- Aplicar métodos de filtro, wrapper e embedded para seleção de features
- Usar SelectKBest, RFE e importância de features de modelos tree-based

---

## 1. Por que Selecionar Features?

- Reduz overfitting
- Melhora desempenho e velocidade
- Aumenta interpretabilidade
- Combate a **maldição da dimensionalidade**: com muitas features, os dados ficam esparsos no espaço de alta dimensão

---

## 2. Métodos de Filtro

Avaliam cada feature **independentemente** do modelo, usando estatísticas.

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Seleção pelo teste F (ANOVA)
selector = SelectKBest(score_func=f_classif, k=10)
X_new = selector.fit_transform(X_train, y_train)

# Ver quais foram selecionadas
selected = selector.get_support(indices=True)
print(X.columns[selected].tolist())
```

**Métricas comuns:**
- Correlação de Pearson (regressão)
- ANOVA F-test (classificação)
- Informação Mútua (não-linear)
- Qui-quadrado (features categóricas)

---

## 3. Métodos Wrapper

Treinam o modelo repetidamente avaliando subsets de features. Mais custosos, mas geralmente melhores.

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X_train, y_train)
print(X.columns[rfe.support_].tolist())
```

---

## 4. Métodos Embedded

A seleção acontece **durante** o treinamento do modelo.

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Selecionar features acima da importância média
selector = SelectFromModel(rf, prefit=True)
X_selected = selector.transform(X_train)
```

**Lasso** também é um método embedded: coeficientes nulos = features eliminadas.

---

## 5. Importância de Features (Tree-based)

```python
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values(ascending=False).head(15).plot(kind='bar')
plt.title('Importância das Features (Random Forest)')
plt.tight_layout()
plt.show()
```

---

## Resumo Comparativo

| Método | Velocidade | Qualidade | Exemplos |
|--------|-----------|-----------|---------|
| Filtro | Rápido | Boa (independente) | SelectKBest, correlação |
| Wrapper | Lento | Ótima | RFE, Sequential |
| Embedded | Médio | Muito boa | Lasso, Feature Importance |

---

## Questões para Reflexão
1. Quando usar informação mútua em vez de correlação de Pearson?
2. Como a maldição da dimensionalidade afeta o KNN especificamente?
3. Seleção de features antes ou depois do split treino/teste? Por quê?

## Referências
- Géron, cap. 4
- Faceli et al., cap. 4

---
*Próxima aula → [Aula 15: Pipeline de Dados](aula-15-pipeline-dados.md)*
