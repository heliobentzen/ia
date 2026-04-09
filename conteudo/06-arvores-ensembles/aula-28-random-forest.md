# Aula 28 — Random Forest

> **Módulo 06 · Métodos Baseados em Árvores e Ensembles** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender bagging e o papel da aleatoriedade no Random Forest
- Implementar e ajustar Random Forest para classificação e regressão
- Interpretar a importância de features e usar permutation importance

---

## 1. Bagging e Random Forest

**Bagging (Bootstrap Aggregating):** treina N modelos em subconjuntos aleatórios (com reposição) do treino e combina as predições por votação/média.

**Random Forest = Bagging + Feature Randomness:**
- Cada árvore é treinada em um bootstrap sample
- Em cada nó, apenas $\sqrt{n}$ features são consideradas para divisão
- Isso descorrelaciona as árvores, reduzindo a variância

$$\text{Var}(\text{média}) = \frac{\rho\sigma^2 + (1-\rho)\sigma^2/n}{\text{→ descorrelação reduz o 1º termo}}$$

---

## 2. Implementação

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Classificação
cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    max_features='sqrt',   # padrão para classificação
    oob_score=True,        # estimativa out-of-bag gratuita
    n_jobs=-1,
    random_state=42
)
rf.fit(X_tr, y_tr)
print(f"OOB score:  {rf.oob_score_:.4f}")
print(f"Teste:      {rf.score(X_te, y_te):.4f}")

# Importância de features (impureza média)
feat_imp = pd.Series(rf.feature_importances_, index=cancer.feature_names)
feat_imp.sort_values(ascending=False).head(10).plot(kind='barh')
plt.title('Importância de Features (Random Forest)')
plt.tight_layout()
plt.show()

# Permutation Importance (mais confiável)
perm_imp = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=42)
perm_df = pd.DataFrame({
    'feature': cancer.feature_names,
    'importance': perm_imp.importances_mean
}).sort_values('importance', ascending=False)
print(perm_df.head(10))
```

---

## 3. Efeito de n_estimators

```python
train_scores, test_scores = [], []
n_range = [10, 25, 50, 100, 200, 300, 500]

for n in n_range:
    rf_n = RandomForestClassifier(n_estimators=n, oob_score=True, n_jobs=-1, random_state=42)
    rf_n.fit(X_tr, y_tr)
    train_scores.append(rf_n.score(X_tr, y_tr))
    test_scores.append(rf_n.score(X_te, y_te))

plt.plot(n_range, train_scores, label='Treino', marker='o')
plt.plot(n_range, test_scores,  label='Teste',  marker='s')
plt.xlabel('n_estimators'); plt.ylabel('Acurácia')
plt.title('Efeito de n_estimators no Random Forest')
plt.legend()
plt.show()
```

---

## 4. Vantagens Práticas

- Raramente precisa de normalização
- Robusto a outliers e features irrelevantes
- Estimativa OOB = validação cruzada gratuita
- Paralelizável (`n_jobs=-1`)
- **Um dos melhores modelos base para dados tabulares**

---

## Questões para Reflexão
1. Por que o OOB score é uma estimativa válida do erro de generalização?
2. A partir de quantas árvores o desempenho tende a estabilizar?
3. Qual a diferença entre importância por impureza e permutation importance?

## Referências
- Géron, cap. 7
- Faceli et al., cap. 6

---
*Próxima aula → [Aula 29: Gradient Boosting](aula-29-gradient-boosting.md)*
