# Aula 31 — Stacking e Blending

> **Módulo 06 · Métodos Baseados em Árvores e Ensembles** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender os princípios de voting, bagging, stacking e blending
- Implementar StackingClassifier com Scikit-learn
- Identificar quando ensembles avançados compensam a complexidade adicional

---

## 1. Voting Ensemble

Combina modelos diferentes por votação (hard ou soft):

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

# Soft voting: usa probabilidades (melhor que hard voting)
voting = VotingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ],
    voting='soft'
)
voting.fit(X_tr, y_tr)
print(f"Voting Ensemble: {voting.score(X_te, y_te):.4f}")
```

---

## 2. Stacking — Meta-Learner

Os modelos base (nível 0) geram predições que servem como features para um meta-modelo (nível 1).

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import xgboost as xgb

estimators = [
    ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
    ('gb',  GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=7))
]

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(C=1.0, max_iter=1000),
    cv=5,  # cross-val para gerar OOF predictions
    passthrough=False,
    n_jobs=-1
)

stacking.fit(X_tr, y_tr)
print(f"Stacking: {stacking.score(X_te, y_te):.4f}")

# Comparar individualmente
for name, est in estimators:
    score = cross_val_score(est, X_tr, y_tr, cv=5, scoring='accuracy').mean()
    print(f"{name}: {score:.4f}")
```

---

## 3. Blending — Validação Holdout

Variante mais simples do stacking: usa um holdout em vez de cross-validation para gerar as predições do nível 0.

```python
from sklearn.model_selection import train_test_split

# Split: treino L0 | blend | treino L1 | teste
X_tr0, X_blend, y_tr0, y_blend = train_test_split(X_tr, y_tr, test_size=0.3, random_state=42)
X_blend_final, X_te_final, y_blend_final, y_te_final = train_test_split(X_te, y_te, test_size=0.5, random_state=42)

# Treinar modelos base no L0
rf_b = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr0, y_tr0)
gb_b = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_tr0, y_tr0)

# Gerar features de blend
import numpy as np
blend_features = np.column_stack([
    rf_b.predict_proba(X_blend)[:, 1],
    gb_b.predict_proba(X_blend)[:, 1]
])

# Meta-modelo
meta = LogisticRegression().fit(blend_features, y_blend)

# Predição no teste
test_features = np.column_stack([
    rf_b.predict_proba(X_te)[:, 1],
    gb_b.predict_proba(X_te)[:, 1]
])
print(f"Blending: {meta.score(test_features, y_te):.4f}")
```

---

## 4. Quando Usar Ensembles Avançados?

- Competições de ML (Kaggle, etc.)
- Alto custo de erro; máxima acurácia necessária
- Modelos base são **diversificados** (erram em exemplos diferentes)
- Dados suficientes para evitar overfitting no meta-modelo

---

## Questões para Reflexão
1. Por que os modelos base precisam ser **diversificados** para o stacking funcionar bem?
2. Qual é o risco de overfitting no stacking e como o CV com OOF mitiga isso?
3. Em produção, como você gerenciaria a complexidade de um ensemble com stacking?

## Referências
- Géron, cap. 7
- Faceli et al., cap. 6

---
*Módulo 06 concluído! Próximo → [Módulo 07: Redes Neurais Artificiais](../07-redes-neurais/README.md)*
