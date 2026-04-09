# Aula 39 — Validação Cruzada

> **Módulo 08 · Avaliação e Validação de Modelos** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Implementar K-Fold, Stratified K-Fold, LOOCV e Time Series CV
- Interpretar médias e desvios padrão dos scores de CV
- Evitar data leakage em validação cruzada com pipelines

---

## 1. Por que Validação Cruzada?

Um único split treino/teste pode ser favorável ou desfavorável por acaso. A CV usa **todos os dados** para avaliação:

```
K-Fold (K=5):
Fold 1: [Te][Tr][Tr][Tr][Tr]
Fold 2: [Tr][Te][Tr][Tr][Tr]
Fold 3: [Tr][Tr][Te][Tr][Tr]
Fold 4: [Tr][Tr][Tr][Te][Tr]
Fold 5: [Tr][Tr][Tr][Tr][Te]
Score final: média ± desvio padrão
```

---

## 2. Implementação

```python
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    TimeSeriesSplit, cross_val_score, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import numpy as np

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])

# K-Fold estratificado (mantém proporção de classes em cada fold)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=skf, scoring='f1', n_jobs=-1)
print(f"Stratified 10-Fold F1: {scores.mean():.4f} ± {scores.std():.4f}")

# Múltiplas métricas de uma vez
result = cross_validate(
    pipe, X, y, cv=skf,
    scoring=['accuracy', 'f1', 'roc_auc'],
    return_train_score=True, n_jobs=-1
)
for metric in ['accuracy', 'f1', 'roc_auc']:
    tr = result[f'train_{metric}'].mean()
    te = result[f'test_{metric}'].mean()
    print(f"{metric:12s}: treino={tr:.4f} | teste={te:.4f}")
```

---

## 3. Validação Cruzada para Séries Temporais

```python
import pandas as pd
import numpy as np

# TimeSeriesSplit: sempre treina no passado, valida no futuro
tscv = TimeSeriesSplit(n_splits=5)

print("TimeSeriesSplit splits:")
for i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
    print(f"  Fold {i+1}: treino={len(tr_idx):4d} | teste={len(te_idx):3d}")

scores_ts = cross_val_score(pipe, X, y, cv=tscv, scoring='accuracy')
print(f"\nTime Series CV: {scores_ts.mean():.4f} ± {scores_ts.std():.4f}")
```

---

## 4. Nested Cross-Validation

Estimativa imparcial quando também fazemos seleção de hiperparâmetros:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'rf__n_estimators': [50, 100, 200], 'rf__max_depth': [None, 5, 10]}

# CV externa: avalia a generalização
# CV interna: seleciona hiperparâmetros
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

grid = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring='f1', n_jobs=-1)
nested_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring='f1', n_jobs=-1)
print(f"Nested CV F1: {nested_scores.mean():.4f} ± {nested_scores.std():.4f}")
```

---

## Questões para Reflexão
1. Por que LOOCV tem alta variância e não é sempre recomendado?
2. O que acontece se você escalar os dados ANTES de fazer o split de CV?
3. Por que usar TimeSeriesSplit em vez de KFold para dados temporais?

## Referências
- Géron, cap. 2
- Faceli et al., cap. 4

---
*Próxima aula → [Aula 40: Curvas de Aprendizado](aula-40-curvas-aprendizado.md)*
