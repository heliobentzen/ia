# Aula 26 — Ajuste de Hiperparâmetros

> **Módulo 05 · Regressão e Classificação** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Distinguir parâmetros de hiperparâmetros
- Aplicar GridSearchCV, RandomizedSearchCV e Optuna
- Evitar data leakage no processo de tuning

---

## 1. Parâmetros vs. Hiperparâmetros

| | Parâmetros | Hiperparâmetros |
|--|-----------|----------------|
| Aprendidos por | Treinamento (gradiente) | Humano / busca |
| Exemplos | $\mathbf{w}$, $b$ da regressão | `C`, `gamma`, `n_estimators` |
| Quando definidos | Durante o fit | Antes do fit |

---

## 2. GridSearchCV — Busca Exaustiva

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(random_state=42))
])

param_grid = {
    'svm__kernel': ['rbf', 'linear'],
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.01, 0.001]
}

grid = GridSearchCV(pipe, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid.fit(X_tr, y_tr)

print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Melhor F1 (CV):      {grid.best_score_:.4f}")
print(f"F1 no teste:         {grid.score(X_te, y_te):.4f}")
```

**Problema:** escala exponencial — 2 × 4 × 3 = 24 combinações × 5 folds = 120 treinamentos.

---

## 3. RandomizedSearchCV — Busca Aleatória

Mais eficiente para espaços grandes de busca:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint
from sklearn.ensemble import RandomForestClassifier

rf_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42))
])

param_dist = {
    'rf__n_estimators': randint(50, 500),
    'rf__max_depth': [None, 5, 10, 20],
    'rf__min_samples_split': randint(2, 20),
    'rf__max_features': ['sqrt', 'log2', 0.3, 0.5]
}

random_search = RandomizedSearchCV(
    rf_pipe, param_dist, n_iter=50, cv=5,
    scoring='f1', n_jobs=-1, random_state=42
)
random_search.fit(X_tr, y_tr)
print(f"Melhores parâmetros: {random_search.best_params_}")
print(f"F1 no teste: {random_search.score(X_te, y_te):.4f}")
```

---

## 4. Optuna — Otimização Bayesiana Moderna

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0)
    }
    model = GradientBoostingClassifier(**params, random_state=42)
    score = cross_val_score(model, X_tr, y_tr, cv=5, scoring='f1').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, show_progress_bar=True)

print(f"Melhores parâmetros: {study.best_params}")
print(f"Melhor F1: {study.best_value:.4f}")
```

---

## 5. Boas Práticas

1. **Sempre use pipeline** — evita data leakage no tuning
2. **Separe um conjunto de teste final** — não use para selecionar hiperparâmetros
3. **Comece com RandomSearch**, depois refine com GridSearch em região menor
4. **Use nested cross-validation** para estimativa imparcial do erro

---

## Questões para Reflexão
1. O que é data leakage no contexto de ajuste de hiperparâmetros?
2. Por que RandomizedSearch às vezes supera GridSearch?
3. Como a otimização bayesiana difere de busca aleatória?

## Referências
- Géron, cap. 2

---
*Módulo 05 concluído! Próximo → [Módulo 06: Árvores e Ensembles](../06-arvores-ensembles/README.md)*
