# Prática 10 — Pipeline Completo de Avaliação

**Módulo:** 08 | **Duração:** ~90 minutos

## Objetivos
- Implementar validação cruzada estratificada com múltiplas métricas
- Construir curvas de aprendizado e de validação
- Comparar modelos com rigor estatístico

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    cross_validate, StratifiedKFold, learning_curve, validation_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, f1_score
from scipy.stats import wilcoxon

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ['accuracy', 'f1', 'roc_auc', 'precision', 'recall']

models = {
    'LR': Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))]),
    'RF': Pipeline([('sc', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, random_state=42))]),
    'SVM': Pipeline([('sc', StandardScaler()), ('svm', SVC(probability=True, random_state=42))]),
    'GBM': Pipeline([('sc', StandardScaler()), ('gbm', GradientBoostingClassifier(n_estimators=100, random_state=42))]),
}

# Tabela comparativa
results = {}
for name, model in models.items():
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    results[name] = {m: cv_results[f'test_{m}'].mean() for m in scoring}

df_results = pd.DataFrame(results).T.round(4)
print(df_results)

# Curvas de aprendizado
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, model, ax in [('Random Forest', models['RF'], axes[0]),
                          ('Logistic Reg', models['LR'], axes[1])]:
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1
    )
    ax.plot(train_sizes, train_scores.mean(1), 'o-', label='Treino')
    ax.plot(train_sizes, val_scores.mean(1), 's-', label='Validação')
    ax.fill_between(train_sizes, val_scores.mean(1)-val_scores.std(1),
                    val_scores.mean(1)+val_scores.std(1), alpha=0.15)
    ax.set_title(f'Curva de Aprendizado — {name}')
    ax.set_xlabel('Tamanho do treino'); ax.set_ylabel('F1')
    ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout(); plt.show()

# Teste de Wilcoxon: RF vs GBM
scores_rf  = cross_validate(models['RF'],  X, y, cv=StratifiedKFold(30, shuffle=True, random_state=0), scoring='f1')['test_score']
scores_gbm = cross_validate(models['GBM'], X, y, cv=StratifiedKFold(30, shuffle=True, random_state=0), scoring='f1')['test_score']
stat, p = wilcoxon(scores_rf, scores_gbm)
print(f"\nWilcoxon RF vs GBM: p={p:.4f} → {'diferença significativa' if p < 0.05 else 'sem diferença significativa'}")
```

## Desafios Extras
1. Implemente nested cross-validation para estimar o erro com seleção de hiperparâmetros
2. Calcule e plote o intervalo de confiança de 95% para cada métrica
3. Use `DummyClassifier` como baseline e compare com os modelos reais
