# Prática 06 — Regressão Logística e SVM

**Módulo:** 05 | **Duração:** ~90 minutos | **Dataset:** Breast Cancer Wisconsin

## Objetivos
- Implementar classificação binária com Regressão Logística e SVM
- Ajustar threshold e analisar curvas ROC e Precisão-Recall
- Comparar modelos com métricas adequadas

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score
)

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Modelos
lr_pipe = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(C=1.0, max_iter=1000))])
svm_pipe = Pipeline([('sc', StandardScaler()), ('svm', SVC(C=1.0, kernel='rbf', probability=True))])

for name, model in [('Logistic Regression', lr_pipe), ('SVM RBF', svm_pipe)]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"\n{'='*40}")
    print(f"{name}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=cancer.target_names))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Curvas ROC
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for name, model in [('LR', lr_pipe), ('SVM', svm_pipe)]:
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc(fpr, tpr):.3f})')

axes[0].plot([0,1],[0,1],'k--')
axes[0].set(xlabel='FPR', ylabel='TPR', title='Curva ROC')
axes[0].legend()

# Ajuste de threshold
y_prob_lr = lr_pipe.predict_proba(X_test)[:, 1]
thresholds = np.arange(0.1, 0.9, 0.05)
from sklearn.metrics import f1_score
f1s = [f1_score(y_test, (y_prob_lr >= t).astype(int)) for t in thresholds]
best_t = thresholds[np.argmax(f1s)]
axes[1].plot(thresholds, f1s, marker='o')
axes[1].axvline(best_t, color='red', linestyle='--', label=f'Best t={best_t:.2f}')
axes[1].set(xlabel='Threshold', ylabel='F1', title='F1 vs Threshold')
axes[1].legend()

plt.tight_layout(); plt.show()
print(f"Melhor threshold: {best_t:.2f} | F1: {max(f1s):.4f}")
```

## Desafios Extras
1. Use `GridSearchCV` para otimizar `C` e `penalty` da Regressão Logística
2. Teste SVM com kernel linear, polinomial e RBF — qual se sai melhor?
3. Implemente F-beta score com β=2 (favorece recall) e compare os modelos
