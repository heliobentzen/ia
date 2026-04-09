# Aula 40 — Curvas de Aprendizado e Diagnóstico

> **Módulo 08 · Avaliação e Validação de Modelos** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Construir e interpretar curvas de aprendizado
- Diagnosticar overfitting e underfitting visualmente
- Usar curvas de validação para escolher hiperparâmetros

---

## 1. Curvas de Aprendizado

Mostram como o erro de treino e validação evoluem com o aumento dos dados de treinamento.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

def plot_learning_curve(estimator, X, y, title, cv=5):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring='f1',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)
    
    plt.plot(train_sizes, train_mean, 'o-', label='Treino')
    plt.fill_between(train_sizes, train_mean-train_std, train_mean+train_std, alpha=0.15)
    plt.plot(train_sizes, val_mean, 's-', label='Validação')
    plt.fill_between(train_sizes, val_mean-val_std, val_mean+val_std, alpha=0.15)
    plt.xlabel('Tamanho do conjunto de treino')
    plt.ylabel('F1 Score')
    plt.title(title)
    plt.legend(); plt.grid(alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Modelo simples (provável underfitting)
plt.sca(axes[0])
pipe_lr = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=500))])
plot_learning_curve(pipe_lr, X, y, 'Regressão Logística (modelo simples)')

# Modelo complexo (possível overfitting)
plt.sca(axes[1])
pipe_rf = Pipeline([('sc', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42))])
plot_learning_curve(pipe_rf, X, y, 'Random Forest (modelo complexo)')

plt.tight_layout(); plt.show()
```

---

## 2. Interpretando as Curvas

```
UNDERFITTING (bias alto):
  Treino e Validação convergem para erro ALTO

OVERFITTING (variância alta):
  Treino ≈ 0 | Validação >> Treino | Grande gap

BOAS CURVAS:
  Treino decresce suavemente
  Validação sobe e aproxima do treino
  Pequeno gap final
```

---

## 3. Curvas de Validação — Escolha de Hiperparâmetro

```python
from sklearn.svm import SVC

pipe_svm = Pipeline([('sc', StandardScaler()), ('svc', SVC())])

param_range = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
train_scores, val_scores = validation_curve(
    pipe_svm, X, y,
    param_name='svc__C',
    param_range=param_range,
    cv=5, scoring='f1', n_jobs=-1
)

plt.semilogx(param_range, train_scores.mean(axis=1), 'o-', label='Treino')
plt.semilogx(param_range, val_scores.mean(axis=1), 's-', label='Validação')
plt.fill_between(param_range, val_scores.mean(axis=1)-val_scores.std(axis=1),
                 val_scores.mean(axis=1)+val_scores.std(axis=1), alpha=0.2)
plt.xlabel('C (log scale)'); plt.ylabel('F1 Score')
plt.title('Curva de Validação — SVM (C)'); plt.legend()
plt.show()
```

---

## Questões para Reflexão
1. Se treino e validação convergem em erro alto, mais dados ajudarão?
2. Se o gap entre treino e validação é grande, o que tentar?
3. Como a curva de validação difere da curva de aprendizado?

## Referências
- Géron, cap. 2, 4
- Faceli et al., cap. 4

---
*Próxima aula → [Aula 41: Comparação de Modelos](aula-41-comparacao-modelos.md)*
