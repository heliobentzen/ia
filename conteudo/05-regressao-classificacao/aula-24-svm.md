# Aula 24 — Support Vector Machines (SVM)

> **Módulo 05 · Regressão e Classificação** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender a geometria do SVM e o conceito de margem máxima
- Aplicar kernels para separação não-linear
- Usar SVM para classificação e regressão com Scikit-learn

---

## 1. Intuição: Margem Máxima

O SVM encontra o **hiperplano que maximiza a margem** entre as classes. Os pontos mais próximos do hiperplano são os **vetores de suporte**.

$$\text{Margem} = \frac{2}{\|\mathbf{w}\|}$$

**Problema de otimização:**
$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|^2 \quad \text{sujeito a } y^{(i)}(\mathbf{w}^T\mathbf{x}^{(i)} + b) \geq 1$$

---

## 2. Soft Margin — SVM com erros tolerados

Na prática, dados raramente são separáveis linearmente. Introduzimos variáveis de folga $\xi^{(i)} \geq 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C\sum_{i=1}^{m}\xi^{(i)}$$

- **C pequeno**: margem maior, mais erros tolerados (underfitting)
- **C grande**: margem menor, menos erros tolerados (overfitting)

---

## 3. Kernel Trick — Separação Não-Linear

Mapeia os dados para espaço de maior dimensão sem calcular explicitamente a transformação:

$$K(\mathbf{x}, \mathbf{x}') = \phi(\mathbf{x})^T\phi(\mathbf{x}')$$

| Kernel | Fórmula | Quando usar |
|--------|---------|-------------|
| Linear | $\mathbf{x}^T\mathbf{x}'$ | Dados linearmente separáveis, alta dimensão |
| Polinomial | $(\gamma\mathbf{x}^T\mathbf{x}' + r)^d$ | Relações polinomiais |
| RBF (Gaussian) | $\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$ | Uso geral (padrão) |
| Sigmoide | $\tanh(\gamma\mathbf{x}^T\mathbf{x}' + r)$ | Redes neurais |

---

## 4. Implementação

```python
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline com SVM RBF
pipe_svm = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1.0, gamma='scale', probability=True))
])

# GridSearch para C e gamma
param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 'auto', 0.01, 0.1]
}
grid = GridSearchCV(pipe_svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Acurácia no teste: {grid.score(X_test, y_test):.4f}")
```

---

## 5. SVR — SVM para Regressão

```python
from sklearn.svm import SVR
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X_h, y_h = housing.data[:2000], housing.target[:2000]
X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=10, epsilon=0.1))
])
svr_pipe.fit(X_tr, y_tr)
print(f"R² SVR: {svr_pipe.score(X_te, y_te):.4f}")
```

---

## Questões para Reflexão
1. Por que o SVM é sensível à escala das features?
2. Como o parâmetro `gamma` afeta a "suavidade" do kernel RBF?
3. Em que situações o SVM pode ser preferível ao Random Forest?

## Referências
- Géron, cap. 5
- Faceli et al., cap. 5

---
*Próxima aula → [Aula 25: Naive Bayes](aula-25-naive-bayes.md)*
