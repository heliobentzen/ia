# Aula 22 — Regressão Polinomial e Regularizada

> **Módulo 05 · Regressão e Classificação** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Estender a regressão linear para relações não-lineares com features polinomiais
- Aplicar Ridge (L2), Lasso (L1) e ElasticNet para controle de overfitting
- Interpretar o papel do hiperparâmetro de regularização $\alpha$

---

## 1. Regressão Polinomial

Transformamos features: $[x] \to [x, x^2, x^3, ...]$ e aplicamos regressão linear.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import numpy as np

X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5*X**3 - X**2 + 2*X + np.random.normal(0, 1, (100, 1))
y = y.ravel()

for grau in [1, 2, 3, 10]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=grau, include_bias=False)),
        ('linear', LinearRegression())
    ])
    pipe.fit(X, y)
    print(f"Grau {grau}: R² treino = {pipe.score(X, y):.4f}")
```

---

## 2. Ridge (L2) — Regularização com Norma Euclidiana

$$J(\boldsymbol{\theta}) = \text{MSE} + \alpha \sum_{j=1}^{n}\theta_j^2$$

- Encolhe coeficientes em direção a zero, mas não os zera
- Útil quando todas as features contribuem um pouco

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    score = cross_val_score(ridge, X, y, cv=5, scoring='neg_mean_squared_error').mean()
    print(f"alpha={alpha:.3f}: CV MSE = {-score:.4f}")
```

---

## 3. Lasso (L1) — Regularização com Norma Manhattan

$$J(\boldsymbol{\theta}) = \text{MSE} + \alpha \sum_{j=1}^{n}|\theta_j|$$

- Pode zerar coeficientes completamente → **seleção automática de features**
- Útil quando há muitas features irrelevantes (dados esparsos)

```python
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
n_zeros = (lasso.coef_ == 0).sum()
print(f"Features zeradas pelo Lasso: {n_zeros}")
```

---

## 4. ElasticNet — Combinação L1 + L2

$$J(\boldsymbol{\theta}) = \text{MSE} + r\cdot\alpha\sum|\theta_j| + \frac{1-r}{2}\cdot\alpha\sum\theta_j^2$$

Parâmetros: `alpha` (força) e `l1_ratio` (proporção L1).

```python
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
en.fit(X_train, y_train)
```

---

## 5. Resumo

| Método | Penalidade | Zera coefs? | Quando usar |
|--------|-----------|-------------|------------|
| Ridge | $\|\boldsymbol{\theta}\|_2^2$ | Não | Multicolinearidade |
| Lasso | $\|\boldsymbol{\theta}\|_1$ | Sim | Features irrelevantes |
| ElasticNet | L1 + L2 | Sim | Alto número de features correlacionadas |

---

## Questões para Reflexão
1. Por que Lasso pode produzir soluções esparsas e Ridge não?
2. Como você escolheria o melhor `alpha` para Ridge em produção?
3. Quando ElasticNet é preferível a usar Lasso ou Ridge separadamente?

## Referências
- Géron, cap. 4
- Faceli et al., cap. 5

---
*Próxima aula → [Aula 23: Regressão Logística](aula-23-regressao-logistica.md)*
