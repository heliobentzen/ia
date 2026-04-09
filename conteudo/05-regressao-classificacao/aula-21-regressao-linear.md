# Aula 21 — Regressão Linear

> **Módulo 05 · Regressão e Classificação** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Derivar a equação normal e compreender o gradiente descendente
- Verificar as suposições da regressão linear
- Implementar e interpretar regressão linear simples e múltipla

---

## 1. Modelo

$$\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \boldsymbol{\theta}^T \mathbf{x}$$

**Função de custo (MSE):**
$$J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2$$

---

## 2. Equação Normal (solução analítica)

$$\boldsymbol{\hat{\theta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$$

Complexidade: $O(n^3)$ — inviável para muitas features.

---

## 3. Gradiente Descendente

$$\theta_j \leftarrow \theta_j - \alpha \frac{\partial J}{\partial \theta_j} = \theta_j - \frac{\alpha}{m}\sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})x_j^{(i)}$$

Onde $\alpha$ é a **taxa de aprendizado**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regressão Linear (equação normal internamente)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.4f} | R²: {r2:.4f}")

# Interpretação dos coeficientes
feature_names = housing.feature_names
for name, coef in zip(feature_names, lr.coef_):
    print(f"  {name:20s}: {coef:+.4f}")
print(f"  {'Intercepto':20s}: {lr.intercept_:+.4f}")
```

---

## 4. Suposições da Regressão Linear (LINE)

| Letra | Suposição | Como verificar |
|-------|-----------|----------------|
| **L** | Linearidade | Gráfico resíduos vs. predito |
| **I** | Independência | Teste Durbin-Watson |
| **N** | Normalidade dos resíduos | QQ-plot, teste Shapiro-Wilk |
| **E** | Homocedasticidade | Gráfico resíduos vs. predito (dispersão uniforme) |

```python
# Análise de resíduos
residuos = y_test - y_pred

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].scatter(y_pred, residuos, alpha=0.3, s=10)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('Valores Preditos')
axes[0].set_ylabel('Resíduos')
axes[0].set_title('Resíduos vs. Predito')

axes[1].hist(residuos, bins=50, edgecolor='black')
axes[1].set_xlabel('Resíduo')
axes[1].set_title('Distribuição dos Resíduos')

plt.tight_layout()
plt.show()
```

---

## Questões para Reflexão
1. Quando usar gradiente descendente estocástico (SGD) em vez da equação normal?
2. O R² pode ser negativo? O que isso significa?
3. Como multicolinearidade afeta os coeficientes da regressão linear?

## Referências
- Géron, cap. 4
- Faceli et al., cap. 5

---
*Próxima aula → [Aula 22: Regressão Polinomial e Regularizada](aula-22-regressao-polinomial-regularizada.md)*
