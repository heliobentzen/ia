# Aula 43 — Regularização L1 e L2

> **Módulo 09 · Overfitting e Regularização** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender geometricamente por que L1 gera esparsidade e L2 não
- Aplicar regularização em modelos lineares e redes neurais
- Usar regularização em redes neurais com Keras

---

## 1. Intuição Geométrica

A regularização adiciona uma **penalidade à complexidade** do modelo:

$$J_{\text{reg}}(\boldsymbol{\theta}) = J(\boldsymbol{\theta}) + \alpha\Omega(\boldsymbol{\theta})$$

**Por que L1 zera coeficientes?**
A restrição L1 forma um diamante em 2D. O mínimo da função de custo tende a "tocar" o canto do diamante, onde uma coordenada é zero.

**Por que L2 não zera?**
A restrição L2 forma uma esfera. O mínimo raramente toca um eixo.

---

## 2. Regularização em Modelos Lineares

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Dataset com muitas features (muitas irrelevantes)
from sklearn.datasets import make_regression
X, y, coef = make_regression(n_samples=200, n_features=50, n_informative=10,
                               noise=20, coef=True, random_state=42)

scaler = StandardScaler()
X_s = scaler.fit_transform(X)

alphas = np.logspace(-3, 3, 50)
ridge_coefs, lasso_coefs = [], []

for alpha in alphas:
    ridge = Ridge(alpha=alpha).fit(X_s, y)
    lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_s, y)
    ridge_coefs.append(ridge.coef_)
    lasso_coefs.append(lasso.coef_)

ridge_coefs = np.array(ridge_coefs)
lasso_coefs = np.array(lasso_coefs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for i in range(50):
    axes[0].plot(alphas, ridge_coefs[:, i], alpha=0.5, linewidth=0.5)
    axes[1].plot(alphas, lasso_coefs[:, i], alpha=0.5, linewidth=0.5)

axes[0].set_xscale('log'); axes[0].set_title('Ridge (L2): coeficientes encolhem mas nunca zeram')
axes[1].set_xscale('log'); axes[1].set_title('Lasso (L1): coeficientes zeram progressivamente')
for ax in axes:
    ax.set_xlabel('Alpha (regularização)'); ax.set_ylabel('Valor do coeficiente')
    ax.axhline(0, color='black', linewidth=0.5)
plt.tight_layout(); plt.show()
```

---

## 3. Regularização em Redes Neurais (Keras)

```python
import tensorflow as tf

def build_model(regularizer=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation='relu',
                              kernel_regularizer=regularizer,
                              input_shape=(20,)),
        tf.keras.layers.Dense(128, activation='relu',
                              kernel_regularizer=regularizer),
        tf.keras.layers.Dense(1)
    ])

# Sem regularização
m_none = build_model()

# L2
m_l2 = build_model(tf.keras.regularizers.l2(1e-4))

# L1
m_l1 = build_model(tf.keras.regularizers.l1(1e-4))

# L1 + L2
m_l1l2 = build_model(tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4))

for name, m in [('Sem reg', m_none), ('L2', m_l2), ('L1', m_l1), ('L1+L2', m_l1l2)]:
    m.compile('adam', 'mse')
    print(f"{name}: {sum(tf.size(v).numpy() for v in m.trainable_variables):,} parâmetros")
```

---

## Questões para Reflexão
1. Por que regularizar o bias (intercepto) geralmente não é recomendado?
2. Em redes neurais, qual técnica de regularização é mais comum: L2, Dropout ou ambas?
3. Como você escolheria entre L1 e L2 para um problema com 1000 features?

## Referências
- Géron, cap. 4, 11
- Faceli et al., cap. 3

---
*Próxima aula → [Aula 44: Dropout e Early Stopping](aula-44-dropout-early-stopping.md)*
