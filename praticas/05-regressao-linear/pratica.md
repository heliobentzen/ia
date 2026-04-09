# Prática 05 — Regressão Linear

**Módulo:** 05 | **Duração:** ~90 minutos | **Dataset:** California Housing

## Objetivos
- Implementar e avaliar regressão linear e regularizada
- Analisar resíduos e verificar suposições
- Comparar Ridge, Lasso e ElasticNet

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

housing = fetch_california_housing(as_frame=True)
X, y = housing.data, housing.target
print(f"Features: {X.columns.tolist()}")
print(f"Target: preço mediano (×$100k) — {y.describe()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Comparar modelos
models = {
    'Linear Regression': Pipeline([('sc', StandardScaler()), ('lr', LinearRegression())]),
    'Ridge (α=1)': Pipeline([('sc', StandardScaler()), ('ridge', Ridge(alpha=1.0))]),
    'Lasso (α=0.01)': Pipeline([('sc', StandardScaler()), ('lasso', Lasso(alpha=0.01))]),
    'ElasticNet': Pipeline([('sc', StandardScaler()), ('en', ElasticNet(alpha=0.01, l1_ratio=0.5))]),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    }

print(pd.DataFrame(results).T.round(4))

# Análise de resíduos
lr_pipe = models['Linear Regression']
y_pred_lr = lr_pipe.predict(X_test)
residuos = y_test - y_pred_lr

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].scatter(y_pred_lr, residuos, alpha=0.3, s=10)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set(xlabel='Predito', ylabel='Resíduo', title='Resíduos vs Predito')

axes[1].hist(residuos, bins=50)
axes[1].set(xlabel='Resíduo', title='Distribuição dos Resíduos')

axes[2].scatter(y_test, y_pred_lr, alpha=0.3, s=10)
axes[2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes[2].set(xlabel='Real', ylabel='Predito', title='Real vs Predito')

plt.tight_layout(); plt.show()

# GridSearch para Ridge
param_grid = {'ridge__alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge_pipe = Pipeline([('sc', StandardScaler()), ('ridge', Ridge())])
grid = GridSearchCV(ridge_pipe, param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid.fit(X_train, y_train)
print(f"Melhor alpha: {grid.best_params_['ridge__alpha']} | RMSE: {-grid.best_score_:.4f}")
```

## Desafios Extras
1. Adicione features polinomiais de grau 2 e compare com o modelo linear
2. Plote os coeficientes do Lasso para diferentes valores de alpha (caminho de regularização)
3. Implemente validação cruzada com `TimeSeriesSplit` (dados ordenados por localização)
