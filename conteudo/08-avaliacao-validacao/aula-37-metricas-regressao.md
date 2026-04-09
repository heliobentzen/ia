# Aula 37 — Métricas de Regressão

> **Módulo 08 · Avaliação e Validação de Modelos** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Calcular e interpretar MAE, MSE, RMSE, R² e MAPE
- Escolher a métrica adequada para cada contexto de negócio
- Identificar as limitações de cada métrica

---

## 1. Principais Métricas

### MAE — Mean Absolute Error
$$\text{MAE} = \frac{1}{m}\sum_{i=1}^{m}|y^{(i)} - \hat{y}^{(i)}|$$
- Interpretação direta (mesma unidade do target)
- Robusto a outliers

### MSE — Mean Squared Error
$$\text{MSE} = \frac{1}{m}\sum_{i=1}^{m}(y^{(i)} - \hat{y}^{(i)})^2$$
- Penaliza erros grandes mais fortemente
- Sensível a outliers

### RMSE — Root Mean Squared Error
$$\text{RMSE} = \sqrt{\text{MSE}}$$
- Mesma unidade do target
- **Mais usada em competições (ex: Kaggle)**

### R² — Coeficiente de Determinação
$$R^2 = 1 - \frac{\sum(y^{(i)} - \hat{y}^{(i)})^2}{\sum(y^{(i)} - \bar{y})^2}$$
- Proporção da variância explicada pelo modelo
- R²=1 perfeito, R²=0 igual à média, R²<0 pior que a média

### MAPE — Mean Absolute Percentage Error
$$\text{MAPE} = \frac{100}{m}\sum_{i=1}^{m}\left|\frac{y^{(i)} - \hat{y}^{(i)}}{y^{(i)}}\right|$$
- Interpretável em porcentagem
- Problemático quando $y \approx 0$

---

## 2. Implementação

```python
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              r2_score, mean_absolute_percentage_error)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
}

for name, model in models.items():
    model.fit(X_tr_s, y_tr)
    y_pred = model.predict(X_te_s)
    
    mae  = mean_absolute_error(y_te, y_pred)
    mse  = mean_squared_error(y_te, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_te, y_pred)
    mape = mean_absolute_percentage_error(y_te, y_pred) * 100
    
    print(f"\n{name}")
    print(f"  MAE:  {mae:.4f} ($100k)")
    print(f"  RMSE: {rmse:.4f} ($100k)")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.1f}%")
```

---

## 3. Quando Usar Cada Métrica?

| Situação | Métrica Recomendada |
|---------|---------------------|
| Erros simétricos, sem outliers | MAE |
| Penalizar erros grandes | RMSE |
| Comparar entre targets diferentes | R² |
| Contexto de negócio em % | MAPE |
| Otimização de modelos lineares | MSE |

---

## Questões para Reflexão
1. Um modelo com R²=0.85 é bom? Depende do quê?
2. Por que usar RMSE em vez de MSE para comparar modelos?
3. Em detecção de fraude com valores muito distintos, qual métrica evitar?

## Referências
- Géron, cap. 2
- Faceli et al., cap. 4

---
*Próxima aula → [Aula 38: Métricas de Classificação](aula-38-metricas-classificacao.md)*
