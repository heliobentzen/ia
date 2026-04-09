# Aula 54 — Séries Temporais

> **Módulo 11 · Aplicações de ML em Problemas Reais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Decompor séries temporais em tendência, sazonalidade e resíduo
- Implementar previsão com Prophet e LSTM
- Avaliar modelos de forecasting com métricas adequadas

---

## 1. Decomposição de Séries Temporais

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Carregar dataset de exemplo (vendas mensais)
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=48, freq='M')
tendencia = np.linspace(100, 180, 48)
sazonalidade = 20 * np.sin(2 * np.pi * np.arange(48) / 12)
ruido = np.random.normal(0, 5, 48)
serie = pd.Series(tendencia + sazonalidade + ruido, index=dates)

# Decomposição
result = seasonal_decompose(serie, model='additive', period=12)
result.plot()
plt.tight_layout()
plt.show()

print(f"Variância da tendência: {result.trend.dropna().var():.2f}")
print(f"Variância sazonal:     {result.seasonal.var():.2f}")
print(f"Variância do resíduo:  {result.resid.dropna().var():.2f}")
```

---

## 2. Prophet — Previsão Rápida e Robusta

```python
from prophet import Prophet
import pandas as pd

# Prophet espera colunas 'ds' (data) e 'y' (valor)
df_prophet = serie.reset_index()
df_prophet.columns = ['ds', 'y']

model = Prophet(
    seasonality_mode='additive',
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.1  # controla a flexibilidade da tendência
)
model.fit(df_prophet)

# Previsão futura
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

model.plot(forecast)
plt.title('Previsão com Prophet')
plt.show()

model.plot_components(forecast)
plt.show()
```

---

## 3. XGBoost para Séries Temporais

```python
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

def create_lag_features(series, lags=12, rolling_windows=[3, 6]):
    df = pd.DataFrame({'y': series})
    for lag in range(1, lags+1):
        df[f'lag_{lag}'] = df['y'].shift(lag)
    for w in rolling_windows:
        df[f'roll_mean_{w}'] = df['y'].rolling(w).mean().shift(1)
        df[f'roll_std_{w}']  = df['y'].rolling(w).std().shift(1)
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df = df.dropna()
    return df

df_feat = create_lag_features(serie)
X = df_feat.drop('y', axis=1)
y = df_feat['y']

split = int(0.8 * len(X))
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

xgb_ts = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
xgb_ts.fit(X_tr, y_tr)
y_pred = xgb_ts.predict(X_te)
mae = mean_absolute_error(y_te, y_pred)
print(f"XGBoost — MAE: {mae:.2f}")
```

---

## 4. Métricas para Forecasting

| Métrica | Fórmula | Interpretação |
|---------|---------|--------------|
| MAE | $\frac{1}{T}\sum|y_t - \hat{y}_t|$ | Erro médio absoluto |
| RMSE | $\sqrt{\frac{1}{T}\sum(y_t-\hat{y}_t)^2}$ | Penaliza grandes erros |
| MAPE | $\frac{100}{T}\sum\|\frac{y_t-\hat{y}_t}{y_t}\|$ | Erro em % |
| sMAPE | Symmetric MAPE | Evita assimetria do MAPE |

---

## Questões para Reflexão
1. Por que TimeSeriesSplit é obrigatório para validação cruzada em séries temporais?
2. Como você detectaria e trataria um ponto de mudança estrutural (changepoint) na série?
3. Quando Prophet é preferível ao LSTM para previsão?

## Referências
- Géron, cap. 15
- Faceli et al., cap. 10

---
*Próxima aula → [Aula 55: Sistemas de Recomendação](aula-55-sistemas-recomendacao.md)*
