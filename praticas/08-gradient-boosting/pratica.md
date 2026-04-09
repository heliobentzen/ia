# Prática 08 — Gradient Boosting com XGBoost e LightGBM

**Módulo:** 06 | **Duração:** ~90 minutos

## Objetivos
- Treinar XGBoost e LightGBM em dados tabulares
- Usar early stopping e ajuste de hiperparâmetros
- Comparar boosting com Random Forest

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import matplotlib.pyplot as plt
import optuna

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)
X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)

# XGBoost com early stopping
xgb_model = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, gamma=1,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
xgb_model.fit(X_tr_s, y_tr_s,
              eval_set=[(X_val_s, y_val_s)],
              early_stopping_rounds=30, verbose=False)

xgb_auc = roc_auc_score(y_te, xgb_model.predict_proba(X_te)[:, 1])
print(f"XGBoost — AUC: {xgb_auc:.4f} (melhor iteração: {xgb_model.best_iteration})")

# LightGBM
lgb_model = lgb.LGBMClassifier(
    n_estimators=1000, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8, n_jobs=-1, random_state=42
)
lgb_model.fit(X_tr_s, y_tr_s,
              eval_set=[(X_val_s, y_val_s)],
              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

lgb_auc = roc_auc_score(y_te, lgb_model.predict_proba(X_te)[:, 1])
print(f"LightGBM — AUC: {lgb_auc:.4f}")

# Otimização com Optuna
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
    }
    model = xgb.XGBClassifier(**params, use_label_encoder=False,
                               eval_metric='logloss', random_state=42)
    score = cross_val_score(model, X_tr, y_tr, cv=5, scoring='roc_auc').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)
print(f"Melhor AUC Optuna: {study.best_value:.4f}")
print(f"Melhores params: {study.best_params}")
```

## Desafios Extras
1. Use SHAP com XGBoost para explicar as predições no conjunto de teste
2. Implemente stacking: XGBoost + LightGBM com meta-modelo LogisticRegression
3. Teste CatBoost e compare com XGBoost e LightGBM
