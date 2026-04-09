# Aula 30 — XGBoost, LightGBM e CatBoost

> **Módulo 06 · Métodos Baseados em Árvores e Ensembles** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender as inovações do XGBoost sobre o GBM clássico
- Aplicar LightGBM em datasets de grande escala
- Usar CatBoost com features categóricas sem codificação manual

---

## 1. XGBoost — Extreme Gradient Boosting

**Inovações principais:**
- Regularização L1/L2 na função objetivo
- Cálculo eficiente via second-order gradient (Hessiana)
- Suporte a dados esparsos e valores ausentes
- Paralelização eficiente

$$\mathcal{L} = \sum_i \ell(\hat{y}_i, y_i) + \sum_k \Omega(f_k), \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda\|\mathbf{w}\|^2$$

```python
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    reg_alpha=0.1,       # L1
    reg_lambda=1.0,      # L2
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

# Early stopping com conjunto de validação
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    early_stopping_rounds=30,
    verbose=False
)

print(f"Melhor iteração: {xgb_model.best_iteration}")
print(f"Acurácia: {xgb_model.score(X_te, y_te):.4f}")
```

---

## 2. LightGBM — Light Gradient Boosting Machine

**Inovações:** Histogram-based splitting, leaf-wise growth (vs. level-wise), GOSS e EFB. **Muito mais rápido** que XGBoost em datasets grandes.

```python
import lightgbm as lgb

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=42
)

lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_te, y_te)],
    callbacks=[lgb.early_stopping(30), lgb.log_evaluation(period=0)]
)
print(f"Acurácia LightGBM: {lgb_model.score(X_te, y_te):.4f}")
```

---

## 3. CatBoost — Categorical Boosting

**Inovação principal:** trata features categóricas nativamente com **target encoding** robusto e ordered boosting para evitar overfitting.

```python
from catboost import CatBoostClassifier
import pandas as pd

# Exemplo com feature categórica
df = pd.DataFrame({
    'city':    ['SP', 'RJ', 'BH', 'SP', 'RJ', 'BH', 'SP', 'RJ'],
    'age':     [25, 35, 45, 30, 28, 52, 41, 33],
    'income':  [5000, 8000, 6000, 7000, 4500, 9000, 6500, 7500],
    'churn':   [0, 0, 1, 0, 1, 0, 0, 1]
})
X_cat = df.drop('churn', axis=1)
y_cat = df['churn']

cat_features = ['city']  # índice ou nome
cb = CatBoostClassifier(
    iterations=200, learning_rate=0.1, depth=4,
    cat_features=cat_features, verbose=0, random_seed=42
)
cb.fit(X_cat, y_cat)
print(f"CatBoost preds: {cb.predict(X_cat)}")
```

---

## 4. Comparação

| Característica | XGBoost | LightGBM | CatBoost |
|---------------|---------|---------|---------|
| Crescimento de árvore | Level-wise | Leaf-wise | Symmetric |
| Velocidade | Média | Alta | Média-Alta |
| Categóricas | Manual | Básico | Nativo |
| Memória | Média | Baixa | Média |
| Overfitting | Moderado | Maior risco | Baixo |

---

## Questões para Reflexão
1. Por que o crescimento leaf-wise do LightGBM pode causar mais overfitting?
2. Quando usar CatBoost em vez de encodar manualmente as categóricas?
3. Como early stopping interage com o número de estimadores?

## Referências
- Géron, cap. 7
- Documentação oficial: xgboost.readthedocs.io, lightgbm.readthedocs.io, catboost.ai

---
*Próxima aula → [Aula 31: Stacking e Blending](aula-31-stacking-blending.md)*
