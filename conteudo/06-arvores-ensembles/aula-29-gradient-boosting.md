# Aula 29 — Gradient Boosting

> **Módulo 06 · Métodos Baseados em Árvores e Ensembles** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender boosting como minimização de função de perda por gradiente
- Implementar AdaBoost e Gradient Boosting com Scikit-learn
- Controlar overfitting com shrinkage e subsampling

---

## 1. Intuição: Aprender com os Erros

Enquanto o Random Forest treina modelos **em paralelo** e independentes, o Boosting treina modelos **sequencialmente**, onde cada novo modelo tenta corrigir os erros do anterior.

---

## 2. AdaBoost

Ajusta os pesos das amostras: amostras classificadas errado recebem **maior peso**.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=2),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_tr, y_tr)
print(f"AdaBoost — Acurácia: {ada.score(X_te, y_te):.4f}")
```

---

## 3. Gradient Boosting — Minimização por Gradiente

A cada etapa, treina uma nova árvore nos **resíduos pseudo** da iteração anterior:

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$

Onde $h_m$ é a árvore ajustada nos resíduos negativos do gradiente de $J$.

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import numpy as np

gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,   # shrinkage: menor = mais conservador
    max_depth=4,
    subsample=0.8,        # stochastic GB: reduz variância
    min_samples_leaf=5,
    random_state=42
)
gb.fit(X_tr, y_tr)
print(f"GBM — Acurácia: {gb.score(X_te, y_te):.4f}")

# Curva de treino ao longo das iterações
train_err = [1 - acc for acc in gb.train_score_]
test_preds_staged = list(gb.staged_predict(X_te))
test_err = [np.mean(yp != y_te) for yp in test_preds_staged]

import matplotlib.pyplot as plt
plt.plot(train_err, label='Treino')
plt.plot(test_err, label='Teste')
plt.xlabel('Iteração'); plt.ylabel('Erro')
plt.title('Curvas de Erro — Gradient Boosting')
plt.legend()
plt.show()
```

---

## 4. Hiperparâmetros Chave

| Parâmetro | Efeito | Valor típico |
|-----------|--------|-------------|
| `n_estimators` | Nº de árvores | 100–1000 |
| `learning_rate` | Shrinkage | 0.01–0.1 |
| `max_depth` | Complexidade de cada árvore | 3–6 |
| `subsample` | Fração de amostras por árvore | 0.5–0.9 |
| `min_samples_leaf` | Poda mínima | 5–30 |

> **Regra prática:** menor `learning_rate` → mais `n_estimators` → melhor generalização.

---

## Questões para Reflexão
1. Por que usar `learning_rate` pequeno com muitos estimadores tende a ser melhor?
2. Qual é a relação entre `subsample < 1` e a aleatoriedade do Random Forest?
3. Como detectar o número ótimo de árvores sem overfitting?

## Referências
- Géron, cap. 7
- Faceli et al., cap. 6

---
*Próxima aula → [Aula 30: XGBoost, LightGBM e CatBoost](aula-30-xgboost-lightgbm-catboost.md)*
