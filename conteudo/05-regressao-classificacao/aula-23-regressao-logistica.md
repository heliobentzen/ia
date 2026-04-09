# Aula 23 — Regressão Logística

> **Módulo 05 · Regressão e Classificação** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender a função sigmoide e a interpretação probabilística
- Implementar regressão logística binária e multiclasse
- Interpretar coeficientes e probabilidades preditas

---

## 1. Da Regressão para Classificação

Na regressão linear, $\hat{y}$ pode ser qualquer valor real. Para classificação, queremos $P(y=1|\mathbf{x}) \in [0, 1]$.

**Função Sigmoide:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = \boldsymbol{\theta}^T\mathbf{x}$$

$$P(y=1|\mathbf{x}) = \sigma(\boldsymbol{\theta}^T\mathbf{x})$$

---

## 2. Função de Custo — Log Loss (Binary Cross-Entropy)

$$J(\boldsymbol{\theta}) = -\frac{1}{m}\sum_{i=1}^{m}\left[y^{(i)}\log(\hat{p}^{(i)}) + (1-y^{(i)})\log(1-\hat{p}^{(i)})\right]$$

Minimizada por gradiente descendente. Convexa → um mínimo global.

---

## 3. Implementação

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.datasets import load_breast_cancer
import seaborn as sns

# Dataset: câncer de mama (binário)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Modelo
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

# Probabilidades
y_prob = lr.predict_proba(X_test_s)[:, 1]
y_pred = lr.predict(X_test_s)

print(classification_report(y_test, y_pred, target_names=cancer.target_names))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=cancer.target_names, yticklabels=cancer.target_names)
plt.title('Matriz de Confusão')
plt.show()

# Interpretação dos coeficientes (odds ratio)
for name, coef in sorted(zip(cancer.feature_names, lr.coef_[0]),
                          key=lambda x: abs(x[1]), reverse=True)[:10]:
    odds_ratio = np.exp(coef)
    print(f"{name:35s}: coef={coef:+.3f}, OR={odds_ratio:.3f}")
```

---

## 4. Multiclasse

**OvR (One-vs-Rest):** treina K classificadores binários.
**Softmax (Multinomial):** generaliza diretamente para K classes.

```python
from sklearn.datasets import load_iris

iris = load_iris()
X_i, y_i = iris.data, iris.target
X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(X_i, y_i, test_size=0.2, random_state=42)

lr_multi = LogisticRegression(multi_class='multinomial', max_iter=1000)
lr_multi.fit(X_tr_i, y_tr_i)
print(f"Acurácia multiclasse: {lr_multi.score(X_te_i, y_te_i):.4f}")
```

---

## Questões para Reflexão
1. Por que não usar MSE como função de custo na regressão logística?
2. O parâmetro `C` na sklearn corresponde a qual penalidade de regularização?
3. O que acontece com as probabilidades preditas quando o modelo está mal calibrado?

## Referências
- Géron, cap. 4
- Faceli et al., cap. 5

---
*Próxima aula → [Aula 24: Support Vector Machines](aula-24-svm.md)*
