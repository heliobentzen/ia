# Aula 17 — K-Vizinhos Mais Próximos (KNN)

> **Módulo 04 · Aprendizado Supervisionado e Não Supervisionado** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender o funcionamento do algoritmo KNN
- Aplicar KNN para classificação e regressão com Scikit-learn
- Escolher K e a métrica de distância adequados

---

## 1. Intuição

O KNN é um algoritmo **lazy** (não há treinamento): para classificar um novo ponto, basta encontrar os K exemplos mais próximos no conjunto de treino e fazer uma votação majoritária (ou média, na regressão).

> "Me diga com quem andas e te direi quem és."

---

## 2. Métricas de Distância

**Distância Euclidiana (padrão):**
$$d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{n}(x_j - x'_j)^2}$$

**Distância Manhattan:**
$$d(\mathbf{x}, \mathbf{x}') = \sum_{j=1}^{n}|x_j - x'_j|$$

**Distância de Minkowski (generalização):**
$$d(\mathbf{x}, \mathbf{x}') = \left(\sum_{j=1}^{n}|x_j - x'_j|^p\right)^{1/p}$$
- $p=1$: Manhattan · $p=2$: Euclidiana

---

## 3. Implementação com Scikit-learn

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# Dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline: escalar é ESSENCIAL para KNN
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=5))
])

pipe.fit(X_train, y_train)
print(f"Acurácia no teste: {pipe.score(X_test, y_test):.4f}")

# Escolhendo K por validação cruzada
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    pipe_k = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
    score = cross_val_score(pipe_k, X_train, y_train, cv=5, scoring='accuracy').mean()
    cv_scores.append(score)

best_k = k_values[np.argmax(cv_scores)]
print(f"Melhor K: {best_k} (acurácia CV: {max(cv_scores):.4f})")

plt.plot(k_values, cv_scores, marker='o')
plt.xlabel('K')
plt.ylabel('Acurácia (CV)')
plt.title('Escolha de K no KNN')
plt.axvline(best_k, color='red', linestyle='--', label=f'K={best_k}')
plt.legend()
plt.show()
```

---

## 4. KNN para Regressão

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()
X_h, y_h = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

pipe_reg = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsRegressor(n_neighbors=10, weights='distance'))
])
pipe_reg.fit(X_tr, y_tr)
y_pred = pipe_reg.predict(X_te)
rmse = np.sqrt(mean_squared_error(y_te, y_pred))
print(f"RMSE: {rmse:.4f}")
```

---

## 5. Pontos Fortes e Fracos

| ✅ Vantagens | ❌ Desvantagens |
|-------------|----------------|
| Simples e intuitivo | Lento na predição (O(n·d)) |
| Sem suposição de distribuição | Sensível à escala das features |
| Naturalmente multiclasse | Sofre com alta dimensionalidade |
| Adaptável (KD-Tree, Ball-Tree) | Armazena todo o conjunto de treino |

---

## Questões para Reflexão
1. Por que escalar as features é essencial para o KNN?
2. K=1 causa overfitting ou underfitting? Por quê?
3. Como o KNN se comporta quando há classes desbalanceadas?

## Referências
- Géron, cap. 5
- Faceli et al., cap. 5

---
*Próxima aula → [Aula 18: Fundamentos do Aprendizado Não Supervisionado](aula-18-fundamentos-nao-supervisionado.md)*
