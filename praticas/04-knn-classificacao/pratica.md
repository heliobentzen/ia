# Prática 04 — KNN — Classificação

**Módulo:** 04 | **Duração:** ~60 minutos | **Dataset:** Iris + Wine

## Objetivos
- Implementar KNN com Scikit-learn para classificação multiclasse
- Escolher K via validação cruzada
- Visualizar fronteiras de decisão

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

# Iris dataset
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Buscar melhor K
k_range = range(1, 31)
cv_scores = []
for k in k_range:
    pipe = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=k))])
    score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy').mean()
    cv_scores.append(score)

best_k = k_range[np.argmax(cv_scores)]
print(f"Melhor K: {best_k} (acurácia CV: {max(cv_scores):.4f})")

plt.plot(k_range, cv_scores, marker='o')
plt.axvline(best_k, color='red', linestyle='--', label=f'K={best_k}')
plt.xlabel('K'); plt.ylabel('Acurácia (CV)'); plt.title('Escolha de K')
plt.legend(); plt.show()

# Modelo final
best_pipe = Pipeline([
    ('sc', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=best_k, weights='distance'))
])
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualização 2D via PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(StandardScaler().fit_transform(X))
knn_2d = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
knn_2d.fit(X_2d[y_train.index if hasattr(y_train, 'index') else np.arange(len(X_train))],
           y_train)
# (plot simplificado)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.title(f'KNN K={best_k} — Iris em 2D (PCA)'); plt.show()
```

## Desafios Extras
1. Compare KNN com diferentes métricas: Euclidiana, Manhattan, Chebyshev
2. Aplique KNN ao dataset `Wine` e compare com Iris
3. Implemente KNN para **regressão** no dataset California Housing e calcule o RMSE
