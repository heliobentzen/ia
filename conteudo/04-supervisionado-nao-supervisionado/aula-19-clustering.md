# Aula 19 — Algoritmos de Clustering

> **Módulo 04 · Aprendizado Supervisionado e Não Supervisionado** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Implementar e comparar K-Means, DBSCAN e clustering hierárquico
- Escolher o número de clusters com Elbow Method e Silhouette
- Identificar quando cada algoritmo é mais adequado

---

## 1. K-Means

**Algoritmo:**
1. Inicializar K centroides aleatoriamente (ou com K-Means++)
2. Atribuir cada ponto ao centroide mais próximo
3. Recalcular centroides como média do cluster
4. Repetir 2–3 até convergir

$$\min_{\{C_k\}} \sum_{k=1}^{K}\sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

X, y_true = make_blobs(n_samples=400, centers=4, cluster_std=0.9, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
inertias = []
K_range = range(1, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, marker='o')
plt.xlabel('K'); plt.ylabel('Inércia')
plt.title('Elbow Method'); plt.show()

# Modelo final
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            marker='X', s=200, c='red', zorder=5)
plt.title('K-Means (K=4)'); plt.show()
```

**Limitações:** clusters esféricos e de tamanho similar; sensível a outliers; requer K pré-definido.

---

## 2. DBSCAN — Density-Based Spatial Clustering

Encontra clusters de **forma arbitrária** e detecta outliers automaticamente.

**Parâmetros:**
- `eps` (ε): raio de vizinhança
- `min_samples`: mínimo de pontos para ser core point

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

db = DBSCAN(eps=0.2, min_samples=5)
labels = db.fit_predict(X_moons)

# -1 = ruído (outlier)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = (labels == -1).sum()
print(f"Clusters encontrados: {n_clusters}, Ruído: {n_noise}")

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels, cmap='tab10')
plt.title(f'DBSCAN: {n_clusters} clusters, {n_noise} outliers')
plt.show()
```

---

## 3. Clustering Hierárquico (Agglomerativo)

Constrói uma **hierarquia** de clusters (dendrograma). Não requer K pré-definido.

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Dendrograma (para escolher o número de clusters)
linkage_matrix = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 4))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title('Dendrograma (Ward linkage)')
plt.xlabel('Amostras'); plt.ylabel('Distância')
plt.show()

# Modelo com K=4
hc = AgglomerativeClustering(n_clusters=4, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
```

---

## 4. Comparação dos Algoritmos

| Característica | K-Means | DBSCAN | Hierárquico |
|---------------|---------|--------|------------|
| Forma dos clusters | Esférica | Arbitrária | Qualquer |
| Requer K | Sim | Não | Opcional |
| Detecta outliers | Não | Sim | Não |
| Escalabilidade | Alta | Média | Baixa |
| Sensível a escala | Sim | Sim | Sim |

---

## Questões para Reflexão
1. Quando DBSCAN é preferível ao K-Means?
2. Como você escolheria os parâmetros `eps` e `min_samples` do DBSCAN?
3. O clustering hierárquico pode ser usado para datasets com milhões de exemplos?

## Referências
- Géron, cap. 9
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 20: Redução de Dimensionalidade](aula-20-reducao-dimensionalidade.md)*
