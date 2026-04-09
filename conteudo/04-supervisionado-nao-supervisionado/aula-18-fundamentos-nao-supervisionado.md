# Aula 18 — Fundamentos do Aprendizado Não Supervisionado

> **Módulo 04 · Aprendizado Supervisionado e Não Supervisionado** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender o paradigma não supervisionado e seus principais casos de uso
- Distinguir clustering, redução de dimensionalidade e detecção de anomalias
- Identificar métricas de avaliação internas para clustering

---

## 1. O Que É Aprendizado Não Supervisionado?

No aprendizado supervisionado temos pares $(\mathbf{x}, y)$. No **não supervisionado** temos apenas $\{\mathbf{x}^{(i)}\}$ — sem rótulos.

O algoritmo deve descobrir **estrutura latente** nos dados:
- **Clustering**: agrupar dados similares
- **Redução de dimensionalidade**: compactar informação
- **Estimação de densidade**: modelar a distribuição $P(\mathbf{x})$
- **Detecção de anomalias**: identificar pontos atípicos

---

## 2. Casos de Uso Reais

| Aplicação | Técnica | Exemplo |
|-----------|---------|---------|
| Segmentação de clientes | Clustering | K-Means em dados de CRM |
| Compressão de imagem | PCA | Redução de cores |
| Detecção de fraude | Anomaly detection | Isolation Forest |
| Visualização | t-SNE / UMAP | Embeddings de palavras em 2D |
| Pré-processamento | PCA | Reduzir features redundantes |

---

## 3. Métricas de Avaliação (sem rótulo)

### 3.1 Inertia (Within-Cluster Sum of Squares)
$$\text{WCSS} = \sum_{k=1}^{K}\sum_{\mathbf{x} \in C_k} \|\mathbf{x} - \boldsymbol{\mu}_k\|^2$$
Menor é melhor (mas sempre decresce com mais clusters → use Elbow Method).

### 3.2 Silhouette Score
Para cada ponto $i$:
$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
- $a(i)$: distância média intra-cluster
- $b(i)$: distância média ao cluster vizinho mais próximo
- Valor entre -1 e 1; quanto maior, melhor

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np

X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    sil = silhouette_score(X, labels)
    db  = davies_bouldin_score(X, labels)
    print(f"K={k}: Silhouette={sil:.3f}, Davies-Bouldin={db:.3f}")
```

### 3.3 Davies-Bouldin Index
Razão entre dispersão intra-cluster e separação inter-cluster. **Menor é melhor**.

---

## 4. Quando Usar Cada Paradigma?

```
Tenho rótulos (y)?
│
├── SIM ──→ Aprendizado SUPERVISIONADO
│           (Classificação, Regressão)
│
└── NÃO ──→ Aprendizado NÃO SUPERVISIONADO
            │
            ├── Quero agrupar dados similares?
            │   └── Clustering (K-Means, DBSCAN...)
            │
            ├── Quero reduzir dimensões?
            │   └── PCA, t-SNE, UMAP, Autoencoders
            │
            └── Quero detectar anomalias?
                └── Isolation Forest, LOF, One-Class SVM
```

---

## Questões para Reflexão
1. É possível usar aprendizado não supervisionado para melhorar um modelo supervisionado? Como?
2. Por que o Silhouette Score é mais informativo que a inércia para comparar clusters?
3. Em que situação a detecção de anomalias por clustering pode falhar?

## Referências
- Géron, cap. 9 (Técnicas de Aprendizado Não Supervisionado)
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 19: Algoritmos de Clustering](aula-19-clustering.md)*
