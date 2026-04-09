# Aula 20 — Redução de Dimensionalidade

> **Módulo 04 · Aprendizado Supervisionado e Não Supervisionado** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Aplicar PCA para compressão e visualização de dados
- Usar t-SNE e UMAP para visualização de alta dimensão em 2D/3D
- Entender a diferença entre métodos lineares e não-lineares

---

## 1. PCA — Análise de Componentes Principais

PCA encontra os **eixos de maior variância** nos dados (componentes principais) e projeta os dados neles.

**Matematicamente:** decomposição SVD da matriz de covariância.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target  # 1797 x 64

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Quantas componentes para 95% da variância?
pca_full = PCA()
pca_full.fit(X_scaled)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_comp_95 = np.searchsorted(cumvar, 0.95) + 1
print(f"Componentes para 95% variância: {n_comp_95} (de 64 features)")

plt.plot(cumvar)
plt.axhline(0.95, color='red', linestyle='--')
plt.xlabel('Número de Componentes')
plt.ylabel('Variância Acumulada')
plt.title('PCA — Variância Explicada')
plt.show()

# Reduzir para 2D e visualizar
pca2d = PCA(n_components=2)
X_2d = pca2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', alpha=0.7, s=15)
plt.colorbar(scatter)
plt.title('Dígitos em 2D (PCA)')
plt.show()
```

---

## 2. t-SNE — Visualização Não-Linear

t-SNE preserva **vizinhanças locais** e é excelente para visualização (não para compressão ou predição).

```python
from sklearn.manifold import TSNE

# Reduzir primeiro com PCA para acelerar t-SNE
pca50 = PCA(n_components=50)
X_pca50 = pca50.fit_transform(X_scaled)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
X_tsne = tsne.fit_transform(X_pca50)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7, s=15)
plt.colorbar(scatter)
plt.title('Dígitos em 2D (t-SNE)')
plt.show()
```

> ⚠️ As escalas dos eixos do t-SNE **não têm significado** — apenas a vizinhança importa.

---

## 3. UMAP — Uniform Manifold Approximation and Projection

Mais rápido que t-SNE, preserva estrutura local **e** global. Pode ser usado para redução de dimensionalidade em pipelines de ML.

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7, s=15)
plt.colorbar(scatter)
plt.title('Dígitos em 2D (UMAP)')
plt.show()
```

---

## 4. Comparação

| Método | Tipo | Usa para predição? | Preserva | Velocidade |
|--------|------|--------------------|---------|-----------|
| PCA | Linear | ✅ Sim | Variância global | Rápido |
| t-SNE | Não-linear | ❌ Só viz | Vizinhança local | Lento |
| UMAP | Não-linear | ✅ Sim | Local + global | Médio |
| LDA | Linear supervisionado | ✅ Sim | Separabilidade | Rápido |

---

## Questões para Reflexão
1. Por que não é recomendado usar t-SNE em produção para reduzir dimensões?
2. PCA pode ser usado para eliminar ruído? Como?
3. Quando UMAP é preferível ao t-SNE?

## Referências
- Géron, cap. 8
- Faceli et al., cap. 8

---
*Módulo 04 concluído! Próximo → [Módulo 05: Regressão e Classificação](../05-regressao-classificacao/README.md)*
