# Aula 27 — Árvores de Decisão

> **Módulo 06 · Métodos Baseados em Árvores e Ensembles** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender critérios de divisão: Gini, Entropia e MSE
- Implementar e visualizar árvores de decisão
- Controlar overfitting com poda e restrições de profundidade

---

## 1. Como Funciona

Uma árvore de decisão divide recursivamente o espaço de features por **thresholds** que maximizam a pureza dos nós filhos.

**Critério Gini (classificação):**
$$G = 1 - \sum_{k=1}^{K}p_k^2$$

**Entropia:**
$$H = -\sum_{k=1}^{K}p_k\log_2(p_k)$$

**MSE (regressão):**
Minimiza a variância dentro de cada nó.

---

## 2. Implementação

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Árvore sem restrições (overfitting total)
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
print(f"Treino: {dt_full.score(X_train, y_train):.4f} | Teste: {dt_full.score(X_test, y_test):.4f}")
print(f"Profundidade: {dt_full.get_depth()} | Nós folha: {dt_full.get_n_leaves()}")

# Poda por profundidade máxima
dt_pruned = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42)
dt_pruned.fit(X_train, y_train)
print(f"Treino: {dt_pruned.score(X_train, y_train):.4f} | Teste: {dt_pruned.score(X_test, y_test):.4f}")

# Visualização
fig, ax = plt.subplots(figsize=(16, 8))
plot_tree(dt_pruned, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, ax=ax)
plt.title("Árvore de Decisão (max_depth=4)")
plt.tight_layout()
plt.show()

# Importância de features
importances = dt_pruned.feature_importances_
for name, imp in sorted(zip(iris.feature_names, importances), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")
```

---

## 3. Escolhendo a Profundidade Ótima

```python
train_scores, test_scores = [], []
depths = range(1, 20)

for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42)
    dt.fit(X_train, y_train)
    train_scores.append(dt.score(X_train, y_train))
    test_scores.append(dt.score(X_test, y_test))

plt.plot(depths, train_scores, label='Treino', marker='o')
plt.plot(depths, test_scores,  label='Teste',  marker='s')
plt.xlabel('Profundidade Máxima')
plt.ylabel('Acurácia')
plt.title('Profundidade vs. Performance')
plt.legend()
plt.show()
```

---

## 4. Pontos Fortes e Fracos

| ✅ Vantagens | ❌ Desvantagens |
|-------------|----------------|
| Interpretável | Alta variância (instável) |
| Não requer escala | Overfitting fácil |
| Trata categóricas nativamente | Fronteiras de decisão ortogonais |
| Lida com dados ausentes | Não extrapola bem |

---

## Questões para Reflexão
1. Por que uma árvore sem restrições de profundidade pode memorizar o treino perfeitamente?
2. Qual critério de divisão é mais adequado para classes muito desbalanceadas?
3. Como o `min_samples_leaf` ajuda a prevenir overfitting?

## Referências
- Géron, cap. 6
- Faceli et al., cap. 6

---
*Próxima aula → [Aula 28: Random Forest](aula-28-random-forest.md)*
