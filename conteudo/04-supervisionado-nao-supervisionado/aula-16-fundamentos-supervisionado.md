# Aula 16 — Fundamentos do Aprendizado Supervisionado

> **Módulo 04 · Aprendizado Supervisionado e Não Supervisionado** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Formalizar matematicamente o problema de aprendizado supervisionado
- Compreender os conceitos de função de perda, espaço de hipóteses e ERM
- Entender o dilema bias-variance e a maldição da dimensionalidade

---

## 1. Formalização do Aprendizado Supervisionado

Dado um conjunto de treinamento $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{m}$, onde:
- $\mathbf{x}^{(i)} \in \mathcal{X}$ é o vetor de features (entrada)
- $y^{(i)} \in \mathcal{Y}$ é o rótulo (saída desejada)

O objetivo é aprender uma função $h: \mathcal{X} \to \mathcal{Y}$ que **generalize** bem para exemplos não vistos.

**Tipos de saída:**
| Tipo | $\mathcal{Y}$ | Exemplos |
|------|--------------|----------|
| Regressão | $\mathbb{R}$ | Preço de imóvel, temperatura |
| Classificação binária | $\{0, 1\}$ | Spam/não-spam, doença/sadio |
| Classificação multiclasse | $\{1, 2, ..., K\}$ | Espécie de flor, dígito (0–9) |

---

## 2. Função de Perda e Risco

A **função de perda** $\ell(h(\mathbf{x}), y)$ mede o erro da hipótese $h$ numa amostra.

**Risco esperado (generalization error):**
$$R(h) = \mathbb{E}_{(\mathbf{x},y) \sim P}[\ell(h(\mathbf{x}), y)]$$

Como $P$ é desconhecida, minimizamos o **risco empírico**:
$$\hat{R}(h) = \frac{1}{m} \sum_{i=1}^{m} \ell(h(\mathbf{x}^{(i)}), y^{(i)})$$

Isso é o princípio **ERM (Empirical Risk Minimization)**.

---

## 3. Dilema Bias-Variance

O erro de generalização pode ser decomposto em:
$$\text{Erro} = \text{Bias}^2 + \text{Variância} + \text{Ruído irredutível}$$

- **Bias alto** → modelo muito simples, não captura padrões (underfitting)
- **Variância alta** → modelo muito complexo, memoriza ruído (overfitting)

```
Complexidade do Modelo
     ←── simples                    complexo ──→
Bias    [██████████████░░░░░░░░░░░░░░░░░░░░░░░]
Variance[░░░░░░░░░░░░░░████████████████████████]
Erro    [████████████░░░░░░░██████████████████░]
                   ↑ ponto ótimo
```

---

## 4. Escolhendo a Hipótese Certa

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, 100)

graus = [1, 3, 9, 15]
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for ax, grau in zip(axes, graus):
    pipe = Pipeline([
        ('poly', PolynomialFeatures(grau)),
        ('model', LinearRegression())
    ])
    pipe.fit(X, y)
    cv_score = cross_val_score(pipe, X, y, cv=5, scoring='neg_mean_squared_error').mean()
    
    X_plot = np.linspace(0, 1, 200).reshape(-1, 1)
    ax.scatter(X, y, alpha=0.5, s=10)
    ax.plot(X_plot, pipe.predict(X_plot), color='red', linewidth=2)
    ax.set_title(f'Grau {grau}\nCV MSE: {-cv_score:.4f}')
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
```

---

## 5. Split Treino/Validação/Teste

```python
from sklearn.model_selection import train_test_split

# Divisão estratificada (mantém proporção de classes)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
)

print(f"Treino:    {len(X_train)} amostras ({len(X_train)/len(X)*100:.0f}%)")
print(f"Validação: {len(X_val)}  amostras ({len(X_val)/len(X)*100:.0f}%)")
print(f"Teste:     {len(X_test)}  amostras ({len(X_test)/len(X)*100:.0f}%)")
```

> 📐 Regra prática: 60%/20%/20% ou 70%/15%/15% dependendo do tamanho do dataset.

---

## Questões para Reflexão
1. Por que minimizar o risco empírico não garante boa generalização?
2. Como o tamanho do dataset afeta o dilema bias-variance?
3. Um modelo com bias alto e variância baixa está em overfitting ou underfitting?

## Referências
- Faceli et al., cap. 4 (Fundamentos do Aprendizado Supervisionado)
- Russell & Norvig, cap. 19

---
*Próxima aula → [Aula 17: K-Vizinhos Mais Próximos (KNN)](aula-17-knn.md)*
