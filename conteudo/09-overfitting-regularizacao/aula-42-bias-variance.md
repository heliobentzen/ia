# Aula 42 — Dilema Bias-Variance

> **Módulo 09 · Overfitting e Regularização** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Decompor matematicamente o erro de generalização
- Visualizar o tradeoff bias-variance com experimentos práticos
- Identificar overfitting e underfitting em situações reais

---

## 1. Decomposição do Erro

$$\mathbb{E}[(\hat{y} - y)^2] = \underbrace{\text{Bias}(\hat{f})^2}_{\text{underfitting}} + \underbrace{\text{Var}(\hat{f})}_{\text{overfitting}} + \underbrace{\sigma^2}_{\text{ruído irredutível}}$$

- **Bias**: quanto o modelo sistematicamente erra (erro na média)
- **Variância**: quanto a predição muda entre diferentes conjuntos de treino
- **Ruído**: componente irreducível do problema

---

## 2. Experimento Prático

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Função verdadeira: y = sin(x) + ruído
np.random.seed(42)
def true_function(x):
    return np.sin(x)

# Gerar múltiplos datasets e treinar modelos
n_datasets = 100
n_train = 30
x_test = np.linspace(0, 2*np.pi, 100)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, degree, title in zip(axes, [1, 4, 20], ['Grau 1 (Underfitting)', 'Grau 4 (Adequado)', 'Grau 20 (Overfitting)']):
    all_preds = []
    for _ in range(n_datasets):
        x_train = np.sort(np.random.uniform(0, 2*np.pi, n_train))
        y_train = true_function(x_train) + np.random.normal(0, 0.3, n_train)
        
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('lr', LinearRegression())
        ])
        pipe.fit(x_train.reshape(-1,1), y_train)
        preds = pipe.predict(x_test.reshape(-1,1))
        all_preds.append(preds)
        ax.plot(x_test, preds, alpha=0.1, color='blue')
    
    all_preds = np.array(all_preds)
    mean_pred = all_preds.mean(axis=0)
    true_vals = true_function(x_test)
    
    bias2 = np.mean((mean_pred - true_vals)**2)
    variance = np.mean(all_preds.var(axis=0))
    
    ax.plot(x_test, true_vals, 'r-', linewidth=2, label='Função real')
    ax.plot(x_test, mean_pred, 'k--', linewidth=2, label='Média predita')
    ax.set_title(f'{title}\nBias²={bias2:.3f}, Var={variance:.3f}')
    ax.legend(); ax.set_ylim(-3, 3)

plt.tight_layout(); plt.show()
```

---

## 3. Estratégias para Reduzir Bias

- Usar modelo mais complexo
- Adicionar features (feature engineering)
- Reduzir regularização
- Usar ensemble

## 4. Estratégias para Reduzir Variância

- Mais dados de treinamento
- Regularização (L1, L2, Dropout)
- Reduzir complexidade do modelo
- Bagging / ensemble
- Early stopping

---

## Questões para Reflexão
1. É possível ter simultaneamente bias zero e variância zero? Por quê?
2. Como o bagging reduz a variância sem aumentar o bias?
3. Por que coletar mais dados é mais eficaz para reduzir variância que bias?

## Referências
- Géron, cap. 4
- Faceli et al., cap. 3

---
*Próxima aula → [Aula 43: Regularização L1 e L2](aula-43-regularizacao-l1-l2.md)*
