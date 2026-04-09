# Aula 33 — MLP e Arquiteturas de Redes Neurais

> **Módulo 07 · Redes Neurais Artificiais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender a arquitetura MLP (Multi-Layer Perceptron)
- Entender o papel das camadas ocultas e funções de ativação
- Aplicar o Teorema da Aproximação Universal

---

## 1. MLP — Multi-Layer Perceptron

```
Camada de     Camadas        Camada de
  Entrada      Ocultas         Saída
  
   x₁ ──┐
   x₂ ──┼─→ [h₁¹ h₂¹ h₃¹] ─→ [h₁² h₂²] ─→ ŷ
   x₃ ──┘
```

**Forward pass:**
$$\mathbf{h}^{(l)} = f^{(l)}\left(\mathbf{W}^{(l)}\mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}\right)$$

---

## 2. Funções de Ativação Não-Lineares

Sem ativação não-linear, camadas empilhadas equivalem a uma única camada linear.

| Ativação | Fórmula | Uso típico |
|----------|---------|-----------|
| Sigmoid | $\frac{1}{1+e^{-z}}$ | Saída binária |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | Camadas ocultas (old) |
| ReLU | $\max(0, z)$ | Camadas ocultas (padrão) |
| Leaky ReLU | $\max(0.01z, z)$ | Evita "neurônio morto" |
| GELU | $z\Phi(z)$ | Transformers |
| Softmax | $\frac{e^{z_k}}{\sum e^{z_j}}$ | Saída multiclasse |

---

## 3. Teorema da Aproximação Universal

> Uma MLP com **uma única camada oculta** e suficientes neurônios pode aproximar qualquer função contínua em $\mathbb{R}^n$.

Na prática: redes **mais profundas** (deep) com menos neurônios aprendem representações mais eficientes.

---

## 4. Capacidade e Profundidade

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

architectures = {
    'Rasa (100)': (100,),
    'Média (50,50)': (50, 50),
    'Profunda (32,32,32)': (32, 32, 32),
    'Larga (500,)': (500,)
}

for name, hidden in architectures.items():
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation='relu',
                        max_iter=500, random_state=42)
    mlp.fit(X_tr_s, y_tr)
    print(f"{name:25s}: treino={mlp.score(X_tr_s, y_tr):.4f} | teste={mlp.score(X_te_s, y_te):.4f}")
```

---

## 5. Regras Práticas para Arquitetura

1. **Camadas:** Começar com 2–3 camadas ocultas
2. **Neurônios:** Pirâmide decrescente (ex: 256 → 128 → 64) ou constante
3. **Ativação oculta:** ReLU (padrão), Leaky ReLU se muitos neurônios mortos
4. **Ativação saída:**
   - Regressão: Linear (sem ativação)
   - Binária: Sigmoid
   - Multiclasse: Softmax
5. **Batch normalization** depois de cada camada densa (avançado)

---

## Questões para Reflexão
1. Por que redes mais profundas generalizam melhor que redes largas e rasas?
2. O que é o "problema do neurônio morto" no ReLU e como evitá-lo?
3. Em qual cenário você usaria Tanh em vez de ReLU?

## Referências
- Géron, cap. 10
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 34: Backpropagation](aula-34-backpropagation.md)*
