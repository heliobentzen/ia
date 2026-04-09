# Aula 32 — Neurônio Artificial e Perceptron

> **Módulo 07 · Redes Neurais Artificiais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender a analogia entre neurônio biológico e artificial
- Implementar o Perceptron de Rosenblatt do zero
- Entender os limites do Perceptron (problema XOR)

---

## 1. Neurônio Biológico vs. Artificial

| Biológico | Artificial |
|-----------|-----------|
| Dendritos | Entradas $x_j$ |
| Sinapses | Pesos $w_j$ |
| Corpo celular | Soma ponderada + bias |
| Axônio | Saída $\hat{y}$ |
| Potencial de ação | Função de ativação |

**Modelo matemático:**
$$\hat{y} = f\left(\sum_{j=0}^{n} w_j x_j\right) = f(\mathbf{w}^T\mathbf{x})$$

---

## 2. Perceptron de Rosenblatt (1957)

Função de ativação: degrau de Heaviside.
$$f(z) = \begin{cases} 1 & \text{se } z \geq 0 \\ 0 & \text{caso contrário} \end{cases}$$

**Regra de aprendizado:**
$$w_j \leftarrow w_j + \eta (y^{(i)} - \hat{y}^{(i)}) x_j^{(i)}$$

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = 0.
        self.errors_ = []
        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X, y):
                update = self.eta * (yi - self.predict_single(xi))
                self.w_ += update * xi
                self.b_  += update
                errors   += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return X @ self.w_ + self.b_

    def predict_single(self, xi):
        return 1 if self.net_input(xi) >= 0 else 0

    def predict(self, X):
        return np.where(self.net_input(X) >= 0, 1, 0)

# Teste: AND gate (linearmente separável)
X_and = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = np.array([0, 0, 0, 1])

ppn = Perceptron(eta=0.1, n_iter=20)
ppn.fit(X_and, y_and)
print("Predições AND:", ppn.predict(X_and))
print("Esperado:     ", y_and)

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Época'); plt.ylabel('Erros')
plt.title('Convergência do Perceptron (AND)')
plt.show()
```

---

## 3. Limitação: O Problema XOR

O XOR **não é linearmente separável** — um único Perceptron não consegue aprender.

```python
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([0, 1, 1, 0])

ppn_xor = Perceptron(eta=0.1, n_iter=100)
ppn_xor.fit(X_xor, y_xor)
print("Predições XOR:", ppn_xor.predict(X_xor))
print("Esperado:     ", y_xor)
# Não converge!
```

**Solução:** múltiplas camadas (MLP) — próxima aula.

---

## Questões para Reflexão
1. Por que o Perceptron converge apenas se os dados forem linearmente separáveis?
2. Qual é a diferença entre o Perceptron e a regressão logística?
3. Como o bias $b$ afeta o hiperplano de separação?

## Referências
- Géron, cap. 10
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 33: MLP e Arquiteturas](aula-33-mlp-arquiteturas.md)*
