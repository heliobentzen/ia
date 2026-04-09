# Aula 34 — Backpropagation

> **Módulo 07 · Redes Neurais Artificiais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender a regra da cadeia aplicada a redes neurais
- Derivar o algoritmo de backpropagation para uma MLP simples
- Entender o papel dos gradientes no treinamento

---

## 1. O Problema

Queremos calcular $\frac{\partial J}{\partial w_{jk}^{(l)}}$ para **todos** os pesos da rede. O backpropagation faz isso eficientemente usando a regra da cadeia.

---

## 2. Forward Pass

Para cada camada $l$:
$$\mathbf{z}^{(l)} = \mathbf{W}^{(l)}\mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}$$
$$\mathbf{a}^{(l)} = f^{(l)}(\mathbf{z}^{(l)})$$

---

## 3. Backward Pass — Regra da Cadeia

**Erro na camada de saída (L):**
$$\boldsymbol{\delta}^{(L)} = \nabla_{\mathbf{a}}J \odot f'^{(L)}(\mathbf{z}^{(L)})$$

**Propagação do erro para camadas anteriores:**
$$\boldsymbol{\delta}^{(l)} = \left(\mathbf{W}^{(l+1)T}\boldsymbol{\delta}^{(l+1)}\right) \odot f'^{(l)}(\mathbf{z}^{(l)})$$

**Gradientes dos pesos:**
$$\frac{\partial J}{\partial \mathbf{W}^{(l)}} = \boldsymbol{\delta}^{(l)}\mathbf{a}^{(l-1)T}$$

---

## 4. Implementação do Zero (1 camada oculta)

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

class SimpleNN:
    def __init__(self, n_input, n_hidden, n_output, lr=0.01):
        self.lr = lr
        # Inicialização Xavier
        self.W1 = np.random.randn(n_input,  n_hidden) * np.sqrt(2/n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2/n_hidden)
        self.b2 = np.zeros((1, n_output))

    def forward(self, X):
        self.z1 = X  @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, y_hat):
        m = X.shape[0]
        # Gradiente da saída
        delta2 = (y_hat - y) * sigmoid_derivative(y_hat)
        dW2 = self.a1.T @ delta2 / m
        db2 = delta2.mean(axis=0, keepdims=True)
        # Propagação
        delta1 = (delta2 @ self.W2.T) * sigmoid_derivative(self.a1)
        dW1 = X.T @ delta1 / m
        db1 = delta1.mean(axis=0, keepdims=True)
        # Atualização
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            y_hat = self.forward(X)
            loss  = -np.mean(y * np.log(y_hat + 1e-8) + (1-y) * np.log(1 - y_hat + 1e-8))
            losses.append(loss)
            self.backward(X, y, y_hat)
            if epoch % 100 == 0:
                acc = ((y_hat > 0.5).astype(int) == y).mean()
                print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.4f}")
        return losses

# XOR
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]])
y_xor = np.array([[0],[1],[1],[0]])

nn = SimpleNN(2, 4, 1, lr=0.5)
losses = nn.train(X_xor, y_xor, epochs=5000)
preds = (nn.forward(X_xor) > 0.5).astype(int)
print("XOR predições:", preds.ravel())
print("Esperado:     ", y_xor.ravel())
```

---

## 5. Problemas de Gradiente

| Problema | Causa | Solução |
|---------|-------|---------|
| Gradientes explodindo | Pesos muito grandes | Gradient clipping, BN |
| Gradientes sumindo | Ativações saturadas | ReLU, LSTM, ResNet |
| Convergência lenta | LR inadequado | Adam, LR scheduling |

---

## Questões para Reflexão
1. Por que o Sigmoid causa o problema de gradientes que somem em redes profundas?
2. O que é a inicialização de Xavier/He e por que ela importa?
3. Como o mini-batch SGD difere do SGD estocástico puro?

## Referências
- Géron, cap. 11
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 35: Otimizadores e Funções de Ativação](aula-35-otimizadores-funcoes-ativacao.md)*
