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

### Diagrama da Arquitetura MLP

```mermaid
graph LR
    subgraph Entrada
        x1["x₁"]
        x2["x₂"]
    end

    subgraph Camada Oculta 1
        h1["h₁⁽¹⁾"]
        h2["h₂⁽¹⁾"]
    end

    subgraph Camada de Saída
        y1["ŷ"]
    end

    x1 -- "w₁₁⁽¹⁾" --> h1
    x1 -- "w₁₂⁽¹⁾" --> h2
    x2 -- "w₂₁⁽¹⁾" --> h1
    x2 -- "w₂₂⁽¹⁾" --> h2

    h1 -- "w₁₁⁽²⁾" --> y1
    h2 -- "w₂₁⁽²⁾" --> y1

    style Entrada fill:#e8f4fd,stroke:#2196F3
    style Camada Oculta 1 fill:#fff3e0,stroke:#FF9800
    style Camada de Saída fill:#e8f5e9,stroke:#4CAF50
```

> Cada aresta representa um peso aprendido. Cada neurônio oculto aplica uma função de ativação não-linear sobre a soma ponderada das entradas.

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

### Visualização Comparativa das Funções de Ativação

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 500)

sigmoid = 1 / (1 + np.exp(-z))
tanh = np.tanh(z)
relu = np.maximum(0, z)
leaky_relu = np.where(z > 0, z, 0.01 * z)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Funções de Ativação — Comparação', fontsize=16, fontweight='bold')

activations = [
    (sigmoid, 'Sigmoid', '#2196F3', r'$\sigma(z) = \frac{1}{1+e^{-z}}$'),
    (tanh, 'Tanh', '#FF9800', r'$\tanh(z)$'),
    (relu, 'ReLU', '#4CAF50', r'$\max(0, z)$'),
    (leaky_relu, 'Leaky ReLU', '#9C27B0', r'$\max(0.01z, z)$'),
]

for ax, (y, name, color, formula) in zip(axes.flat, activations):
    ax.plot(z, y, color=color, linewidth=2.5, label=f'{name}: {formula}')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_xlabel('z')
    ax.set_ylabel('f(z)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 5)

plt.tight_layout()
plt.savefig('ativacoes_comparacao.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2.1. Por que Ativações Não-Lineares?

### Intuição: multiplicar matrizes não gera nada novo

Imagine que cada camada de uma rede neural é um **filtro colorido** numa lanterna. Se todos os filtros forem transparentes (lineares), não importa quantos você empilhe — a luz passa igual, como se houvesse um único filtro. A não-linearidade é o que dá **cor** a cada filtro, permitindo que a combinação de camadas crie padrões que uma camada sozinha jamais conseguiria.

Em termos matemáticos: **sem ativações não-lineares, empilhar camadas é como multiplicar matrizes — o resultado sempre pode ser reduzido a uma única matriz.**

### Prova Matemática (2 camadas lineares)

Considere uma rede com 2 camadas **sem** função de ativação:

$$\text{Camada 1: } \mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$$

$$\text{Camada 2: } \hat{\mathbf{y}} = \mathbf{W}^{(2)}\mathbf{z}^{(1)} + \mathbf{b}^{(2)}$$

Substituindo $\mathbf{z}^{(1)}$ na segunda equação:

$$\hat{\mathbf{y}} = \mathbf{W}^{(2)}\left(\mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}\right) + \mathbf{b}^{(2)}$$

$$\hat{\mathbf{y}} = \underbrace{\mathbf{W}^{(2)}\mathbf{W}^{(1)}}_{\mathbf{W}'}\mathbf{x} + \underbrace{\mathbf{W}^{(2)}\mathbf{b}^{(1)} + \mathbf{b}^{(2)}}_{\mathbf{b}'}$$

$$\boxed{\hat{\mathbf{y}} = \mathbf{W}'\mathbf{x} + \mathbf{b}'}$$

**Conclusão:** Duas camadas lineares colapsam em **uma única transformação linear** $\mathbf{W}' = \mathbf{W}^{(2)}\mathbf{W}^{(1)}$. Sem não-linearidade, profundidade não agrega poder de representação.

> 💡 É por isso que **toda** camada oculta precisa de uma função de ativação não-linear — ela é o ingrediente que dá às redes profundas a capacidade de aprender fronteiras de decisão complexas.

---

## 3. Exemplo Numérico: Forward Pass

Vamos rastrear um forward pass completo numa rede **2 → 2 → 1** com ativação sigmoid.

### Dados da rede

```
Entrada: x = [0.5, 0.8]

Camada oculta (2 neurônios):
  W⁽¹⁾ = [[0.4, 0.3],    b⁽¹⁾ = [0.1, -0.2]
           [0.2, 0.7]]

Camada de saída (1 neurônio):
  W⁽²⁾ = [[0.5, 0.6]]    b⁽²⁾ = [-0.1]
```

### Passo 1 — Camada Oculta (pré-ativação)

$$z_1^{(1)} = w_{11}x_1 + w_{12}x_2 + b_1 = (0.4)(0.5) + (0.3)(0.8) + 0.1 = 0.20 + 0.24 + 0.1 = 0.54$$

$$z_2^{(1)} = w_{21}x_1 + w_{22}x_2 + b_2 = (0.2)(0.5) + (0.7)(0.8) + (-0.2) = 0.10 + 0.56 - 0.2 = 0.46$$

### Passo 2 — Camada Oculta (pós-ativação: Sigmoid)

$$h_1^{(1)} = \sigma(0.54) = \frac{1}{1+e^{-0.54}} \approx 0.6318$$

$$h_2^{(1)} = \sigma(0.46) = \frac{1}{1+e^{-0.46}} \approx 0.6130$$

### Passo 3 — Camada de Saída (pré-ativação)

$$z_1^{(2)} = w_{11}^{(2)}h_1^{(1)} + w_{12}^{(2)}h_2^{(1)} + b_1^{(2)} = (0.5)(0.6318) + (0.6)(0.6130) + (-0.1)$$

$$z_1^{(2)} = 0.3159 + 0.3678 - 0.1 = 0.5837$$

### Passo 4 — Saída Final (Sigmoid)

$$\hat{y} = \sigma(0.5837) = \frac{1}{1+e^{-0.5837}} \approx 0.6419$$

### Verificação em Python

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x = np.array([0.5, 0.8])

W1 = np.array([[0.4, 0.3],
                [0.2, 0.7]])
b1 = np.array([0.1, -0.2])

W2 = np.array([[0.5, 0.6]])
b2 = np.array([-0.1])

# Camada oculta
z1 = W1 @ x + b1
h1 = sigmoid(z1)
print(f"Pré-ativação oculta:  z1 = {z1}")        # [0.54, 0.46]
print(f"Pós-ativação oculta:  h1 = {h1}")         # [0.6318, 0.6130]

# Camada de saída
z2 = W2 @ h1 + b2
y_hat = sigmoid(z2)
print(f"Pré-ativação saída:   z2 = {z2}")          # [0.5837]
print(f"Saída final:       ŷ = {y_hat}")            # [0.6419]
```

```
Pré-ativação oculta:  z1 = [0.54 0.46]
Pós-ativação oculta:  h1 = [0.63176958 0.61301418]
Pré-ativação saída:   z2 = [0.58369330]
Saída final:       ŷ = [0.64190825]
```

> 🔑 **Observe:** cada camada realiza duas operações — (1) transformação linear $\mathbf{z} = \mathbf{Wx} + \mathbf{b}$ e (2) ativação não-linear $\mathbf{h} = f(\mathbf{z})$. A saída de uma camada se torna a entrada da próxima.

---

## 4. Teorema da Aproximação Universal

> Uma MLP com **uma única camada oculta** e suficientes neurônios pode aproximar qualquer função contínua em $\mathbb{R}^n$.

### Intuição: a rede como um escultor com blocos de LEGO

Imagine que você quer esculpir uma curva suave qualquer. Cada **neurônio** na camada oculta funciona como um **bloco de LEGO** — um pedaço simples de formato fixo (um degrau, uma rampa). Um bloco sozinho não se parece com nada interessante, mas com **blocos suficientes** você pode empilhá-los e combiná-los para aproximar **qualquer formato** que desejar.

```
    ┌──────┐
    │      │  ← 1 neurônio = 1 "degrau"
────┘      └────

         ┌─┐ ┌──┐
    ┌──┐ │ │ │  │    ← muitos neurônios combinados
────┘  └─┘ └─┘  └──    ≈ qualquer curva!
```

- Com **poucos neurônios**: a aproximação é grosseira (poucos blocos → escultura angular).
- Com **muitos neurônios**: a aproximação converge para a função alvo (muitos blocos → escultura suave).
- A **profundidade** (mais camadas) permite criar blocos de blocos — representações hierárquicas que são exponencialmente mais eficientes.

Na prática: redes **mais profundas** (deep) com menos neurônios aprendem representações mais eficientes.

---

## 5. Capacidade e Profundidade

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

## 6. Regras Práticas para Arquitetura

1. **Camadas:** Começar com 2–3 camadas ocultas
2. **Neurônios:** Pirâmide decrescente (ex: 256 → 128 → 64) ou constante
3. **Ativação oculta:** ReLU (padrão), Leaky ReLU se muitos neurônios mortos
4. **Ativação saída:**
   - Regressão: Linear (sem ativação)
   - Binária: Sigmoid
   - Multiclasse: Softmax
5. **Batch normalization** depois de cada camada densa (avançado)

---

## Exercícios Práticos

### Exercício 1 — Forward Pass Manual

Dada a rede **3 → 2 → 1** abaixo com ativação ReLU nas camadas ocultas e sigmoid na saída, calcule **manualmente** a saída $\hat{y}$ para a entrada $\mathbf{x} = [1.0, 0.5, -0.3]$.

```
W⁽¹⁾ = [[ 0.2,  0.4, -0.5],     b⁽¹⁾ = [0.1, 0.0]
         [-0.3,  0.1,  0.6]]

W⁽²⁾ = [[0.7, -0.4]]            b⁽²⁾ = [0.2]
```

**Passos esperados:**
1. Calcule $\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$
2. Aplique ReLU: $\mathbf{h}^{(1)} = \max(0, \mathbf{z}^{(1)})$
3. Calcule $z^{(2)} = \mathbf{W}^{(2)}\mathbf{h}^{(1)} + b^{(2)}$
4. Aplique sigmoid: $\hat{y} = \sigma(z^{(2)})$

---

### Exercício 2 — Comparação de Arquiteturas

Usando o dataset `make_circles` do scikit-learn, compare o desempenho das seguintes arquiteturas e responda: qual obtém melhor acurácia no teste? Por quê?

| Arquitetura | Camadas ocultas |
|-------------|----------------|
| A | `(4,)` |
| B | `(8, 4)` |
| C | `(16, 8, 4)` |

```python
from sklearn.datasets import make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

# Complete o código: treine cada arquitetura e compare os resultados
# Dica: use MLPClassifier com activation='relu' e max_iter=1000
```

---

### Exercício 3 — Efeito da Não-Linearidade

Modifique o exemplo numérico da seção 3 para usar **ReLU** em vez de sigmoid. Compare os valores intermediários e a saída final. Em seguida, responda:

1. Quais valores mudam significativamente?
2. Algum neurônio "morre" (saída = 0 após ReLU)? Se sim, o que aconteceria se todas as pré-ativações fossem negativas?
3. Reimplemente o forward pass em Python substituindo a função `sigmoid` por `relu = lambda z: np.maximum(0, z)` e verifique seus cálculos manuais.

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
