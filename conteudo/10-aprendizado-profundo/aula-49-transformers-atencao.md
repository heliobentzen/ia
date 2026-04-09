# Aula 49 — Mecanismo de Atenção e Transformers

> **Módulo 10 · Introdução ao Aprendizado Profundo** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender o mecanismo de self-attention e multi-head attention
- Entender a arquitetura Transformer (encoder-decoder)
- Conhecer modelos BERT, GPT e sua relação com o Transformer

---

## 1. Motivação: O Problema do Gargalo

Em Seq2Seq com LSTM, toda a informação da sequência de entrada é comprimida em **um único vetor** (encoder final state). Isso é problemático para sequências longas.

**Atenção:** em vez de usar apenas o estado final, a atenção usa **todos os estados do encoder** e aprende quais são mais relevantes para cada posição da saída.

---

## 2. Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- **Q** (Queries): o que estou procurando
- **K** (Keys): o que cada posição oferece
- **V** (Values): o conteúdo que será agregado
- $\sqrt{d_k}$: escala para evitar gradientes pequenos

---

## 3. Multi-Head Attention

Executa h atenções em paralelo em subEspaços diferentes:
$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

---

## 4. Arquitetura Transformer Completa

```
ENCODER:                          DECODER:
┌────────────────────────┐        ┌────────────────────────┐
│ Multi-Head Self-Attn   │        │ Masked Multi-Head Attn  │
│ Add & Norm             │        │ Add & Norm              │
│ Feed Forward           │   →    │ Cross-Attention          │
│ Add & Norm             │        │ Add & Norm              │
└────────────────────────┘        │ Feed Forward            │
  × N layers                      │ Add & Norm              │
                                  └────────────────────────┘
                                    × N layers
```

---

## 5. BERT vs. GPT

| Modelo | Tipo | Pre-training | Uso |
|--------|------|-------------|-----|
| BERT | Encoder | Masked Language Model | Classificação, NER, QA |
| GPT | Decoder | Causal Language Model | Geração de texto |
| T5 | Encoder-Decoder | Span corruption | Tradução, sumarização |
| BART | Encoder-Decoder | Denoising | Sumarização, geração |

---

## 6. Implementação Simplificada de Self-Attention

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)
    
    if mask is not None:
        scores = scores + mask * (-1e9)
    
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    output = weights @ V
    return output, weights

# Demonstração
np.random.seed(42)
seq_len = 5
d_model = 8

# Sequência de entrada (5 tokens, 8 dimensões)
X = np.random.randn(seq_len, d_model)

W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Input shape: {X.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights (token 0):\n{weights[0].round(3)}")
```

---

## Questões para Reflexão
1. Por que dividir por $\sqrt{d_k}$ no mecanismo de atenção?
2. O que é "atenção causal" (masked) e por que é necessária em modelos GPT?
3. O que são positional encodings e por que o Transformer precisa deles?

## Referências
- Géron, cap. 16
- Tunstall et al., cap. 1–3
- Vaswani et al. "Attention Is All You Need" (2017)

---
*Próxima aula → [Aula 50: Transfer Learning](aula-50-transfer-learning.md)*
