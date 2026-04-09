# Aula 48 — RNNs, LSTMs e GRUs

> **Módulo 10 · Introdução ao Aprendizado Profundo** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender o funcionamento das RNNs e seus problemas (gradiente sumindo)
- Implementar LSTM e GRU para processamento de sequências
- Aplicar modelos sequenciais para previsão de séries temporais

---

## 1. RNN — Recurrent Neural Network

Redes que mantêm **estado oculto** entre timesteps:
$$\mathbf{h}_t = f(\mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{xh}\mathbf{x}_t + \mathbf{b}_h)$$

**Problema:** em sequências longas, gradientes explodem ou somem → LSTM resolve.

---

## 2. LSTM — Long Short-Term Memory

```
      forget gate    input gate    output gate
          ↓               ↓              ↓
c_t = f_t * c_{t-1} + i_t * g_t
h_t = o_t * tanh(c_t)
```

- **Forget gate** $f_t$: o que esquecer da memória anterior
- **Input gate** $i_t$: o que adicionar à memória
- **Output gate** $o_t$: o que ativar na saída

---

## 3. Previsão de Séries Temporais com LSTM

```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Gerar série temporal sintética (senoide com tendência + ruído)
np.random.seed(42)
t = np.arange(0, 400)
serie = np.sin(0.1*t) + 0.005*t + np.random.normal(0, 0.2, 400)

# Criar janelas deslizantes
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

SEQ_LEN = 30
X_seq, y_seq = create_sequences(serie, SEQ_LEN)
X_seq = X_seq.reshape(-1, SEQ_LEN, 1)  # (samples, timesteps, features)

# Split
split = int(0.8 * len(X_seq))
X_tr, X_te = X_seq[:split], X_seq[split:]
y_tr, y_te = y_seq[:split], y_seq[split:]

# Modelo LSTM
model_lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])
model_lstm.compile('adam', 'mse', metrics=['mae'])

# GRU alternativo (mais rápido que LSTM)
model_gru = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(1)
])
model_gru.compile('adam', 'mse', metrics=['mae'])

callbacks = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

for name, model in [('LSTM', model_lstm), ('GRU', model_gru)]:
    model.fit(X_tr, y_tr, epochs=100, batch_size=32,
              validation_split=0.15, callbacks=callbacks, verbose=0)
    mae = model.evaluate(X_te, y_te, verbose=0)[1]
    print(f"{name} — MAE no teste: {mae:.4f}")
```

---

## 4. LSTM vs. GRU vs. Transformer

| Modelo | Parâmetros | Memória longa | Paralelizável | Uso |
|--------|-----------|--------------|--------------|-----|
| RNN | Baixo | Ruim | Não | Sequências curtas |
| LSTM | Médio | Boa | Não | Séries temporais, NLP |
| GRU | Médio | Boa | Não | Alternativa mais rápida ao LSTM |
| Transformer | Alto | Excelente | Sim | NLP, imagens, moderno |

---

## Questões para Reflexão
1. Por que a GRU tem menos parâmetros que a LSTM e ainda assim funciona bem?
2. Em que situações um Transformer pode ser preferível ao LSTM para séries temporais?
3. O que é `return_sequences=True` e quando deve ser usado?

## Referências
- Géron, cap. 15
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 49: Transformers e Atenção](aula-49-transformers-atencao.md)*
