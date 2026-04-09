# Aula 35 — Otimizadores e Funções de Ativação Modernas

> **Módulo 07 · Redes Neurais Artificiais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender e comparar SGD, Momentum, RMSProp e Adam
- Aplicar learning rate scheduling
- Escolher a função de ativação correta para cada contexto

---

## 1. Gradiente Descendente e Variantes

### SGD com Momentum
Acumula velocidade na direção de gradientes consistentes:
$$\mathbf{v} \leftarrow \beta\mathbf{v} - \eta\nabla J$$
$$\boldsymbol{\theta} \leftarrow \boldsymbol{\theta} + \mathbf{v}$$

### RMSProp
Adapta o LR para cada parâmetro dividindo pelo RMS dos gradientes recentes:
$$s \leftarrow \rho s + (1-\rho)g^2; \quad \theta \leftarrow \theta - \frac{\eta}{\sqrt{s+\epsilon}}g$$

### Adam — Adaptive Moment Estimation (padrão atual)
Combina Momentum + RMSProp com correção de bias:
$$m \leftarrow \beta_1 m + (1-\beta_1)g; \quad v \leftarrow \beta_2 v + (1-\beta_2)g^2$$
$$\hat{m} = \frac{m}{1-\beta_1^t}; \quad \hat{v} = \frac{v}{1-\beta_2^t}$$
$$\theta \leftarrow \theta - \eta\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}$$

**Parâmetros padrão:** $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-7}$, $\eta=0.001$

---

## 2. Implementação no Keras

```python
import tensorflow as tf
import numpy as np

# Comparar otimizadores
optimizers = {
    'SGD':           tf.keras.optimizers.SGD(learning_rate=0.01),
    'SGD+Momentum':  tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSProp':       tf.keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam':          tf.keras.optimizers.Adam(learning_rate=0.001),
    'AdamW':         tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
}

def build_model(optimizer):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(20,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Dataset sintético
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
X_tr, X_te = X[:1600], X[1600:]
y_tr, y_te = y[:1600], y[1600:]

results = {}
for name, opt in optimizers.items():
    model = build_model(opt)
    hist = model.fit(X_tr, y_tr, epochs=50, batch_size=32,
                     validation_split=0.2, verbose=0)
    acc = model.evaluate(X_te, y_te, verbose=0)[1]
    results[name] = {'val_acc': max(hist.history['val_accuracy']), 'test_acc': acc}
    print(f"{name:15s}: val={results[name]['val_acc']:.4f} | test={acc:.4f}")
```

---

## 3. Learning Rate Scheduling

```python
# ReduceLROnPlateau: reduz LR quando val_loss para de melhorar
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
)

# Cosine Annealing
cosine_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.001, decay_steps=1000
)
opt_cosine = tf.keras.optimizers.Adam(learning_rate=cosine_schedule)

# Warmup + Decay (padrão em Transformers)
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.peak_lr * step / self.warmup_steps
        cosine_lr = self.peak_lr * 0.5 * (1 + tf.cos(
            np.pi * (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        ))
        return tf.where(step < self.warmup_steps, warmup_lr, cosine_lr)
```

---

## 4. Funções de Ativação Modernas

```python
import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(-4, 4, 200)

activations = {
    'ReLU':       np.maximum(0, z),
    'Leaky ReLU': np.where(z > 0, z, 0.01*z),
    'ELU':        np.where(z > 0, z, np.expm1(z)),
    'GELU':       z * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(z + 0.044715*z**3))),
    'Swish':      z / (1 + np.exp(-z)),
}

fig, axes = plt.subplots(1, len(activations), figsize=(18, 4))
for ax, (name, vals) in zip(axes, activations.items()):
    ax.plot(z, vals, linewidth=2)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_title(name)
    ax.set_ylim(-2, 4)
plt.tight_layout()
plt.show()
```

---

## Questões para Reflexão
1. Por que Adam é o otimizador padrão para a maioria dos modelos de deep learning?
2. O que é weight decay e qual sua relação com regularização L2?
3. Em qual cenário SGD com momentum pode superar Adam?

## Referências
- Géron, cap. 11
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 36: Implementando Redes Neurais com Keras](aula-36-implementando-rna-keras.md)*
