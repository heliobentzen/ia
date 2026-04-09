# Aula 44 — Dropout e Early Stopping

> **Módulo 09 · Overfitting e Regularização** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender o mecanismo do Dropout e sua interpretação como ensemble
- Implementar Early Stopping com restauração dos melhores pesos
- Aplicar Batch Normalization para estabilizar o treinamento

---

## 1. Dropout

Em cada mini-batch de treino, cada neurônio é "desligado" com probabilidade $p$. Em inferência, todos os neurônios são usados e os pesos são escalados por $(1-p)$.

**Interpretação:** equivale a treinar um ensemble de $2^n$ redes diferentes e fazer a média das predições.

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=5000, n_features=50, n_informative=20, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(dropout_rate=0.0, batch_norm=False):
    layers = [tf.keras.layers.Input(shape=(50,))]
    for units in [256, 128, 64]:
        layers.append(tf.keras.layers.Dense(units))
        if batch_norm:
            layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Activation('relu'))
        if dropout_rate > 0:
            layers.append(tf.keras.layers.Dropout(dropout_rate))
    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    model = tf.keras.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Comparar dropout rates
configs = {
    'Sem dropout': {'dropout_rate': 0.0, 'batch_norm': False},
    'Dropout 0.3':  {'dropout_rate': 0.3, 'batch_norm': False},
    'Dropout 0.5':  {'dropout_rate': 0.5, 'batch_norm': False},
    'Dropout 0.3 + BN': {'dropout_rate': 0.3, 'batch_norm': True},
}

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                      restore_best_weights=True)
]

results = {}
for name, cfg in configs.items():
    model = build_model(**cfg)
    hist = model.fit(X_tr, y_tr, epochs=100, batch_size=128,
                     validation_split=0.2, callbacks=callbacks, verbose=0)
    test_acc = model.evaluate(X_te, y_te, verbose=0)[1]
    train_acc = model.evaluate(X_tr, y_tr, verbose=0)[1]
    results[name] = {'train': train_acc, 'test': test_acc}
    print(f"{name:25s}: treino={train_acc:.4f} | teste={test_acc:.4f}")
```

---

## 2. Early Stopping

Para o treinamento quando a métrica de validação para de melhorar:

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,           # épocas sem melhora antes de parar
    restore_best_weights=True,  # restaura pesos da melhor época
    min_delta=1e-4         # melhora mínima considerada significativa
)

# Em TF, o EarlyStopping pode ser combinado com ReduceLROnPlateau
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=7,
    min_lr=1e-6, verbose=1
)

model.fit(X_tr, y_tr, epochs=500,
          callbacks=[early_stop, reduce_lr],
          validation_split=0.2, verbose=0)
print(f"Treinamento parou na época: {early_stop.stopped_epoch}")
```

---

## 3. Batch Normalization

Normaliza as ativações de cada camada durante o treinamento:

$$\hat{z} = \frac{z - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y = \gamma\hat{z} + \beta$$

**Benefícios:**
- Permite taxas de aprendizado maiores
- Reduz sensibilidade à inicialização
- Funciona como regularizador (dispensando parcialmente o Dropout)

```python
# BN antes ou depois da ativação?
# Autores originais: Linear → BN → Ativação
# Prática moderna: Linear → Ativação → BN (ambos funcionam)

tf.keras.layers.Dense(128),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),
```

---

## Questões para Reflexão
1. Por que o Dropout se comporta diferente no treino e na inferência?
2. Qual é a relação entre o Dropout e o Bagging?
3. O Batch Normalization é colocado antes ou depois da função de ativação?

## Referências
- Géron, cap. 11
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 45: Data Augmentation](aula-45-data-augmentation.md)*
