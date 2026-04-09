# Aula 36 — Implementando Redes Neurais com Keras

> **Módulo 07 · Redes Neurais Artificiais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Construir modelos com Sequential e Functional API do Keras
- Usar callbacks: EarlyStopping, ModelCheckpoint, TensorBoard
- Salvar, carregar e usar modelos em produção

---

## 1. Sequential API — Modelos Lineares

```python
import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s  = scaler.transform(X_te)

# Regressão com MLP
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_tr_s.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)  # saída linear para regressão
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)

model.summary()
```

---

## 2. Callbacks

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.keras', monitor='val_loss', save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
]

history = model.fit(
    X_tr_s, y_tr,
    epochs=200,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# Curvas de aprendizado
import matplotlib.pyplot as plt
pd_hist = history.history
epochs = range(1, len(pd_hist['loss'])+1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(epochs, pd_hist['loss'], label='Treino')
axes[0].plot(epochs, pd_hist['val_loss'], label='Validação')
axes[0].set_title('Loss (MSE)'); axes[0].legend()

axes[1].plot(epochs, pd_hist['mae'], label='Treino')
axes[1].plot(epochs, pd_hist['val_mae'], label='Validação')
axes[1].set_title('MAE'); axes[1].legend()
plt.tight_layout(); plt.show()

# Avaliação
test_loss, test_mae = model.evaluate(X_te_s, y_te, verbose=0)
print(f"Teste — MAE: {test_mae:.4f}")
```

---

## 3. Functional API — Modelos Complexos

```python
# Exemplo: entrada compartilhada com duas saídas
inputs = tf.keras.Input(shape=(8,))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)

# Saída 1: regressão
output_reg = tf.keras.layers.Dense(1, name='price')(x)

# Saída 2: classificação (ex: categoria de preço)
output_cls = tf.keras.layers.Dense(3, activation='softmax', name='category')(x)

model_multi = tf.keras.Model(inputs=inputs, outputs=[output_reg, output_cls])
model_multi.compile(
    optimizer='adam',
    loss={'price': 'mse', 'category': 'sparse_categorical_crossentropy'},
    loss_weights={'price': 1.0, 'category': 0.5}
)
```

---

## 4. Salvar e Carregar

```python
# Salvar modelo completo
model.save('meu_modelo.keras')

# Carregar
model_loaded = tf.keras.models.load_model('meu_modelo.keras')
y_pred = model_loaded.predict(X_te_s)

# Exportar para TensorFlow Serving / TFLite
model.export('saved_model/')
```

---

## Questões para Reflexão
1. Qual a diferença entre `restore_best_weights=True` e carregar o checkpoint manualmente?
2. Por que BatchNormalization é colocada antes (ou depois) do Dropout?
3. Como você converteria este modelo para rodar em um dispositivo mobile?

## Referências
- Géron, cap. 10–11
- Documentação oficial: keras.io

---
*Módulo 07 concluído! Próximo → [Módulo 08: Avaliação e Validação de Modelos](../08-avaliacao-validacao/README.md)*
