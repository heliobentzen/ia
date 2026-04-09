# Prática 09 — Redes Neurais com Keras

**Módulo:** 07 | **Duração:** ~90 minutos | **Dataset:** MNIST

## Objetivos
- Implementar MLP e CNN para classificação de dígitos
- Usar callbacks: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- Visualizar curvas de aprendizado e exemplos de erro

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Carregar MNIST
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0
X_train_flat = X_train.reshape(-1, 784)
X_test_flat  = X_test.reshape(-1, 784)

print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

# MLP
mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),
    tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

mlp.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
mlp.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1)
]
hist_mlp = mlp.fit(X_train_flat, y_train, epochs=30, batch_size=256,
                    validation_split=0.1, callbacks=callbacks, verbose=1)

print(f"MLP — Acurácia no teste: {mlp.evaluate(X_test_flat, y_test, verbose=0)[1]:.4f}")

# CNN
X_train_cnn = X_train[..., np.newaxis]  # (60000, 28, 28, 1)
X_test_cnn  = X_test[..., np.newaxis]

cnn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

cnn.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
hist_cnn = cnn.fit(X_train_cnn, y_train, epochs=20, batch_size=128,
                    validation_split=0.1, callbacks=callbacks, verbose=1)

cnn_acc = cnn.evaluate(X_test_cnn, y_test, verbose=0)[1]
print(f"CNN — Acurácia no teste: {cnn_acc:.4f}")

# Visualizar erros
y_pred = cnn.predict(X_test_cnn, verbose=0).argmax(axis=1)
wrong_idx = np.where(y_pred != y_test)[0]

fig, axes = plt.subplots(3, 5, figsize=(12, 8))
for ax, idx in zip(axes.ravel(), wrong_idx[:15]):
    ax.imshow(X_test[idx], cmap='gray')
    ax.set_title(f'Real:{y_test[idx]} Pred:{y_pred[idx]}', fontsize=8, color='red')
    ax.axis('off')
plt.suptitle('Exemplos de Classificação Errada — CNN')
plt.tight_layout(); plt.show()
```

## Desafios Extras
1. Adicione data augmentation (rotação, zoom leve) e compare com CNN sem augmentation
2. Teste a mesma CNN no Fashion-MNIST e compare os resultados
3. Visualize os feature maps da primeira camada Conv2D para uma imagem de entrada
