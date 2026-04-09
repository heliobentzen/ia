# Prática 11 — CNN para Visão Computacional

**Módulo:** 10 | **Duração:** ~120 minutos | **Dataset:** CIFAR-10

## Objetivos
- Implementar CNN do zero para CIFAR-10
- Aplicar transfer learning com EfficientNetB0
- Comparar CNN própria vs. transfer learning

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0
y_train = y_train.ravel()
y_test  = y_test.ravel()

CLASS_NAMES = ['avião','automóvel','pássaro','gato','cervo',
               'cachorro','sapo','cavalo','navio','caminhão']

# Data augmentation
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# CNN do zero
def build_cnn():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(32,32,3)),
        augment,
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(2),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

cnn_scratch = build_cnn()
cnn_scratch.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy', metrics=['accuracy']
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
]

hist = cnn_scratch.fit(X_train, y_train, epochs=50, batch_size=128,
                        validation_split=0.1, callbacks=callbacks)

acc_scratch = cnn_scratch.evaluate(X_test, y_test, verbose=0)[1]
print(f"CNN from scratch — Acurácia: {acc_scratch:.4f}")

# Transfer Learning com EfficientNetB0 (imagens 32x32 → resize para 224x224)
X_train_big = tf.image.resize(X_train, (96, 96))
X_test_big  = tf.image.resize(X_test, (96, 96))

base = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet',
                                              input_shape=(96, 96, 3))
base.trainable = False

tl_model = tf.keras.Sequential([
    base,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])
tl_model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
tl_model.fit(X_train_big, y_train, epochs=5, batch_size=64,
              validation_split=0.1, callbacks=callbacks)

acc_tl = tl_model.evaluate(X_test_big, y_test, verbose=0)[1]
print(f"Transfer Learning (EfficientNetB0) — Acurácia: {acc_tl:.4f}")

# Visualizar erros
y_pred = cnn_scratch.predict(X_test, verbose=0).argmax(axis=1)
wrong = np.where(y_pred != y_test)[0][:12]
fig, axes = plt.subplots(3, 4, figsize=(10, 8))
for ax, idx in zip(axes.ravel(), wrong):
    ax.imshow(X_test[idx])
    ax.set_title(f'Real:{CLASS_NAMES[y_test[idx]]}\nPred:{CLASS_NAMES[y_pred[idx]]}', fontsize=7)
    ax.axis('off')
plt.tight_layout(); plt.show()
```

## Desafios Extras
1. Fine-tune as últimas 20 camadas do EfficientNetB0 com LR de 1e-5
2. Use a Functional API para criar um modelo ResNet-style com skip connections
3. Implemente Grad-CAM para visualizar quais regiões da imagem o modelo usa para classificar
