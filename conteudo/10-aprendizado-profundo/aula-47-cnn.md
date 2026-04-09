# Aula 47 — Redes Neurais Convolucionais (CNN)

> **Módulo 10 · Introdução ao Aprendizado Profundo** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender operações de convolução, pooling e feature maps
- Implementar uma CNN para classificação de imagens com Keras
- Conhecer as arquiteturas clássicas (LeNet, AlexNet, VGG, ResNet)

---

## 1. Operação de Convolução

$$(\mathbf{I} * \mathbf{K})[i,j] = \sum_{m}\sum_{n} \mathbf{I}[i+m, j+n] \cdot \mathbf{K}[m,n]$$

- **Filtros (kernels)** aprendem detectores de features: bordas, texturas, formas
- **Feature maps**: saída da convolução — mapas de ativação
- **Parâmetro compartilhado**: o mesmo kernel varre toda a imagem (translation invariance)

---

## 2. Camadas Principais

| Camada | Função |
|--------|--------|
| Conv2D | Extração de features locais |
| MaxPooling2D | Redução espacial (downsampling) |
| GlobalAveragePooling2D | Colapso espacial → vetor |
| BatchNormalization | Estabilização do treino |
| Dropout | Regularização |
| Dense | Classificação final |

---

## 3. Implementação — Classificação CIFAR-10

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.cifar10.load_data()
X_tr = X_tr.astype('float32') / 255.0
X_te = X_te.astype('float32') / 255.0

class_names = ['avião','automóvel','pássaro','gato','cervo',
               'cachorro','sapo','cavalo','navio','caminhão']

# Data augmentation
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Modelo CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    augment,

    # Bloco 1
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.2),

    # Bloco 2
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Dropout(0.3),

    # Bloco 3
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.4),

    # Classificador
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
]

history = model.fit(
    X_tr, y_tr, epochs=50, batch_size=128,
    validation_split=0.15, callbacks=callbacks, verbose=1
)

test_loss, test_acc = model.evaluate(X_te, y_te, verbose=0)
print(f"Acurácia no teste: {test_acc:.4f}")
```

---

## 4. Arquiteturas Famosas

| Arquitetura | Ano | Inovação |
|------------|-----|---------|
| LeNet-5 | 1998 | Primeira CNN prática (dígitos) |
| AlexNet | 2012 | ReLU, Dropout, GPU → ImageNet |
| VGG | 2014 | Kernels 3×3 empilhados |
| ResNet | 2015 | Skip connections (redes muito profundas) |
| EfficientNet | 2019 | Escalonamento composto |
| ViT | 2020 | Transformer para imagens |
| ConvNeXt | 2022 | CNN moderna inspirada em ViT |

---

## Questões para Reflexão
1. Por que kernels 3×3 são preferidos a kernels maiores em redes modernas?
2. O que é o problema do gradiente que sumiu em redes profundas e como o ResNet resolve?
3. Para que serve o GlobalAveragePooling2D em vez de Flatten?

## Referências
- Géron, cap. 14
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 48: RNNs, LSTMs e GRUs](aula-48-rnn-lstm.md)*
