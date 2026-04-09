# Aula 45 — Data Augmentation

> **Módulo 09 · Overfitting e Regularização** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender data augmentation como regularização implícita
- Aplicar augmentation em imagens com Keras e Albumentations
- Usar SMOTE para balancear classes em dados tabulares
- Conhecer técnicas de augmentation para texto

---

## 1. O Que é Data Augmentation?

Criar **novas amostras artificiais** a partir das existentes por meio de transformações que preservam o rótulo. Aumenta o dataset efetivo sem coletar novos dados.

---

## 2. Augmentation para Imagens

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Camada de augmentation integrada ao modelo (recomendado)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),       # ±10%
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomTranslation(0.1, 0.1),
])

# Exemplo com CIFAR-10
(X_tr, y_tr), (X_te, y_te) = tf.keras.datasets.cifar10.load_data()
X_tr = X_tr.astype('float32') / 255.0
X_te = X_te.astype('float32') / 255.0

# Visualizar augmentations
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for i in range(8):
    img = X_tr[i]
    axes[0, i].imshow(img); axes[0, i].axis('off')
    for j in [1, 2]:
        aug = data_augmentation(img[np.newaxis], training=True)[0]
        axes[j, i].imshow(aug.numpy().clip(0, 1)); axes[j, i].axis('off')
plt.suptitle('Original (linha 1) | Augmented (linhas 2-3)')
plt.tight_layout(); plt.show()

# Modelo com augmentation embutida
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    data_augmentation,  # só ativa durante training=True
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

## 3. SMOTE — Balanceamento de Classes Tabulares

```python
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from collections import Counter

# Dataset muito desbalanceado: 95% negativos, 5% positivos
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
print(f"Antes do SMOTE: {Counter(y)}")

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"Após o SMOTE:   {Counter(y_resampled)}")
```

> ⚠️ Aplique SMOTE **apenas no treino**, nunca no conjunto de validação/teste.

---

## 4. Augmentation para Texto (NLP)

```python
# Técnicas comuns:
# 1. Synonym replacement
# 2. Random insertion / deletion / swap
# 3. Back-translation (PT → EN → PT)
# 4. Paraphrase with LLMs

# Exemplo simples: troca de sinônimos com NLTK
import random
try:
    from nltk.corpus import wordnet
    import nltk
    nltk.download('wordnet', quiet=True)

    def synonym_replacement(text, n=2):
        words = text.split()
        augmented = words.copy()
        changed = 0
        for i, word in enumerate(words):
            if changed >= n:
                break
            syns = wordnet.synsets(word, lang='por')
            if syns:
                syn_words = [l.name().replace('_', ' ') for l in syns[0].lemmas('por')]
                if syn_words and syn_words[0] != word:
                    augmented[i] = random.choice(syn_words)
                    changed += 1
        return ' '.join(augmented)
except Exception:
    print("NLTK wordnet não disponível — use uma biblioteca como nlpaug")
```

---

## Questões para Reflexão
1. Por que o data augmentation deve ser aplicado apenas no treino?
2. Que transformações de imagem NÃO fazem sentido para classificação de dígitos manuscritos (MNIST)?
3. SMOTE cria novas amostras reais ou sintéticas? Isso pode ser um problema?

## Referências
- Géron, cap. 11, 14
- Faceli et al., cap. 3

---
*Módulo 09 concluído! Próximo → [Módulo 10: Introdução ao Aprendizado Profundo](../10-aprendizado-profundo/README.md)*
