# Aula 50 — Transfer Learning e Fine-Tuning

> **Módulo 10 · Introdução ao Aprendizado Profundo** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Aplicar transfer learning com modelos pré-treinados de visão e NLP
- Entender feature extraction vs. fine-tuning
- Implementar fine-tuning com HuggingFace Transformers

---

## 1. O Que é Transfer Learning?

Reutilizar conhecimento aprendido em uma tarefa para resolver outra:
```
Tarefa Fonte (ImageNet, Common Crawl)
    ↓ pré-treinamento
Modelo com representações ricas
    ↓ fine-tuning
Tarefa Alvo (sua tarefa específica)
```

---

## 2. Feature Extraction vs. Fine-Tuning

| Estratégia | Quando usar | Como fazer |
|-----------|------------|-----------|
| Feature extraction | Poucos dados; domínio próximo | Congela pesos base, treina só o classificador |
| Fine-tuning parcial | Dados moderados | Congela primeiras camadas, treina últimas |
| Fine-tuning completo | Muitos dados; domínio diferente | Treina tudo com LR baixo |

---

## 3. Transfer Learning para Visão (Keras)

```python
import tensorflow as tf

# Carregar EfficientNetB0 pré-treinado no ImageNet
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# FASE 1: Feature Extraction — congela a base
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(5, activation='softmax')(x)  # 5 classes

model = tf.keras.Model(inputs, outputs)
model.compile('adam', 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar apenas o classificador
# model.fit(...)

# FASE 2: Fine-Tuning — descongela as últimas 20 camadas
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # LR muito baixo!
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"Parâmetros treináveis: {sum(v.numpy().size for v in model.trainable_weights):,}")
```

---

## 4. Fine-Tuning NLP com HuggingFace

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import numpy as np
from datasets import load_dataset

# Carregar modelo e tokenizer BERT
model_name = "neuralmind/bert-base-portuguese-cased"  # BERT em português!
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Dataset de exemplo
dataset = load_dataset("amazon_polarity", split={'train': 'train[:1000]', 'test': 'test[:200]'})

def tokenize(batch):
    return tokenizer(batch['content'], padding=True, truncation=True, max_length=128)

dataset = dataset.map(tokenize, batched=True)

# Treinamento
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

trainer.train()
```

---

## Questões para Reflexão
1. Por que a taxa de aprendizado no fine-tuning deve ser muito menor que no treinamento do zero?
2. O que é "catastrophic forgetting" e como o fine-tuning pode causar isso?
3. Quando pode ser melhor treinar do zero em vez de usar transfer learning?

## Referências
- Géron, cap. 14, 16
- Tunstall et al., cap. 2–4

---
*Próxima aula → [Aula 51: LLMs e Foundation Models](aula-51-llms-foundation-models.md)*
