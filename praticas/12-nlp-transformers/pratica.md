# Prática 12 — NLP com Transformers

**Módulo:** 10–11 | **Duração:** ~120 minutos

## Objetivos
- Fazer fine-tuning de BERT para classificação de texto
- Usar HuggingFace Pipelines para tarefas NLP
- Processar texto em português com modelos multilíngues

```python
from transformers import (pipeline, AutoTokenizer,
                           AutoModelForSequenceClassification, TrainingArguments, Trainer)
from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# ── 1. Pipelines pré-construídos ──────────────────────────────────
print("=== Análise de Sentimentos ===")
sentimentos = pipeline("text-classification",
                       model="cardiffnlp/twitter-xlm-roberta-base-sentiment")
textos = [
    "Adorei o produto! Superou todas as expectativas.",
    "Péssima experiência. O produto chegou quebrado.",
    "O produto é razoável. Entrega dentro do prazo."
]
for t, r in zip(textos, sentimentos(textos)):
    print(f"  {t[:50]}... → {r['label']} ({r['score']:.3f})")

print("\n=== NER em Português ===")
ner = pipeline("ner", model="pierreguillou/ner-bert-base-cased-pt-lenerbr",
               aggregation_strategy="simple")
texto_ner = "O presidente Luiz Inácio Lula da Silva se reuniu com a ministra Simone Tebet em Brasília."
for ent in ner(texto_ner):
    print(f"  {ent['word']:20s} → {ent['entity_group']} ({ent['score']:.3f})")

print("\n=== Geração de Texto ===")
gerador = pipeline("text-generation", model="gpt2", max_new_tokens=50)
resultado = gerador("Machine learning is", do_sample=True, temperature=0.7)
print(resultado[0]['generated_text'])

# ── 2. Fine-tuning BERT ───────────────────────────────────────────
# Dataset de avaliações de produtos (sintético)
data = {
    'text': [
        "Produto excelente, superou minhas expectativas!", "Chegou rápido e bem embalado.",
        "Qualidade ruim, não recomendo.", "Péssimo atendimento ao cliente.",
        "Bom custo-benefício, compraria novamente.", "Produto ok, nada especial.",
        "Totalmente decepcionante, dinheiro jogado fora.", "Ótima qualidade e durabilidade!",
        "Não funcionou como esperado.", "Entrega atrasou mas o produto é bom."
    ],
    'label': [1, 1, 0, 0, 1, 2, 0, 1, 0, 2]  # 0=neg, 1=pos, 2=neutro
}

df = pd.DataFrame(data)
dataset = Dataset.from_pandas(df)

model_name = "neuralmind/bert-base-portuguese-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_fn(examples):
    return tokenizer(examples['text'], padding='max_length',
                     truncation=True, max_length=128)

dataset = dataset.map(tokenize_fn, batched=True)
dataset = dataset.train_test_split(test_size=0.3, seed=42)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import f1_score
    return {'f1': f1_score(labels, preds, average='macro')}

args = TrainingArguments(
    output_dir='./bert-sentiment',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    learning_rate=2e-5,
    evaluation_strategy='epoch',
    save_strategy='no',
    logging_steps=10,
    warmup_ratio=0.1,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)
trainer.train()
metrics = trainer.evaluate()
print(f"\nF1 macro no teste: {metrics['eval_f1']:.4f}")
```

## Desafios Extras
1. Aplique o modelo fine-tuned para classificar avaliações do Mercado Livre ou Amazon Brasil
2. Use `sentence-transformers` para calcular similaridade semântica entre frases
3. Implemente um sistema simples de FAQ com busca semântica usando embeddings
