# Aula 52 — Processamento de Linguagem Natural (NLP)

> **Módulo 11 · Aplicações de ML em Problemas Reais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Aplicar tokenização, embeddings e bag-of-words
- Usar BERT/HuggingFace para análise de sentimentos e NER
- Processar textos em português com modelos multilíngues

---

## 1. Pipeline NLP Tradicional

```python
import re
import string
from collections import Counter
import numpy as np

def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

exemplos = [
    "O produto chegou RÁPIDO! Adorei 😍 #compras",
    "Terrível. Não comprem nunca mais nessa loja.",
    "Serviço ok, mas o produto estava danificado."
]
textos = [preprocess(t) for t in exemplos]
print(textos)
```

---

## 2. TF-IDF com Scikit-learn

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Dataset de sentimentos
corpus = [
    "produto excelente superou expectativas",
    "adorei a compra chegou rapido",
    "produto pessimo nao recomendo",
    "terrivel qualidade horrorosa",
    "produto normal sem problemas",
    "entrega ok qualidade mediana"
]
labels = [1, 1, 0, 0, 2, 2]  # 0=neg, 1=pos, 2=neutro

pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=5000)),
    ('lr', LogisticRegression(max_iter=1000, C=1.0))
])
pipe.fit(corpus, labels)
novos = ["produto muito bom", "terrivel experiencia de compra"]
print(pipe.predict(novos))
```

---

## 3. Word Embeddings

```python
# Word2Vec com Gensim
from gensim.models import Word2Vec

sentencas = [
    ["rei", "rainha", "trono", "coroa"],
    ["homem", "mulher", "pessoa", "humano"],
    ["cachorro", "gato", "animal", "pet"],
    ["python", "programacao", "codigo", "algoritmo"]
]

w2v = Word2Vec(sentencas, vector_size=50, window=3, min_count=1, epochs=100)

# Operações semânticas
print(w2v.wv.most_similar('rei'))
# rei - homem + mulher ≈ rainha
```

---

## 4. BERT para Análise de Sentimentos (HuggingFace)

```python
from transformers import pipeline

# Modelo multilíngue (funciona em português!)
sentiment = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    tokenizer="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

textos_pt = [
    "Estou muito feliz com minha compra! Produto excelente.",
    "Péssimo atendimento, nunca mais compro aqui.",
    "O produto é razoável, não é nem bom nem ruim."
]

resultados = sentiment(textos_pt)
for texto, res in zip(textos_pt, resultados):
    print(f"'{texto[:50]}...' → {res['label']} ({res['score']:.3f})")
```

---

## 5. NER — Reconhecimento de Entidades Nomeadas

```python
ner = pipeline("ner", model="pierreguillou/ner-bert-base-cased-pt-lenerbr",
               aggregation_strategy="simple")

texto = "O presidente Lula se reuniu com o ministro Haddad em Brasília na quinta-feira."
entidades = ner(texto)
for ent in entidades:
    print(f"{ent['word']:20s} → {ent['entity_group']} ({ent['score']:.3f})")
```

---

## Questões para Reflexão
1. Por que TF-IDF captura melhor a relevância de palavras que a simples contagem?
2. O que são embeddings contextuais (BERT) e como diferem do Word2Vec?
3. Em que situações NER é mais útil que análise de sentimentos?

## Referências
- Tunstall et al., cap. 2–6
- Géron, cap. 16

---
*Próxima aula → [Aula 53: Visão Computacional](aula-53-visao-computacional.md)*
