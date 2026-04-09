# Aula 25 — Naive Bayes

> **Módulo 05 · Regressão e Classificação** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender o Teorema de Bayes e a suposição de independência condicional
- Aplicar Gaussian NB, Multinomial NB e Bernoulli NB
- Identificar quando Naive Bayes é uma boa escolha

---

## 1. Teorema de Bayes

$$P(y|\mathbf{x}) = \frac{P(\mathbf{x}|y) \cdot P(y)}{P(\mathbf{x})}$$

Classificação: escolher $y$ que maximiza $P(y|\mathbf{x})$.

**Suposição Naive:** features são **condicionalmente independentes** dado $y$:
$$P(\mathbf{x}|y) = \prod_{j=1}^{n} P(x_j|y)$$

Portanto:
$$\hat{y} = \arg\max_y P(y) \prod_{j=1}^{n} P(x_j|y)$$

---

## 2. Variantes

### Gaussian NB (features contínuas)
$$P(x_j|y) = \frac{1}{\sqrt{2\pi\sigma_{jy}^2}}\exp\left(-\frac{(x_j - \mu_{jy})^2}{2\sigma_{jy}^2}\right)$$

### Multinomial NB (contagens — texto)
Adequado para TF, TF-IDF, bag-of-words.

### Bernoulli NB (features binárias)
Para presença/ausência de palavras.

---

## 3. Implementação

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import load_iris, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

# ── Gaussian NB em dados tabulares ──────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"Gaussian NB — Acurácia: {gnb.score(X_test, y_test):.4f}")

# ── Multinomial NB em texto ──────────────────────────────────────
categories = ['sci.med', 'sci.space', 'rec.sport.baseball', 'talk.politics.guns']
news = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
X_news, y_news = news.data, news.target

pipe_nb = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
    ('nb', MultinomialNB(alpha=0.1))  # alpha = suavização de Laplace
])
scores = cross_val_score(pipe_nb, X_news, y_news, cv=5, scoring='f1_macro')
print(f"Multinomial NB (text) — F1 macro: {scores.mean():.4f} ± {scores.std():.4f}")

pipe_nb.fit(X_news, y_news)
exemplos = [
    "The astronaut launched into orbit successfully",
    "The pitcher threw a fastball in the ninth inning",
    "The doctor prescribed antibiotics for the infection"
]
preds = pipe_nb.predict(exemplos)
for ex, pred in zip(exemplos, preds):
    print(f"  '{ex[:50]}...' → {news.target_names[pred]}")
```

---

## 4. Vantagens e Limitações

| ✅ Vantagens | ❌ Limitações |
|-------------|--------------|
| Muito rápido (treino e inferência) | Suposição naive frequentemente viola realidade |
| Funciona bem com poucos dados | Não modela interações entre features |
| Excelente para NLP (texto) | Estimativas de probabilidade podem ser ruins |
| Naturalmente multiclasse | Features numéricas requerem GaussianNB |
| Robusto a ruído | Sensível ao desbalanceamento de classes |

---

## Questões para Reflexão
1. Por que Naive Bayes funciona bem em texto mesmo violando a suposição de independência?
2. O que é suavização de Laplace e por que ela é necessária?
3. Em que situações você preferiria Naive Bayes a Regressão Logística?

## Referências
- Russell & Norvig, cap. 12
- Faceli et al., cap. 5

---
*Próxima aula → [Aula 26: Ajuste de Hiperparâmetros](aula-26-hiperparametros.md)*
