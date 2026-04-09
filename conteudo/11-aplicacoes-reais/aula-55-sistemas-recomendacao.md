# Aula 55 — Sistemas de Recomendação

> **Módulo 11 · Aplicações de ML em Problemas Reais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Implementar filtragem colaborativa baseada em usuário e item
- Aplicar fatoração de matrizes (SVD, ALS)
- Compreender abordagens híbridas e baseadas em conteúdo

---

## 1. Tipos de Sistemas de Recomendação

| Tipo | Como funciona | Vantagem | Limitação |
|------|--------------|---------|---------|
| Baseado em conteúdo | Similaridade entre itens | Sem cold-start de item | Cold-start de usuário |
| Filtragem colaborativa | Padrões de usuários similares | Sem features de item | Cold-start de ambos |
| Híbrido | Combina ambos | Robusto | Complexidade |
| Knowledge-based | Regras explícitas | Sem cold-start | Rigoroso, manual |

---

## 2. Filtragem Colaborativa — KNN

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Matriz usuário-item (0 = não avaliado)
ratings = pd.DataFrame({
    'item1': [5, 4, 0, 1, 0],
    'item2': [4, 0, 3, 1, 0],
    'item3': [0, 5, 4, 0, 2],
    'item4': [1, 0, 0, 4, 5],
    'item5': [0, 3, 4, 0, 4],
}, index=['user1', 'user2', 'user3', 'user4', 'user5'])

# Similaridade entre usuários (cosine)
user_sim = pd.DataFrame(
    cosine_similarity(ratings.values),
    index=ratings.index, columns=ratings.index
)

def recomendar_para_usuario(usuario, n_rec=3, n_vizinhos=3):
    # Usuários mais similares
    vizinhos = user_sim[usuario].drop(usuario).nlargest(n_vizinhos).index
    
    # Itens não avaliados pelo usuário
    nao_avaliados = ratings.columns[ratings.loc[usuario] == 0]
    
    # Média ponderada das avaliações dos vizinhos
    scores = {}
    for item in nao_avaliados:
        pesos = user_sim[usuario][vizinhos]
        avaliações = ratings.loc[vizinhos, item]
        mask = avaliações > 0
        if mask.sum() > 0:
            scores[item] = (pesos[mask] * avaliações[mask]).sum() / pesos[mask].sum()
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n_rec]

print("Recomendações para user1:", recomendar_para_usuario('user1'))
```

---

## 3. Fatoração de Matrizes com Surprise

```python
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split
from surprise import KNNWithMeans

# Instalar: pip install scikit-surprise
reader = Reader(rating_scale=(1, 5))

# Converter para formato Surprise
data_list = []
for user in ratings.index:
    for item in ratings.columns:
        r = ratings.loc[user, item]
        if r > 0:
            data_list.append((user, item, r))

df_ratings = pd.DataFrame(data_list, columns=['user', 'item', 'rating'])
data = Dataset.load_from_df(df_ratings, reader)

# SVD (Fatoração de Matrizes)
svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02)
results = cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
print(f"SVD RMSE: {results['test_rmse'].mean():.4f}")
```

---

## 4. Sistema Baseado em Conteúdo (NLP)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Catálogo de filmes com descrições
filmes = pd.DataFrame({
    'titulo': ['The Matrix', 'Inception', 'Interstellar', 'Dark Knight', 'Memento'],
    'descricao': [
        'hacker discovers reality is a simulation programmed by machines',
        'thief who steals secrets through dream-sharing technology',
        'astronauts travel through wormhole to save humanity',
        'batman fights joker a criminal mastermind in gotham city',
        'man with short-term memory loss hunts his wife murderer'
    ]
})

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(filmes['descricao'])
sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recomendar_filme(titulo, n=3):
    idx = filmes[filmes['titulo'] == titulo].index[0]
    scores = list(enumerate(sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    return [filmes['titulo'].iloc[i] for i, _ in scores]

print(recomendar_filme('Inception'))
```

---

## Questões para Reflexão
1. O que é o problema de cold-start e como pode ser mitigado?
2. Como sistemas de recomendação podem criar "bolhas de filtro" (filter bubbles)?
3. Qual a diferença entre rating explícito e feedback implícito?

## Referências
- Faceli et al., cap. 9
- Géron, cap. 9

---
*Próxima aula → [Aula 56: MLOps](aula-56-mlops.md)*
