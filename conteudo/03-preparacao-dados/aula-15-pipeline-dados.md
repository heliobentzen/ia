# Aula 15 — Pipeline de Dados com Scikit-learn

> **Módulo 03 · Preparação e Análise de Dados** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Construir pipelines reprodutíveis com `sklearn.pipeline.Pipeline`
- Combinar pré-processamento e modelo em um único objeto
- Usar `ColumnTransformer` para tratar features numéricas e categóricas juntas

---

## 1. O Problema sem Pipeline

```python
# ❌ Código frágil: fácil de cometer data leakage
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
model.fit(X_train_s, y_train)

X_test_s = scaler.transform(X_test)  # ok, mas e se esquecer?
```

---

## 2. Pipeline Básico

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(pipe.score(X_test, y_test))
```

> ✅ O pipeline garante que o scaler seja ajustado **apenas** no treino.

---

## 3. ColumnTransformer — Tratamento Heterogêneo

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

num_features = ['age', 'income', 'score']
cat_features = ['city', 'education', 'gender']

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features)
])

full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

full_pipe.fit(X_train, y_train)
print(f"Acurácia: {full_pipe.score(X_test, y_test):.4f}")
```

---

## 4. Pipeline + Cross-Validation + GridSearch

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'model__C': [0.01, 0.1, 1, 10],
    'model__solver': ['lbfgs', 'liblinear']
}

grid = GridSearchCV(full_pipe, param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Melhores parâmetros: {grid.best_params_}")
print(f"Melhor F1: {grid.best_score_:.4f}")
```

---

## 5. Salvando e Carregando o Pipeline

```python
import joblib

# Salvar
joblib.dump(full_pipe, 'modelo_pipeline.pkl')

# Carregar
pipe_loaded = joblib.load('modelo_pipeline.pkl')
y_pred = pipe_loaded.predict(novos_dados)
```

---

## Questões para Reflexão
1. Por que encapsular o pré-processamento no pipeline facilita o deploy?
2. Como você adicionaria uma etapa de seleção de features ao pipeline?
3. O que acontece se você usar `fit_transform` no conjunto de teste dentro de um pipeline?

## Referências
- Géron, cap. 2 (Pipelines de Transformação)
- Documentação Scikit-learn: `sklearn.pipeline`

---
*Módulo concluído! Próximo → [Módulo 04: Aprendizado Supervisionado e Não Supervisionado](../04-supervisionado-nao-supervisionado/README.md)*
