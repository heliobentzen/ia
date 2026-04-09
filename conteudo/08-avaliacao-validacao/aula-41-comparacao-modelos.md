# Aula 41 — Comparação de Modelos

> **Módulo 08 · Avaliação e Validação de Modelos** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Comparar modelos usando baseline e testes estatísticos
- Aplicar o teste de Wilcoxon e o teste de Friedman
- Evitar erros comuns na comparação de modelos

---

## 1. Sempre Compare com um Baseline

Antes de qualquer modelo sofisticado, estabeleça um baseline:

```python
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# Baselines
dummy_majority = DummyClassifier(strategy='most_frequent')
dummy_stratified = DummyClassifier(strategy='stratified')
rf = RandomForestClassifier(n_estimators=100, random_state=42)

models = {
    'Majority Baseline': dummy_majority,
    'Stratified Baseline': dummy_stratified,
    'Random Forest': rf
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=10, scoring='f1')
    print(f"{name:22s}: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## 2. Teste de Wilcoxon — Comparação de 2 Modelos

Teste não-paramétrico para comparar dois modelos com os mesmos folds de CV:

```python
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

skf = StratifiedKFold(n_splits=30, shuffle=True, random_state=42)

pipe_lr = Pipeline([('sc', StandardScaler()), ('lr', LogisticRegression(max_iter=1000))])
pipe_rf = Pipeline([('sc', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=100, random_state=42))])

scores_lr, scores_rf = [], []
for tr_idx, te_idx in skf.split(X, y):
    X_tr, X_te = X[tr_idx], X[te_idx]
    y_tr, y_te = y[tr_idx], y[te_idx]
    
    pipe_lr.fit(X_tr, y_tr)
    scores_lr.append(f1_score(y_te, pipe_lr.predict(X_te)))
    
    pipe_rf.fit(X_tr, y_tr)
    scores_rf.append(f1_score(y_te, pipe_rf.predict(X_te)))

stat, pvalue = wilcoxon(scores_lr, scores_rf)
print(f"Wilcoxon — statistic={stat:.2f}, p-value={pvalue:.4f}")
print("Diferença estatisticamente significativa!" if pvalue < 0.05 else "Sem diferença significativa.")
```

---

## 3. Teste de Friedman — Comparação de N Modelos

```python
from scipy.stats import friedmanchisquare

# Simular scores de 4 modelos em 10 folds
np.random.seed(42)
m1 = np.random.normal(0.85, 0.03, 10)
m2 = np.random.normal(0.87, 0.03, 10)
m3 = np.random.normal(0.84, 0.04, 10)
m4 = np.random.normal(0.88, 0.02, 10)

stat, pvalue = friedmanchisquare(m1, m2, m3, m4)
print(f"Friedman — statistic={stat:.2f}, p-value={pvalue:.4f}")
# Se p < 0.05: há diferença entre pelo menos um par
```

---

## 4. Erros Comuns na Comparação

1. **Comparar no conjunto de teste múltiplas vezes** → overfitting no teste
2. **Ignorar variância** → uma diferença de 0.001 pode não ser significativa
3. **Não usar o mesmo conjunto de CV** → comparação injusta
4. **Usar acurácia com dados desbalanceados** → métrica enganosa
5. **Selecionar o melhor modelo e reportar o score de CV** → otimismo excessivo

---

## Questões para Reflexão
1. O que significa "p-value < 0.05" no contexto de comparação de modelos?
2. Por que é problemático comparar modelos usando conjuntos de CV diferentes?
3. Como você reportaria resultados de comparação de modelos num artigo científico?

## Referências
- Faceli et al., cap. 4 (Avaliação de Modelos)
- Géron, cap. 2

---
*Módulo 08 concluído! Próximo → [Módulo 09: Overfitting e Regularização](../09-overfitting-regularizacao/README.md)*
