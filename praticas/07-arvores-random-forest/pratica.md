# Prática 07 — Árvores de Decisão e Random Forest

**Módulo:** 06 | **Duração:** ~90 minutos | **Dataset:** Titanic

## Objetivos
- Treinar e visualizar árvores de decisão
- Controlar overfitting com poda
- Implementar Random Forest e analisar importância de features

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.inspection import permutation_importance

# Preparar Titanic
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)

df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna('S', inplace=True)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
df['Sex_enc'] = LabelEncoder().fit_transform(df['Sex'])
df['Embarked_enc'] = LabelEncoder().fit_transform(df['Embarked'])

features = ['Pclass', 'Sex_enc', 'Age', 'FamilySize', 'IsAlone', 'Fare', 'Embarked_enc']
X = df[features]
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Árvore sem restrições
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)
print(f"Árvore full — treino: {dt_full.score(X_train, y_train):.4f} | teste: {dt_full.score(X_test, y_test):.4f}")

# Árvore podada
dt_pruned = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt_pruned.fit(X_train, y_train)
print(f"Árvore podada — treino: {dt_pruned.score(X_train, y_train):.4f} | teste: {dt_pruned.score(X_test, y_test):.4f}")

# Visualizar
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(dt_pruned, feature_names=features, class_names=['Morreu', 'Sobreviveu'],
          filled=True, rounded=True, ax=ax)
plt.title("Árvore de Decisão — Titanic (max_depth=5)")
plt.tight_layout(); plt.show()

# Random Forest
rf = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(f"\nRandom Forest:")
print(f"  OOB score: {rf.oob_score_:.4f}")
print(f"  Teste:     {rf.score(X_test, y_test):.4f}")
print(f"  AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
print(classification_report(y_test, y_pred, target_names=['Morreu', 'Sobreviveu']))

# Importância de features
feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
feat_imp.plot(kind='barh')
plt.title('Importância das Features (Random Forest)')
plt.tight_layout(); plt.show()
```

## Desafios Extras
1. Use `cost_complexity_pruning_path` para encontrar o alpha ótimo de poda por minimal cost-complexity
2. Teste `ExtraTreesClassifier` e compare com Random Forest
3. Use `ConfusionMatrixDisplay` para visualizar a matriz de confusão
