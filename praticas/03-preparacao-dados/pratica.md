# Prática 03 — Pipeline de Preparação de Dados

**Módulo:** 03 | **Duração:** ~90 minutos | **Dataset:** Adult Income

## Objetivos
- Construir um pipeline completo de pré-processamento com Scikit-learn
- Tratar dados ausentes, codificar categóricas e normalizar numéricas
- Avaliar impacto do pré-processamento na performance do modelo

```python
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Adult Income dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
cols = ['age','workclass','fnlwgt','education','education-num','marital-status',
        'occupation','relationship','race','sex','capital-gain','capital-loss',
        'hours-per-week','native-country','income']
df = pd.read_csv(url, names=cols, sep=', ', engine='python', na_values='?')

print(f"Shape: {df.shape}")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Target
y = (df['income'] == '>50K').astype(int)
X = df.drop('income', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Features
num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_features = ['workclass', 'education', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'native-country']

# Pipeline numérico
num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Pipeline categórico
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer
preprocessor = ColumnTransformer([
    ('num', num_pipe, num_features),
    ('cat', cat_pipe, cat_features)
])

# Pipeline completo
full_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticRegression(max_iter=1000, C=1.0))
])

full_pipe.fit(X_train, y_train)
y_pred = full_pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))

# Comparar imputação: median vs KNN
from sklearn.pipeline import Pipeline as pipe

scores_median = cross_val_score(full_pipe, X_train, y_train, cv=5, scoring='f1')
print(f"Median imputer — F1: {scores_median.mean():.4f} ± {scores_median.std():.4f}")
```

## Desafios Extras
1. Experimente `TargetEncoder` em vez de `OneHotEncoder` para as categóricas
2. Adicione uma etapa de seleção de features com `SelectKBest` ao pipeline
3. Compare o impacto de StandardScaler vs RobustScaler vs MinMaxScaler no modelo final
