# Aula 09 — Ecossistema de Ferramentas de ML

> **Módulo 02 · Aula 09 · Carga horária: 1 hora-aula**

---

## Objetivos da Aula

- Conhecer as principais bibliotecas e frameworks do ecossistema Python para ML.
- Entender a API unificada do scikit-learn e seu modelo de Pipelines.
- Compreender as diferenças entre TensorFlow/Keras e PyTorch.
- Explorar ferramentas de MLOps, visualização e computação em nuvem.
- Saber escolher a ferramenta certa para cada tipo de problema.

---

## 1. Visão Geral do Ecossistema Python para ML

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ECOSSISTEMA PYTHON DE ML                        │
├────────────────────┬────────────────────┬───────────────────────────┤
│   DADOS & EDA      │  ML CLÁSSICO       │  DEEP LEARNING            │
│                    │                    │                           │
│  NumPy             │  Scikit-learn      │  TensorFlow / Keras       │
│  Pandas            │  XGBoost           │  PyTorch                  │
│  Matplotlib        │  LightGBM          │  JAX                      │
│  Seaborn           │  CatBoost          │  Hugging Face             │
│  Plotly            │  Statsmodels       │  FastAI                   │
├────────────────────┼────────────────────┼───────────────────────────┤
│   MLOPS            │  DADOS EM ESCALA   │  CLOUD / INFRA            │
│                    │                    │                           │
│  MLflow            │  Spark (PySpark)   │  AWS SageMaker            │
│  DVC               │  Dask              │  Google Vertex AI         │
│  Weights & Biases  │  Ray               │  Azure ML                 │
│  BentoML           │  Polars            │  Google Colab             │
│  Seldon Core       │  Vaex              │  Kaggle Kernels           │
└────────────────────┴────────────────────┴───────────────────────────┘
```

---

## 2. NumPy e Pandas: A Base de Tudo

### 2.1 NumPy

NumPy é a fundação do ecossistema científico Python. Oferece arrays N-dimensionais eficientes e operações vetorizadas.

```python
import numpy as np

# Arrays e operações básicas
a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
b = np.linspace(0, 1, 100)
M = np.random.randn(3, 4)    # Matriz 3×4 aleatória

# Operações vetorizadas (sem loops Python)
print(np.sqrt(a))                   # raiz quadrada elemento a elemento
print(M.T)                          # transposta
print(M @ M.T)                      # produto matricial (3×4 @ 4×3 = 3×3)
print(np.linalg.norm(a))            # norma L2
print(np.dot(a[:3], a[2:]))         # produto escalar

# Broadcasting
X = np.random.randn(100, 10)
media = X.mean(axis=0)              # média de cada coluna
dp    = X.std(axis=0)
X_norm = (X - media) / dp          # normalização via broadcasting
```

### 2.2 Pandas

Pandas oferece estruturas de dados tabulares (DataFrame/Series) e operações de análise de dados.

```python
import pandas as pd

# Criação e leitura
df = pd.read_csv("clientes.csv")
df = pd.read_excel("vendas.xlsx", sheet_name="2023")
df = pd.read_json("api_response.json")

# Operações essenciais
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Seleção e filtragem
ativos = df[df['status'] == 'ativo']
colunas = df[['nome', 'valor', 'data']]
df_novo = df.query("valor > 1000 and regiao == 'Sul'")

# Agrupamento e agregação
resumo = df.groupby('regiao').agg(
    total_vendas=('valor', 'sum'),
    n_clientes=('id', 'nunique'),
    ticket_medio=('valor', 'mean')
).reset_index()

# Merge/Join
df_merged = pd.merge(df_clientes, df_pedidos, on='cliente_id', how='left')

# Apply com função customizada
df['categoria_valor'] = df['valor'].apply(
    lambda x: 'alto' if x > 5000 else ('médio' if x > 1000 else 'baixo')
)
```

---

## 3. Scikit-learn: O Framework de ML Clássico

### 3.1 Design Philosophy: A API Unificada

Scikit-learn define uma API consistente baseada em três interfaces:

| Interface | Métodos | Exemplos |
|-----------|---------|---------|
| **Estimator** | `fit(X, y)` | Todos os modelos e transformadores |
| **Predictor** | `predict(X)`, `predict_proba(X)`, `score(X, y)` | Classificadores, Regressores |
| **Transformer** | `transform(X)`, `fit_transform(X)` | Scalers, Encoders, Selectors |

```python
from sklearn.base import BaseEstimator, TransformerMixin

# Qualquer estimador/transformador segue este padrão
class MeuTransformador(BaseEstimator, TransformerMixin):
    def __init__(self, fator=1.0):
        self.fator = fator        # hiperparâmetros no __init__

    def fit(self, X, y=None):
        self.media_ = X.mean()    # parâmetros aprendidos com underscore
        return self               # SEMPRE retornar self

    def transform(self, X):
        return (X - self.media_) * self.fator
```

### 3.2 Model Selection: Divisão, Validação Cruzada

```python
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import GradientBoostingClassifier

# Divisão básica
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(GradientBoostingClassifier(), X_train, y_train,
                         cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# GridSearchCV — busca exaustiva
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth':    [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}
grid_search = GridSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
)
grid_search.fit(X_train, y_train)
print(f"Melhores parâmetros: {grid_search.best_params_}")
print(f"Melhor AUC: {grid_search.best_score_:.4f}")

# RandomizedSearchCV — mais eficiente para espaços grandes
from scipy.stats import randint, uniform
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth':    randint(2, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample':    uniform(0.5, 0.5)
}
rand_search = RandomizedSearchCV(
    GradientBoostingClassifier(random_state=42),
    param_dist, n_iter=50, cv=cv, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
rand_search.fit(X_train, y_train)
print(f"Melhor AUC (Random): {rand_search.best_score_:.4f}")
```

### 3.3 Pipeline: Encadeamento de Transformações

```python
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Features por tipo
features_num = ['idade', 'salario', 'tempo_emprego']
features_cat = ['estado_civil', 'nivel_educacao', 'regiao']

# Pré-processamento diferenciado por tipo de feature
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ]), features_num),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), features_cat)
])

# Pipeline completo: pré-processamento + modelo
pipeline_completo = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier',   RandomForestClassifier(n_estimators=200, random_state=42))
])

# Treinar: aplica todas as transformações + treina o modelo
pipeline_completo.fit(X_train, y_train)

# Prever: aplica as mesmas transformações + faz previsão
y_pred = pipeline_completo.predict(X_test)

# O pipeline pode ser usado com GridSearchCV
param_grid_pipeline = {
    'preprocessor__num__imputer__strategy': ['median', 'mean'],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, None]
}
```

---

## 4. TensorFlow / Keras

### 4.1 Visão Geral

- **TensorFlow** (Google, 2015): framework para computação diferenciável em grafos.
- **Keras** (integrado ao TF 2.x): API de alto nível para construir e treinar redes neurais.
- **TF 2.x**: modo eager por padrão (execução imediata, mais pythônico).
- Ideal para: produção em escala, mobile (TFLite), web (TF.js), edge (TF Micro).

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import numpy as np

print(f"TensorFlow versão: {tf.__version__}")
print(f"GPUs disponíveis: {tf.config.list_physical_devices('GPU')}")

# -------------------------------------------------------
# EXEMPLO 1: Classificação com a API Sequential
# -------------------------------------------------------
def criar_modelo_sequencial(n_features, n_classes):
    modelo = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax')
    ], name="classificador")
    return modelo

modelo = criar_modelo_sequencial(n_features=20, n_classes=3)
modelo.summary()

# Compilar
modelo.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks úteis
callbacks_lista = [
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    callbacks.ModelCheckpoint('melhor_modelo.keras', save_best_only=True)
]

# Treinar
historico = modelo.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=callbacks_lista,
    verbose=1
)

# -------------------------------------------------------
# EXEMPLO 2: API Funcional (para arquiteturas complexas)
# -------------------------------------------------------
# Útil para: múltiplas entradas/saídas, conexões residuais

entradas = keras.Input(shape=(20,), name="features")
x = layers.Dense(128, activation='relu')(entradas)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)

# Bloco residual
residual = layers.Dense(64)(entradas)   # skip connection
x = layers.Add()([x, residual])
x = layers.Activation('relu')(x)

# Múltiplas saídas
saida_classe  = layers.Dense(3, activation='softmax', name="classe")(x)
saida_score   = layers.Dense(1, activation='sigmoid', name="score")(x)

modelo_funcional = keras.Model(inputs=entradas, outputs=[saida_classe, saida_score])
modelo_funcional.compile(
    optimizer='adam',
    loss={'classe': 'sparse_categorical_crossentropy', 'score': 'binary_crossentropy'},
    metrics={'classe': 'accuracy', 'score': 'AUC'}
)

# -------------------------------------------------------
# EXEMPLO 3: Loop de treinamento customizado (tf.GradientTape)
# -------------------------------------------------------
optimizer = optimizers.Adam(learning_rate=1e-3)
loss_fn   = keras.losses.SparseCategoricalCrossentropy()

@tf.function  # compila para grafo → mais rápido
def passo_treino(X_batch, y_batch):
    with tf.GradientTape() as tape:
        predicoes = modelo(X_batch, training=True)
        loss = loss_fn(y_batch, predicoes)

    gradientes = tape.gradient(loss, modelo.trainable_variables)
    optimizer.apply_gradients(zip(gradientes, modelo.trainable_variables))
    return loss
```

---

## 5. PyTorch

### 5.1 Visão Geral

- **PyTorch** (Meta/Facebook, 2016): grafos dinâmicos, execução eager nativa.
- Preferido pela comunidade de pesquisa (mais flexível para arquiteturas experimentais).
- `autograd`: diferenciação automática via `requires_grad=True`.
- **TorchScript**: compilação para produção.
- Ideal para: pesquisa, NLP (com HuggingFace), computer vision, arquiteturas customizadas.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

print(f"PyTorch versão: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo: {device}")

# -------------------------------------------------------
# EXEMPLO 1: Rede Neural com nn.Module
# -------------------------------------------------------
class RedeClassificacao(nn.Module):
    def __init__(self, n_features, n_classes, hidden_dims=[256, 128, 64]):
        super().__init__()

        camadas = []
        dim_entrada = n_features
        for dim_saida in hidden_dims:
            camadas.extend([
                nn.Linear(dim_entrada, dim_saida),
                nn.BatchNorm1d(dim_saida),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            dim_entrada = dim_saida
        camadas.append(nn.Linear(dim_entrada, n_classes))

        self.rede = nn.Sequential(*camadas)

    def forward(self, x):
        return self.rede(x)

modelo = RedeClassificacao(n_features=20, n_classes=3).to(device)
print(modelo)
print(f"Parâmetros treináveis: {sum(p.numel() for p in modelo.parameters() if p.requires_grad):,}")

# -------------------------------------------------------
# EXEMPLO 2: Dataset e DataLoader customizados
# -------------------------------------------------------
class MeuDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_sample, y_sample = self.X[idx], self.y[idx]
        if self.transform:
            X_sample = self.transform(X_sample)
        return X_sample, y_sample

train_dataset = MeuDataset(X_train_np, y_train_np)
test_dataset  = MeuDataset(X_test_np, y_test_np)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False)

# -------------------------------------------------------
# EXEMPLO 3: Loop de Treinamento
# -------------------------------------------------------
criterio  = nn.CrossEntropyLoss()
optimizer = optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

def treinar_epoca(modelo, loader, criterio, optimizer):
    modelo.train()
    loss_total, correto, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = modelo(X_batch)
        loss   = criterio(logits, y_batch)
        loss.backward()               # backpropagation
        optimizer.step()
        loss_total += loss.item() * len(y_batch)
        correto    += (logits.argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return loss_total / total, correto / total

def avaliar(modelo, loader, criterio):
    modelo.eval()
    loss_total, correto, total = 0.0, 0, 0
    with torch.no_grad():             # desabilitar gradientes na avaliação
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits  = modelo(X_batch)
            loss    = criterio(logits, y_batch)
            loss_total += loss.item() * len(y_batch)
            correto    += (logits.argmax(1) == y_batch).sum().item()
            total      += len(y_batch)
    return loss_total / total, correto / total

melhor_acc_val = 0.0
for epoca in range(1, 51):
    loss_tr, acc_tr = treinar_epoca(modelo, train_loader, criterio, optimizer)
    loss_val, acc_val = avaliar(modelo, test_loader, criterio)
    scheduler.step()

    if acc_val > melhor_acc_val:
        melhor_acc_val = acc_val
        torch.save(modelo.state_dict(), 'melhor_modelo.pt')

    if epoca % 10 == 0:
        lr_atual = optimizer.param_groups[0]['lr']
        print(f"Época {epoca:3d} | LR: {lr_atual:.6f} | "
              f"Loss (tr/val): {loss_tr:.4f}/{loss_val:.4f} | "
              f"Acc (tr/val): {acc_tr:.4f}/{acc_val:.4f}")

print(f"\nMelhor acurácia de validação: {melhor_acc_val:.4f}")

# Carregar melhor modelo
modelo.load_state_dict(torch.load('melhor_modelo.pt'))
modelo.eval()
```

---

## 6. Hugging Face

### 6.1 Visão Geral

Hugging Face é o hub central da comunidade de NLP e visão computacional. Oferece:

- **Transformers**: 150.000+ modelos pré-treinados (BERT, GPT, T5, LLaMA, etc.).
- **Datasets**: 50.000+ datasets prontos para uso.
- **Hub**: repositório colaborativo de modelos, datasets e Spaces (demos).
- **PEFT**: fine-tuning eficiente (LoRA, QLoRA).
- **Accelerate**: treinamento distribuído simplificado.

```python
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, TrainingArguments, Trainer
)
from datasets import load_dataset
import torch

# -------------------------------------------------------
# EXEMPLO 1: Pipeline de NLP pronto para uso
# -------------------------------------------------------
# Análise de sentimento em português
analisador_sentimento = pipeline(
    "text-classification",
    model="neuralmind/bert-base-portuguese-cased",
    device=0 if torch.cuda.is_available() else -1
)

textos = [
    "O produto chegou no prazo e a qualidade é excelente!",
    "Péssimo atendimento, nunca mais compro aqui.",
    "O serviço é razoável, mas poderia melhorar."
]
resultados = analisador_sentimento(textos)
for texto, resultado in zip(textos, resultados):
    print(f"Texto: {texto[:50]}...")
    print(f"  Label: {resultado['label']} | Score: {resultado['score']:.4f}\n")

# -------------------------------------------------------
# EXEMPLO 2: Tokenização e embeddings com BERT
# -------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

# Tokenizar um par de sentenças
codificacao = tokenizer(
    "Aprendizado de máquina é fascinante.",
    "Redes neurais imitam o cérebro humano.",
    padding=True, truncation=True, return_tensors="pt",
    max_length=128
)
print("IDs dos tokens:", codificacao['input_ids'].shape)
print("Tokens:", tokenizer.convert_ids_to_tokens(codificacao['input_ids'][0]))

# -------------------------------------------------------
# EXEMPLO 3: Fine-tuning com Trainer API
# -------------------------------------------------------
modelo_ft = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-multilingual-cased", num_labels=2
)

training_args = TrainingArguments(
    output_dir="./resultados_bert",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# trainer = Trainer(
#     model=modelo_ft,
#     args=training_args,
#     train_dataset=train_dataset_hf,
#     eval_dataset=eval_dataset_hf,
#     compute_metrics=compute_metrics_fn
# )
# trainer.train()
```

---

## 7. Gradient Boosting: XGBoost, LightGBM e CatBoost

### 7.1 Comparativo

| Framework | Desenvolvedor | Destaques |
|-----------|--------------|-----------|
| **XGBoost** | DMLC | Regularização L1/L2, tratamento de NaN, alta performance |
| **LightGBM** | Microsoft | Histogram-based, muito rápido, eficiente em memória |
| **CatBoost** | Yandex | Nativo para categorias, sem encoding manual, robusto |

### 7.2 Regra prática

- **LightGBM**: Melhor escolha para datasets grandes (>100k linhas) — mais rápido.
- **CatBoost**: Dataset com muitas variáveis categóricas — menos pré-processamento.
- **XGBoost**: Dataset médio, boa interpretabilidade via SHAP, amplamente suportado.

```python
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import roc_auc_score
import time

print(f"XGBoost:  {xgb.__version__}")
print(f"LightGBM: {lgb.__version__}")
print(f"CatBoost: {cb.__version__}")

# --- XGBoost ---
t0 = time.time()
xgb_modelo = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,      # L1
    reg_lambda=1.0,     # L2
    use_label_encoder=False,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1
)
xgb_modelo.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
auc_xgb = roc_auc_score(y_test, xgb_modelo.predict_proba(X_test)[:, 1])
print(f"XGBoost  — AUC: {auc_xgb:.4f} | Tempo: {time.time()-t0:.1f}s")

# --- LightGBM ---
t0 = time.time()
lgb_modelo = lgb.LGBMClassifier(
    n_estimators=500,
    num_leaves=63,
    learning_rate=0.05,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_modelo.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(20, verbose=False)]
)
auc_lgb = roc_auc_score(y_test, lgb_modelo.predict_proba(X_test)[:, 1])
print(f"LightGBM — AUC: {auc_lgb:.4f} | Tempo: {time.time()-t0:.1f}s")

# --- CatBoost ---
t0 = time.time()
cat_modelo = cb.CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=0
)
cat_modelo.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=20,
    use_best_model=True
)
auc_cat = roc_auc_score(y_test, cat_modelo.predict_proba(X_test)[:, 1])
print(f"CatBoost — AUC: {auc_cat:.4f} | Tempo: {time.time()-t0:.1f}s")
```

---

## 8. MLflow: Rastreamento de Experimentos

```python
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature

# Iniciar servidor MLflow: mlflow ui --port 5000
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("comparacao-modelos-churn")

# Registrar experimento automaticamente
with mlflow.start_run(run_name="xgboost_tuned") as run:
    # Hiperparâmetros
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8
    }
    mlflow.log_params(params)

    # Treinar
    modelo = xgb.XGBClassifier(**params, random_state=42)
    modelo.fit(X_train, y_train)

    # Métricas
    y_pred   = modelo.predict(X_test)
    y_proba  = modelo.predict_proba(X_test)[:, 1]
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # Registrar modelo
    signature = infer_signature(X_train, modelo.predict(X_train))
    mlflow.xgboost.log_model(
        modelo, "model",
        signature=signature,
        registered_model_name="ChurnXGBoost"
    )

    # Artefatos (gráficos, relatórios)
    # mlflow.log_artifact("churn_eda.png")
    # mlflow.log_artifact("feature_importance.png")

    run_id = run.info.run_id
    print(f"Run ID: {run_id}")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Carregar modelo registrado
modelo_carregado = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
```

---

## 9. Visualização: Matplotlib, Seaborn e Plotly

```python
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --------------------------------------------------------
# MATPLOTLIB: controle total sobre o gráfico
# --------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Visualizações com Matplotlib', fontsize=14)

# Histograma com KDE
np.random.seed(42)
dados = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(5, 1.5, 300)])
axes[0,0].hist(dados, bins=40, density=True, alpha=0.6, color='steelblue', edgecolor='white')
from scipy import stats
kde = stats.gaussian_kde(dados)
x_grid = np.linspace(dados.min(), dados.max(), 200)
axes[0,0].plot(x_grid, kde(x_grid), 'r-', lw=2)
axes[0,0].set(title='Histograma + KDE', xlabel='Valor', ylabel='Densidade')

# Scatter com regressão
x = np.random.randn(200)
y = 2*x + np.random.randn(200) * 0.8
axes[0,1].scatter(x, y, alpha=0.4, s=30, color='steelblue')
m, b = np.polyfit(x, y, 1)
axes[0,1].plot(np.sort(x), m*np.sort(x)+b, 'r-', lw=2)
axes[0,1].set(title='Scatter + Regressão', xlabel='X', ylabel='Y')

# Boxplot
grupos = {'Grupo A': np.random.normal(70, 10, 100),
          'Grupo B': np.random.normal(75, 15, 100),
          'Grupo C': np.random.normal(65, 8, 100)}
axes[1,0].boxplot(grupos.values(), tick_labels=grupos.keys(), notch=True)
axes[1,0].set(title='Boxplot por Grupo', ylabel='Valor')

# Heatmap de correlação (Seaborn no subplot)
df_corr = pd.DataFrame(np.random.randn(100, 5), columns=['A','B','C','D','E'])
df_corr['B'] = 0.7*df_corr['A'] + 0.3*np.random.randn(100)  # correlação artificial
sns.heatmap(df_corr.corr(), ax=axes[1,1], annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-1, vmax=1, linewidths=0.5)
axes[1,1].set_title('Heatmap de Correlação (Seaborn)')

plt.tight_layout()
plt.savefig("visualizacoes_matplotlib_seaborn.png", dpi=150, bbox_inches='tight')
plt.show()

# --------------------------------------------------------
# PLOTLY: gráficos interativos
# --------------------------------------------------------
import pandas as pd
df_plot = pd.DataFrame({'x': x, 'y': y, 'grupo': np.random.choice(['A','B','C'], 200)})

fig_interativo = px.scatter(
    df_plot, x='x', y='y', color='grupo',
    title='Gráfico Interativo com Plotly Express',
    labels={'x': 'Variável X', 'y': 'Variável Y'},
    hover_data=['grupo']
)
fig_interativo.update_layout(template='plotly_white')
fig_interativo.write_html("scatter_interativo.html")
print("Gráfico interativo salvo: scatter_interativo.html")
```

---

## 10. Plataformas de Nuvem para ML

### 10.1 Google Colab (gratuito)
- Notebooks Jupyter com GPU/TPU gratuitos (T4, A100 no Colab Pro).
- Integração com Google Drive.
- Ideal para: aprendizado, prototipagem, competições Kaggle.
- Limitações: sessão desconecta após inatividade (~90min), sem persistência.

### 10.2 Kaggle Kernels (gratuito)
- 30h/semana de GPU gratuita (T4 × 2 ou P100).
- Integração direta com datasets do Kaggle.
- Ideal para: competições, exploração de datasets públicos.

### 10.3 AWS SageMaker
- Plataforma end-to-end: notebooks, treinamento distribuído, deploy.
- `SageMaker Studio`: IDE integrado para ML.
- `SageMaker Autopilot`: AutoML automatizado.
- `SageMaker Endpoints`: deploy de modelos com auto-scaling.

### 10.4 Google Vertex AI
- Successor do AI Platform.
- Treinamento com GPUs/TPUs, `Vertex AI Pipelines` (Kubeflow/TFX).
- `Model Registry`, `Feature Store`, `Explainable AI`.

### 10.5 Azure Machine Learning
- `Azure ML Studio`: interface low-code/no-code.
- `Automated ML`: AutoML integrado.
- Integração com MLflow nativa.
- Boa escolha para empresas no ecossistema Microsoft.

---

## 11. Guia de Escolha da Ferramenta

```
Qual é o seu problema?
│
├── Dados tabulares (CSV, SQL, DataFrame)
│   ├── Modelagem clássica? → scikit-learn, XGBoost/LightGBM/CatBoost
│   ├── Muitas categorias?  → CatBoost (menos pré-processamento)
│   └── Dataset grande (>1M linhas)? → LightGBM (mais rápido)
│
├── NLP (texto)
│   ├── Classificação/NER/QA? → HuggingFace Transformers + PyTorch
│   ├── Geração de texto?      → HuggingFace + PEFT/LoRA para fine-tuning
│   └── Análise rápida?        → spaCy, NLTK + scikit-learn (TF-IDF + SVM)
│
├── Visão Computacional (imagens)
│   ├── Transfer learning rápido? → TensorFlow/Keras (ResNet, EfficientNet)
│   ├── Pesquisa / custom arch?   → PyTorch + torchvision
│   └── Detecção de objetos?      → YOLOv8/ultralytics, Detectron2
│
├── Séries Temporais
│   ├── Estatístico?   → statsmodels (ARIMA, SARIMA, Prophet)
│   ├── ML clássico?   → scikit-learn (features de lag + XGBoost)
│   └── Deep Learning? → PyTorch (LSTM, Transformer) / Darts
│
└── Áudio
    ├── Speech-to-text?  → HuggingFace Whisper
    ├── Classificação?   → torchaudio + HuggingFace Audio
    └── Geração musical? → Magenta, AudioCraft (Meta)

Para DEPLOY:
├── API REST simples?     → FastAPI + scikit-learn/XGBoost + Docker
├── TF Serving?          → TensorFlow Serving
├── Produção em escala?  → BentoML, Triton Inference Server
└── Serverless?          → AWS Lambda + SageMaker, Google Cloud Run
```

---

## 12. Exercícios

1. **Scikit-learn**: Crie um Pipeline com `ColumnTransformer` para o dataset de churn da Aula 08. Use `OneHotEncoder` para features categóricas e `StandardScaler` + `KNNImputer` para numéricas. Combine com `GradientBoostingClassifier` e avalie com `cross_val_score`.

2. **PyTorch vs Keras**: Implemente uma rede neural simples de 3 camadas para o dataset Iris tanto em Keras quanto em PyTorch. Compare a verbosidade do código e os resultados.

3. **Gradient Boosting**: Compare XGBoost, LightGBM e CatBoost no dataset de churn. Meça o tempo de treinamento e o AUC-ROC. Qual framework performou melhor?

4. **MLflow**: Configure um servidor MLflow local (`mlflow ui`) e registre pelo menos 5 experimentos com diferentes configurações. Use a interface web para comparar os runs.

5. **Pesquisa**: Explore o Hugging Face Hub (huggingface.co/models). Encontre 3 modelos para NLP em português e descreva: (a) o nome do modelo, (b) a tarefa, (c) o número de parâmetros e (d) como utilizá-lo com `pipeline()`.

---

## 13. Referências

- GÉRON, Aurélien. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow*. 3. ed. Alta Books, 2023. Caps. 1–3, 10–12.
- Scikit-learn Documentation. *User Guide*. [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- PyTorch Documentation. [https://pytorch.org/docs/stable/](https://pytorch.org/docs/stable/)
- TensorFlow Documentation. [https://www.tensorflow.org/guide](https://www.tensorflow.org/guide)
- Hugging Face Documentation. [https://huggingface.co/docs](https://huggingface.co/docs)
- XGBoost Documentation. [https://xgboost.readthedocs.io](https://xgboost.readthedocs.io)
- LightGBM Documentation. [https://lightgbm.readthedocs.io](https://lightgbm.readthedocs.io)
- MLflow Documentation. [https://mlflow.org/docs/latest/](https://mlflow.org/docs/latest/)

---

*← [Aula 08 — Fluxo de um Projeto de ML](aula-08-fluxo-projeto-ml.md) | Próximo Módulo: [Módulo 03 — Preparação e Análise de Dados](../03-preparacao-dados/README.md) →*
