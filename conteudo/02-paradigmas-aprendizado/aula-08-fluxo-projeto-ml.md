# Aula 08 — Fluxo de um Projeto de Machine Learning

> **Módulo 02 · Aula 08 · Carga horária: 1 hora-aula**

---

## Objetivos da Aula

- Compreender as etapas do ciclo de vida de um projeto de ML com CRISP-DM.
- Diferenciar métricas de negócio de métricas de ML.
- Reconhecer as armadilhas comuns em projetos de ML.
- Aplicar boas práticas de versionamento e rastreabilidade.
- Implementar um pipeline end-to-end de predição de churn em Python.

---

## 1. Por Que Metodologia Importa?

Projetos de ML falham por razões diversas:

- **73%** dos projetos de ML nunca chegam à produção (Gartner, 2020).
- Foco excessivo em modelagem, ignorando entendimento do negócio.
- Dados mal preparados → modelos ruins (garbage in, garbage out).
- Falta de monitoramento pós-deploy.
- Ausência de rastreabilidade de experimentos.

Uma metodologia estruturada reduz drasticamente essas falhas.

---

## 2. Metodologia CRISP-DM

**CRISP-DM** (Cross-Industry Standard Process for Data Mining) é o framework mais utilizado para projetos de ML/Data Mining na indústria. Foi desenvolvido em 1996 por um consórcio de empresas (SPSS, NCR, Daimler-Benz, etc.).

```
                    ┌─────────────────────────────────────┐
                    │          CRISP-DM                   │
                    │                                     │
                    │   ┌─────────────────────────┐       │
                    │   │   1. Entendimento        │       │
                    │   │      do Negócio          │       │
                    │   └────────────┬────────────┘       │
                    │                │                     │
                    │   ┌────────────▼────────────┐       │
                    │   │   2. Entendimento        │       │
                    │   │      dos Dados           │       │
                    │   └────────────┬────────────┘       │
                    │                │                     │
                    │   ┌────────────▼────────────┐       │
                    │   │   3. Preparação          │       │
            ◄───────┤   │      dos Dados           ├───►   │
                    │   └────────────┬────────────┘       │
                    │                │                     │
                    │   ┌────────────▼────────────┐       │
                    │   │   4. Modelagem           │       │
                    │   └────────────┬────────────┘       │
                    │                │                     │
                    │   ┌────────────▼────────────┐       │
                    │   │   5. Avaliação           │       │
                    │   └────────────┬────────────┘       │
                    │                │                     │
                    │   ┌────────────▼────────────┐       │
                    │   │   6. Implantação         │       │
                    │   │      (Deploy)            │       │
                    │   └─────────────────────────┘       │
                    │                                     │
                    │  ← Processo iterativo e cíclico →   │
                    └─────────────────────────────────────┘
```

### 2.1 Fase 1: Entendimento do Negócio (Business Understanding)

**Objetivo**: Traduzir o problema de negócio em um problema de ML bem definido.

**Atividades:**
- Reuniões com stakeholders: O que precisam prever/decidir?
- Definir o **objetivo de negócio** (ex.: "reduzir churn em 15%").
- Traduzir para **objetivo de mineração de dados** (ex.: "classificar clientes com alta probabilidade de cancelamento nos próximos 30 dias").
- Identificar **restrições**: custo, tempo, privacidade, regulatório.
- Definir critérios de sucesso do projeto.

**Perguntas-chave:**
- Qual decisão será tomada com base no modelo?
- Quais são os custos de falso positivo e falso negativo?
- O modelo substitui ou auxilia uma decisão humana?
- Qual é a frequência de inferência? (tempo real vs. batch)

### 2.2 Fase 2: Entendimento dos Dados (Data Understanding)

**Objetivo**: Conhecer profundamente os dados disponíveis.

**Atividades:**
- Inventário de fontes de dados (CRM, logs, APIs, etc.).
- Análise exploratória (EDA): distribuições, correlações, outliers.
- Identificar problemas: valores ausentes, desbalanceamento, inconsistências.
- Avaliar qualidade e suficiência dos dados.

**Artefato:** Relatório de EDA com visualizações e insights.

### 2.3 Fase 3: Preparação dos Dados (Data Preparation)

**Objetivo**: Transformar dados brutos em um dataset adequado para modelagem.

> **"80% do tempo de um projeto de Data Science é gasto aqui."** (regra empírica amplamente citada)

**Atividades:**
- Limpeza: tratar valores ausentes, outliers, duplicatas.
- Integração: unir dados de múltiplas fontes.
- Transformação: encoding, normalização, feature engineering.
- Seleção: escolher as features mais relevantes.
- Formatação: estruturar no formato esperado pelo algoritmo.

### 2.4 Fase 4: Modelagem (Modeling)

**Objetivo**: Selecionar, treinar e ajustar algoritmos de ML.

**Atividades:**
- Selecionar técnicas de modelagem (baseline simples primeiro!).
- Design do experimento: como dividir os dados, que métricas usar.
- Construir e treinar modelos.
- Ajuste de hiperparâmetros (GridSearch, RandomSearch, Optuna).
- Iteração: múltiplos modelos, ensembles.

**Regra de ouro:** Comece sempre com um baseline simples (média, regressão logística, árvore rasa). Modelos complexos só valem a pena se melhorarem significativamente o baseline.

### 2.5 Fase 5: Avaliação (Evaluation)

**Objetivo**: Verificar se o modelo atende aos critérios de negócio.

**Atividades:**
- Avaliação técnica: métricas de ML (acurácia, F1, RMSE, AUC-ROC).
- Avaliação de negócio: traduzir métricas técnicas em impacto financeiro.
- Revisão do processo: algo foi negligenciado?
- Decisão: ir para produção, iterar ou cancelar.

**Métricas de negócio vs. ML:**

| Problema | Métrica de ML | Métrica de Negócio |
|---------|--------------|-------------------|
| Churn   | AUC-ROC, F1  | Receita retida, custo de retenção |
| Crédito | AUC-PR       | Perda por inadimplência, custo de análise |
| Fraude  | Recall, precisão | Prejuízo evitado vs. custo de investigação |
| Recomendação | NDCG, MAP | Receita adicional, engajamento |

### 2.6 Fase 6: Implantação (Deployment)

**Objetivo**: Disponibilizar o modelo para uso em produção.

**Modalidades de deploy:**
- **Batch scoring**: rodar o modelo periodicamente (ex.: diariamente).
- **REST API**: modelo exposto via endpoint HTTP (Flask, FastAPI, TorchServe).
- **Edge deployment**: modelo embarcado no dispositivo (mobile, IoT).
- **Streaming**: inferência em tempo real (Kafka + modelo).

**Monitoramento pós-deploy:**
- **Concept drift**: a relação entre X e y mudou (ex.: comportamento pós-COVID).
- **Data drift**: a distribuição de X mudou (ex.: novo segmento de clientes).
- **Performance degradation**: métricas de ML caindo ao longo do tempo.
- Alertas automáticos e gatilhos de re-treinamento.

---

## 3. Definição do Problema: Armadilhas Comuns

### 3.1 Problema Mal Definido
❌ **Errado**: "Quero um modelo de IA para melhorar o negócio."
✅ **Correto**: "Quero identificar, com 14 dias de antecedência, os clientes com probabilidade de cancelamento > 70%, para acionar uma oferta de retenção."

### 3.2 Métricas Erradas
- Usar acurácia em datasets desbalanceados (99% não-fraude → modelo que classifica tudo como "não-fraude" tem 99% de acurácia).
- Otimizar a métrica errada para o negócio (ex.: maximizar recall quando precisão é crítica).

### 3.3 Data Leakage
- Feature com informação do futuro vazando para o treino.
- Fazer fit do scaler/imputer em todos os dados antes da divisão treino/teste.
- Exemplos reais: incluir a data do evento como feature; usar o saldo pós-transação para prever fraude.

---

## 4. Versionamento e Rastreabilidade

### 4.1 Versionamento de Dados com DVC

**DVC** (Data Version Control) versiona datasets e modelos como o Git versiona código:

```bash
# Instalação
pip install dvc

# Inicializar DVC no repositório
git init && dvc init

# Adicionar dataset ao controle do DVC
dvc add data/raw/clientes.csv

# Rastrear com Git (metadata, não os dados em si)
git add data/raw/clientes.csv.dvc .gitignore
git commit -m "Adicionar dataset de clientes v1"

# Armazenar dados em remote (S3, GCS, Azure Blob)
dvc remote add -d myremote s3://meu-bucket/dvc-store
dvc push
```

### 4.2 Rastreamento de Experimentos com MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("churn-prediction")

with mlflow.start_run(run_name="random_forest_v2"):
    # Registrar hiperparâmetros
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("min_samples_split", 5)

    modelo = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
    )
    modelo.fit(X_train, y_train)

    # Registrar métricas
    acc = accuracy_score(y_test, modelo.predict(X_test))
    auc = roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1])
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("roc_auc", auc)

    # Registrar o modelo
    mlflow.sklearn.log_model(modelo, "model", registered_model_name="ChurnModel")

    print(f"Acurácia: {acc:.4f} | AUC: {auc:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

---

## 5. Exemplo Prático End-to-End: Predição de Churn

```python
# =============================================================
# Aula 08 — Pipeline End-to-End: Predição de Churn
# Simulando o fluxo CRISP-DM completo
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# FASE 1: CRIAÇÃO DO DATASET DE CHURN (simulado)
# ============================================================
print("=" * 60)
print("FASE 1: ENTENDIMENTO DO NEGÓCIO E DADOS")
print("=" * 60)

np.random.seed(42)
n_clientes = 5000

# Simular base de clientes com padrões realistas
tempo_contrato   = np.random.exponential(scale=24, size=n_clientes).clip(1, 72).astype(int)
uso_mensal_gb    = np.random.gamma(shape=2, scale=10, size=n_clientes).clip(0.1, 100)
num_chamados     = np.random.poisson(lam=1.5, size=n_clientes)
plano            = np.random.choice(['Básico', 'Standard', 'Premium'], n_clientes, p=[0.4, 0.4, 0.2])
valor_mensal     = np.where(plano == 'Básico', 50, np.where(plano == 'Standard', 80, 120))
valor_mensal    += np.random.normal(0, 10, n_clientes)
pagamento_auto   = np.random.choice([0, 1], n_clientes, p=[0.3, 0.7])
satisfacao       = np.random.randint(1, 11, n_clientes)  # 1 a 10

# Introduzir valores ausentes realistas
mask_satisf = np.random.random(n_clientes) < 0.08
satisfacao = satisfacao.astype(float)
satisfacao[mask_satisf] = np.nan

# Churn com padrões realistas:
# - Clientes com alto num_chamados → mais churn
# - Plano Básico → mais churn
# - Pagamento automático → menos churn
# - Satisfação baixa → mais churn
logit_churn = (
    -3.0
    + 0.05 * num_chamados
    - 0.02 * tempo_contrato
    + 0.01 * uso_mensal_gb
    + np.where(plano == 'Básico', 0.8, np.where(plano == 'Standard', 0.3, 0.0))
    - 0.6 * pagamento_auto
    + np.where(satisfacao < 5, 1.2, np.where(satisfacao < 7, 0.3, -0.5))
    + np.random.normal(0, 0.3, n_clientes)
)
prob_churn = 1 / (1 + np.exp(-logit_churn))
churn = (np.random.random(n_clientes) < prob_churn).astype(int)

df = pd.DataFrame({
    'tempo_contrato_meses': tempo_contrato,
    'uso_mensal_gb':        uso_mensal_gb.round(2),
    'num_chamados_suporte': num_chamados,
    'plano':                plano,
    'valor_mensal':         valor_mensal.round(2),
    'pagamento_automatico': pagamento_auto,
    'nota_satisfacao':      satisfacao,
    'churn':                churn
})

print(f"Dataset gerado: {df.shape}")
print(f"\nDistribuição de Churn:")
print(df['churn'].value_counts())
print(f"Taxa de Churn: {df['churn'].mean():.1%}")
print(f"\nValores ausentes:")
print(df.isnull().sum())

# ============================================================
# FASE 2: ANÁLISE EXPLORATÓRIA
# ============================================================
print("\n" + "=" * 60)
print("FASE 2: ANÁLISE EXPLORATÓRIA (EDA)")
print("=" * 60)

print("\nEstatísticas por grupo:")
print(df.groupby('churn')[['tempo_contrato_meses', 'num_chamados_suporte',
                             'nota_satisfacao', 'valor_mensal']].agg(['mean', 'median']))

# Visualizações
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Distribuição de churn
df['churn'].value_counts().plot(kind='bar', ax=axes[0,0], color=['steelblue', 'coral'])
axes[0,0].set(title='Distribuição de Churn', xlabel='Churn', ylabel='Contagem')
axes[0,0].set_xticklabels(['Não Churn', 'Churn'], rotation=0)

# 2. Churn por plano
churn_plano = df.groupby('plano')['churn'].mean().reset_index()
axes[0,1].bar(churn_plano['plano'], churn_plano['churn'] * 100, color=['coral', 'gold', 'steelblue'])
axes[0,1].set(title='Taxa de Churn por Plano', xlabel='Plano', ylabel='Taxa de Churn (%)')

# 3. Nota de satisfação vs churn
df.boxplot(column='nota_satisfacao', by='churn', ax=axes[0,2])
axes[0,2].set(title='Satisfação vs Churn', xlabel='Churn', ylabel='Nota de Satisfação')

# 4. Tempo de contrato
for churn_val, cor, label in [(0, 'steelblue', 'Não Churn'), (1, 'coral', 'Churn')]:
    subset = df[df['churn'] == churn_val]['tempo_contrato_meses']
    axes[1,0].hist(subset, bins=30, alpha=0.6, color=cor, label=label)
axes[1,0].set(title='Distribuição: Tempo de Contrato', xlabel='Meses', ylabel='Frequência')
axes[1,0].legend()

# 5. Chamados de suporte
for churn_val, cor, label in [(0, 'steelblue', 'Não Churn'), (1, 'coral', 'Churn')]:
    subset = df[df['churn'] == churn_val]['num_chamados_suporte']
    axes[1,1].hist(subset, bins=15, alpha=0.6, color=cor, label=label)
axes[1,1].set(title='Distribuição: Chamados de Suporte', xlabel='Chamados', ylabel='Frequência')
axes[1,1].legend()

# 6. Correlação
df_num = df.select_dtypes(include=[np.number])
sns.heatmap(df_num.corr(), ax=axes[1,2], annot=True, fmt='.2f',
            cmap='coolwarm', center=0, vmin=-1, vmax=1)
axes[1,2].set_title('Matriz de Correlação')

plt.suptitle('Análise Exploratória — Churn de Clientes', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("churn_eda.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# FASE 3: PREPARAÇÃO DOS DADOS
# ============================================================
print("\n" + "=" * 60)
print("FASE 3: PREPARAÇÃO DOS DADOS")
print("=" * 60)

# Separar features e target
X = df.drop('churn', axis=1)
y = df['churn']

# Codificar variáveis categóricas
le = LabelEncoder()
X['plano_encoded'] = le.fit_transform(X['plano'])
X = X.drop('plano', axis=1)

print(f"Features após encoding: {X.columns.tolist()}")

# Dividir treino/teste com estratificação
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTreino: {X_train.shape[0]} | Teste: {X_test.shape[0]}")
print(f"Taxa churn treino: {y_train.mean():.1%} | Teste: {y_test.mean():.1%}")

# ============================================================
# FASE 4: MODELAGEM
# ============================================================
print("\n" + "=" * 60)
print("FASE 4: MODELAGEM (3 modelos)")
print("=" * 60)

modelos = {
    'Regressão Logística': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('clf',     LogisticRegression(max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf',     RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42))
    ]),
    'Gradient Boosting': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('clf',     GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                               learning_rate=0.1, random_state=42))
    ])
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados = {}

for nome, pipeline in modelos.items():
    scores_auc = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring='roc_auc', n_jobs=-1)
    scores_f1  = cross_val_score(pipeline, X_train, y_train, cv=cv,
                                  scoring='f1', n_jobs=-1)
    resultados[nome] = {
        'AUC-ROC': scores_auc,
        'F1':      scores_f1
    }
    print(f"\n{nome}:")
    print(f"  AUC-ROC (CV): {scores_auc.mean():.4f} ± {scores_auc.std():.4f}")
    print(f"  F1      (CV): {scores_f1.mean():.4f} ± {scores_f1.std():.4f}")

# ============================================================
# FASE 5: AVALIAÇÃO DO MODELO CAMPEÃO
# ============================================================
print("\n" + "=" * 60)
print("FASE 5: AVALIAÇÃO DO MODELO CAMPEÃO (Gradient Boosting)")
print("=" * 60)

modelo_campeao = modelos['Gradient Boosting']
modelo_campeao.fit(X_train, y_train)

y_pred_proba = modelo_campeao.predict_proba(X_test)[:, 1]
y_pred_class = modelo_campeao.predict(X_test)

auc_final = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC no teste: {auc_final:.4f}")
print("\nRelatório de Classificação (threshold=0.5):")
print(classification_report(y_test, y_pred_class, target_names=['Não Churn', 'Churn']))

# Análise de threshold — impacto no negócio
print("\n=== Análise de Threshold — Impacto no Negócio ===")
print("Custo retenção/cliente: R$30  |  Receita salva/cliente retido: R$200")
custo_retencao = 30
receita_salva  = 200
thresholds = np.arange(0.1, 0.9, 0.05)
resultados_negocio = []

for thr in thresholds:
    y_pred_thr = (y_pred_proba >= thr).astype(int)
    vp = ((y_pred_thr == 1) & (y_test == 1)).sum()  # Verdadeiros Positivos
    fp = ((y_pred_thr == 1) & (y_test == 0)).sum()  # Falsos Positivos
    roi = vp * receita_salva - (vp + fp) * custo_retencao
    resultados_negocio.append({'threshold': thr, 'VP': vp, 'FP': fp, 'ROI': roi})

df_negocio = pd.DataFrame(resultados_negocio)
idx_melhor = df_negocio['ROI'].idxmax()
print(df_negocio.to_string(index=False))
print(f"\n→ Threshold ótimo para ROI: {df_negocio.loc[idx_melhor, 'threshold']:.2f}")
print(f"  ROI máximo: R${df_negocio.loc[idx_melhor, 'ROI']:,.0f}")

# Curvas ROC e Precision-Recall
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[0].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC = {auc_final:.3f}')
axes[0].plot([0, 1], [0, 1], 'k--', label='Aleatório')
axes[0].set(xlabel='Taxa Falsos Positivos', ylabel='Taxa Verdadeiros Positivos', title='Curva ROC')
axes[0].legend()
axes[0].fill_between(fpr, tpr, alpha=0.1, color='steelblue')

# Precision-Recall
prec, rec, thr_pr = precision_recall_curve(y_test, y_pred_proba)
axes[1].plot(rec, prec, color='darkorange', lw=2)
axes[1].set(xlabel='Recall', ylabel='Precisão', title='Curva Precision-Recall')
axes[1].axhline(y=y_test.mean(), color='k', linestyle='--', label=f'Baseline ({y_test.mean():.2f})')
axes[1].legend()

# ROI por threshold
axes[2].plot(df_negocio['threshold'], df_negocio['ROI'] / 1000,
             'o-', color='green', lw=2)
axes[2].axvline(x=df_negocio.loc[idx_melhor, 'threshold'],
                color='red', linestyle='--', label='Threshold ótimo')
axes[2].set(xlabel='Threshold', ylabel='ROI (R$ mil)', title='ROI por Threshold de Decisão')
axes[2].legend()

plt.tight_layout()
plt.savefig("churn_model_evaluation.png", dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# FASE 6: PREPARAÇÃO PARA DEPLOY
# ============================================================
print("\n" + "=" * 60)
print("FASE 6: SERIALIZAÇÃO DO MODELO (Deploy)")
print("=" * 60)

import joblib
import json
from datetime import datetime

# Salvar o pipeline completo (inclui imputer e modelo)
joblib.dump(modelo_campeao, 'churn_model_v1.joblib')
print("Modelo salvo: churn_model_v1.joblib")

# Metadados do modelo
metadados = {
    "nome_modelo":     "GradientBoostingClassifier",
    "versao":          "1.0",
    "data_treino":     datetime.now().isoformat(),
    "n_amostras_treino": int(len(X_train)),
    "features":        X.columns.tolist(),
    "metricas": {
        "roc_auc_teste": round(float(auc_final), 4)
    },
    "threshold_recomendado": float(df_negocio.loc[idx_melhor, 'threshold'])
}

with open('churn_model_v1_metadata.json', 'w') as f:
    json.dump(metadados, f, indent=2, ensure_ascii=False)

print("Metadados salvos: churn_model_v1_metadata.json")

# Demonstrar carregamento e uso
modelo_carregado = joblib.load('churn_model_v1.joblib')
novos_clientes = X_test.head(5)
probs = modelo_carregado.predict_proba(novos_clientes)[:, 1]
threshold_otimo = metadados['threshold_recomendado']

print("\n=== Previsão para 5 novos clientes ===")
for i, (prob, real) in enumerate(zip(probs, y_test.values[:5])):
    acao = "ACIONAR RETENÇÃO" if prob >= threshold_otimo else "manter"
    print(f"  Cliente {i+1}: prob_churn={prob:.3f} | real={real} | ação={acao}")
```

---

## 6. Boas Práticas e Armadilhas Comuns

### 6.1 Boas Práticas ✅

| Prática | Descrição |
|---------|-----------|
| Baseline primeiro | Sempre comece com um modelo simples como referência |
| Stratify nas divisões | Manter proporção das classes em treino/teste/validação |
| Evitar data leakage | Fit apenas nos dados de treino (scaler, imputer, encoders) |
| Versionar código e dados | Git para código, DVC para dados, MLflow para experimentos |
| Documentar experimentos | Registrar hiperparâmetros, métricas e artefatos de cada run |
| Monitorar em produção | Alertas de data drift e performance degradation |
| Reproducibilidade | `random_state` em todos os componentes estocásticos |

### 6.2 Armadilhas Comuns ❌

| Armadilha | Consequência | Solução |
|-----------|-------------|---------|
| Data Leakage | Overfitting otimista; falha em produção | Fazer fit apenas no treino |
| Acurácia em dados desbalanceados | Modelo trivial parece bom | Usar F1, AUC-ROC, AUC-PR |
| Tuning no conjunto de teste | Overfitting no teste | Usar conjunto de validação separado |
| Ignorar o negócio | Modelo tecnicamente bom, negócio ruim | Alinhar métricas com stakeholders |
| Sem monitoramento | Performance deteriora silenciosamente | Implementar monitoring desde o início |

---

## 7. Exercícios

1. **Mapeamento CRISP-DM**: Para o problema de "prever quais alunos vão reprovar no semestre", descreva como cada fase do CRISP-DM seria executada. Quais dados seriam coletados? Quais features? Qual métrica de ML? Qual métrica de negócio?

2. **Data Leakage**: No código de churn acima, identifique onde poderia ocorrer data leakage se o código fosse escrito de forma ingênua (sem Pipeline). Como o Pipeline resolve isso?

3. **Análise de Threshold**: Modifique os parâmetros de custo de retenção (R$30) e receita salva (R$200) para R$80 e R$150, respectivamente. Como muda o threshold ótimo? Interprete o resultado.

4. **MLflow**: Instale o MLflow (`pip install mlflow`) e adicione rastreamento ao pipeline de churn, registrando hiperparâmetros e métricas de cada modelo. Compare os runs na interface web (`mlflow ui`).

---

## 8. Referências

- GÉRON, Aurélien. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow*. 3. ed. Alta Books, 2023. Cap. 2 (Projeto End-to-End).
- FACELI, Katti et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. 2. ed. LTC, 2021. Cap. 1.
- WIRTH, R.; HIPP, J. CRISP-DM: Towards a standard process model for data mining. *4th International Conference on the Practical Applications of Knowledge Discovery and Data Mining*, 2000.
- MLflow Documentation. [https://mlflow.org/docs/latest/index.html](https://mlflow.org/docs/latest/index.html)
- DVC Documentation. [https://dvc.org/doc](https://dvc.org/doc)

---

*← [Aula 07 — Tipos de Aprendizado](aula-07-tipos-de-aprendizado.md) | [Aula 09 — Ecossistema de Ferramentas](aula-09-ecossistema-ferramentas.md) →*
