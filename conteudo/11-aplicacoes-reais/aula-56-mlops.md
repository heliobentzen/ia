# Aula 56 — MLOps — Deploy e Monitoramento

> **Módulo 11 · Aplicações de ML em Problemas Reais** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Entender o ciclo de vida de um modelo em produção
- Implementar uma API de ML com FastAPI
- Usar MLflow para tracking de experimentos e registro de modelos

---

## 1. O Ciclo MLOps

```
Desenvolvimento  →  Empacotamento  →  Deploy  →  Monitoramento
     ↑                                                  |
     └──────────────── Re-treinamento ←─────────────────┘
```

**Ferramentas:**
- **MLflow**: tracking de experimentos, registro de modelos
- **DVC**: versionamento de dados
- **FastAPI**: serving de modelos como API REST
- **Docker**: containerização
- **Kubernetes**: orquestração em escala
- **Evidently**: monitoramento de data drift

---

## 2. Tracking de Experimentos com MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("cancer-classification")

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42
)

configs = [
    {'n_estimators': 100, 'max_depth': 5},
    {'n_estimators': 200, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': None},
]

for cfg in configs:
    with mlflow.start_run():
        rf = RandomForestClassifier(**cfg, random_state=42)
        rf.fit(X_tr, y_tr)
        y_pred = rf.predict(X_te)
        y_prob = rf.predict_proba(X_te)[:, 1]
        
        # Log parâmetros e métricas
        mlflow.log_params(cfg)
        mlflow.log_metric("f1", f1_score(y_te, y_pred))
        mlflow.log_metric("auc_roc", roc_auc_score(y_te, y_prob))
        
        # Log do modelo
        mlflow.sklearn.log_model(rf, "model")
        print(f"Run logged: {cfg}")
```

---

## 3. Servindo o Modelo com FastAPI

```python
# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Cancer Classifier API", version="1.0")

# Carregar modelo
model = joblib.load("model.pkl")

class PredictionRequest(BaseModel):
    features: list[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    label: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if len(request.features) != 30:
        raise HTTPException(status_code=400, detail="Expected 30 features")
    
    X = np.array(request.features).reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][pred]
    
    return PredictionResponse(
        prediction=int(pred),
        probability=float(prob),
        label="maligno" if pred == 0 else "benigno"
    )

# Rodar: uvicorn main:app --reload
# Testar: http://localhost:8000/docs
```

---

## 4. Monitoramento — Data Drift

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
import pandas as pd

# Referência (dados de treino) vs. produção (dados novos)
reference = pd.DataFrame(X_tr, columns=[f'f{i}' for i in range(30)])
production = pd.DataFrame(X_te, columns=[f'f{i}' for i in range(30)])

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=production)
report.save_html("drift_report.html")
print("Relatório de drift salvo!")
```

---

## Questões para Reflexão
1. O que é "model decay" e como o monitoramento de drift pode detectá-lo?
2. Por que versionar dados (DVC) é tão importante quanto versionar código (Git)?
3. Como você implementaria A/B testing para comparar dois modelos em produção?

## Referências
- Géron, cap. 19
- mlflow.org, fastapi.tiangolo.com, evidently.ai

---
*Módulo 11 concluído! Próximo → [Módulo 12: Ética, Interpretabilidade e Uso Responsável](../12-etica-interpretabilidade/README.md)*
