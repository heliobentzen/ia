# Aula 38 — Métricas de Classificação

> **Módulo 08 · Avaliação e Validação de Modelos** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Interpretar matriz de confusão, precisão, recall, F1 e AUC-ROC
- Escolher métricas de acordo com o custo dos erros (FP vs. FN)
- Trabalhar com dados desbalanceados

---

## 1. Matriz de Confusão

|                | Predito Positivo | Predito Negativo |
|---------------|-----------------|-----------------|
| **Real Positivo** | TP (Verdadeiro Positivo) | FN (Falso Negativo) |
| **Real Negativo** | FP (Falso Positivo) | TN (Verdadeiro Negativo) |

---

## 2. Métricas Derivadas

$$\text{Acurácia} = \frac{TP+TN}{TP+TN+FP+FN}$$

$$\text{Precisão} = \frac{TP}{TP+FP} \quad \text{(quando prevejo positivo, estou certo?)}$$

$$\text{Recall (Sensibilidade)} = \frac{TP}{TP+FN} \quad \text{(capturo todos os positivos?)}$$

$$F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\text{Precisão} + \text{Recall}}$$

$$F_\beta = (1+\beta^2) \cdot \frac{\text{Precisão} \cdot \text{Recall}}{\beta^2\cdot\text{Precisão} + \text{Recall}}$$

---

## 3. Implementação Completa

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)

cancer = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(
    cancer.data, cancer.target, test_size=0.2, random_state=42, stratify=cancer.target
)

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)
y_pred  = rf.predict(X_te)
y_prob  = rf.predict_proba(X_te)[:, 1]

# Matriz de confusão
cm = confusion_matrix(y_te, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Maligno','Benigno'], yticklabels=['Maligno','Benigno'])
plt.title('Matriz de Confusão'); plt.show()

# Relatório completo
print(classification_report(y_te, y_pred, target_names=cancer.target_names))

# Curva ROC
fpr, tpr, _ = roc_curve(y_te, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR'); plt.ylabel('TPR (Recall)')
plt.title('Curva ROC'); plt.legend(); plt.show()

# Curva Precisão-Recall (melhor para dados desbalanceados)
precision, recall, _ = precision_recall_curve(y_te, y_prob)
ap = average_precision_score(y_te, y_prob)
plt.plot(recall, precision, label=f'AP = {ap:.4f}')
plt.xlabel('Recall'); plt.ylabel('Precisão')
plt.title('Curva Precisão-Recall'); plt.legend(); plt.show()
```

---

## 4. Ajuste de Threshold

```python
# Threshold padrão = 0.5, mas pode ser ajustado
thresholds = np.arange(0.1, 0.9, 0.05)
f1_scores  = []

from sklearn.metrics import f1_score
for t in thresholds:
    y_pred_t = (y_prob >= t).astype(int)
    f1_scores.append(f1_score(y_te, y_pred_t))

best_t = thresholds[np.argmax(f1_scores)]
print(f"Melhor threshold: {best_t:.2f} | F1: {max(f1_scores):.4f}")
```

---

## 5. Guia de Seleção de Métrica

| Situação | Métrica |
|---------|---------|
| Classes balanceadas | Acurácia, F1 macro |
| FP custoso (spam filter) | Alta Precisão |
| FN custoso (diagnóstico câncer) | Alto Recall |
| Dados desbalanceados | F1, AUC-PR |
| Comparação geral | AUC-ROC |

---

## Questões para Reflexão
1. Em diagnóstico de câncer, o que é pior: FP ou FN? Como isso afeta a escolha de métricas?
2. Por que a acurácia é enganosa com classes desbalanceadas (ex: 99% negativo)?
3. Como o threshold afeta o trade-off Precisão/Recall?

## Referências
- Géron, cap. 3
- Faceli et al., cap. 4

---
*Próxima aula → [Aula 39: Validação Cruzada](aula-39-cross-validation.md)*
