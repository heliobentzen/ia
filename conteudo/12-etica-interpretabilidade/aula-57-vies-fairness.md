# Aula 57 — Viés e Fairness em IA

> **Módulo 12 · Ética, Interpretabilidade e Uso Responsável de IA** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Identificar fontes de viés em dados e modelos de ML
- Calcular métricas de fairness (demographic parity, equalized odds)
- Aplicar técnicas de mitigação de viés

---

## 1. Tipos de Viés em IA

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| Viés histórico | Dados refletem desigualdades passadas | Modelos de crédito que discriminam por raça |
| Viés de representação | Sub/super-representação nos dados | Reconhecimento facial com baixa acurácia para negros |
| Viés de medição | Features proxy de atributos protegidos | CEP como proxy de raça |
| Viés de feedback | Decisões do modelo afetam dados futuros | Policiamento preditivo auto-reforçador |
| Viés de publicação | Só modelos com bons resultados são publicados | Super-estimativa da SOTA |

---

## 2. Atributos Protegidos

Características que legalmente não devem influenciar decisões:
- Raça, cor, etnia
- Gênero, identidade de gênero
- Idade
- Deficiência
- Religião
- Origem nacional

> **Atenção:** remover o atributo protegido do modelo NÃO garante equidade — outros atributos podem ser proxies (ex: nome, CEP, escola).

---

## 3. Métricas de Fairness

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Dataset sintético com atributo protegido
np.random.seed(42)
n = 1000
grupo = np.random.choice(['A', 'B'], n, p=[0.6, 0.4])

# Grupo A tem mais features favoráveis (desigualdade histórica simulada)
X = np.random.randn(n, 5)
X[grupo == 'A'] += 0.3  # vantagem histórica

y = (X[:, 0] + X[:, 1] > 0).astype(int)

df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
df['grupo'] = grupo
df['y'] = y

X_tr, X_te, y_tr, y_te, g_tr, g_te = train_test_split(
    df.drop(['grupo','y'], axis=1), df['y'], df['grupo'],
    test_size=0.2, random_state=42
)

model = LogisticRegression().fit(X_tr, y_tr)
y_pred = model.predict(X_te)

# Métricas por grupo
def fairness_metrics(y_true, y_pred, grupo):
    grupos = np.unique(grupo)
    for g in grupos:
        mask = grupo == g
        yt = y_true[mask]
        yp = y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0,1]).ravel()
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision
        pos_rate = yp.mean()  # Taxa de predições positivas
        
        print(f"Grupo {g}: TPR={tpr:.3f} | FPR={fpr:.3f} | PPV={ppv:.3f} | Pos Rate={pos_rate:.3f}")

fairness_metrics(y_te.values, y_pred, g_te.values)

# Demographic Parity: diferença nas taxas de predição positiva entre grupos
pos_A = y_pred[g_te == 'A'].mean()
pos_B = y_pred[g_te == 'B'].mean()
print(f"\nDemographic Parity Gap: {abs(pos_A - pos_B):.4f}")
print("(< 0.1 é geralmente aceitável)")
```

---

## 4. Técnicas de Mitigação

**Pre-processing:** rebalancear dados, remover proxies
**In-processing:** adicionar restrições de fairness ao treinamento
**Post-processing:** ajustar thresholds por grupo

```python
# Ajuste de threshold por grupo (post-processing)
def threshold_por_grupo(y_prob, grupos, g_te, target_metric='tpr', target=0.8):
    thresholds = {}
    for g in np.unique(grupos):
        mask = g_te == g
        # Encontrar threshold que atinge a TPR alvo
        from sklearn.metrics import roc_curve
        fpr, tpr, thrs = roc_curve(y_te[mask], y_prob[mask])
        idx = np.argmin(np.abs(tpr - target))
        thresholds[g] = thrs[idx]
    return thresholds
```

---

## Questões para Reflexão
1. É possível que um modelo seja simultaneamente justo em demographic parity E equalized odds? Sempre?
2. Por que remover o atributo protegido do dataset não elimina o viés?
3. Quem deve ser responsável quando um sistema de IA discrimina: o desenvolvedor, a empresa ou o usuário?

## Referências
- Russell & Norvig, cap. 27 (IA: Filosofia, Ética e Segurança)
- Faceli et al., cap. 13

---
*Próxima aula → [Aula 58: Explicabilidade — SHAP e LIME](aula-58-explicabilidade.md)*
