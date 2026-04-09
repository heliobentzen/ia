# Aula 58 — Explicabilidade — SHAP e LIME

> **Módulo 12 · Ética, Interpretabilidade e Uso Responsável de IA** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Aplicar SHAP para explicar predições de qualquer modelo
- Usar LIME para explicações locais
- Distinguir interpretabilidade global e local

---

## 1. Por Que Explicabilidade?

- **Regulatório:** LGPD e GDPR exigem direito à explicação em decisões automatizadas
- **Confiança:** usuários confiam mais em modelos que podem ser explicados
- **Debug:** SHAP revela features problemáticas ou features que são proxies de atributos protegidos
- **Conhecimento:** modelos podem confirmar hipóteses de domínio

---

## 2. SHAP — SHapley Additive exPlanations

Baseado na teoria dos jogos cooperativos. O valor de Shapley da feature $j$ é a média ponderada de sua contribuição marginal em todas as ordens possíveis de features.

$$\phi_j(f) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S \cup \{j\}) - f(S)]$$

```python
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_tr, y_tr)

# SHAP TreeExplainer (rápido para modelos tree-based)
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_te)

# Importância global (bee swarm plot)
shap.summary_plot(shap_values[1], X_te, plot_type="dot",
                  max_display=10, show=True)

# Importância global (bar plot)
shap.summary_plot(shap_values[1], X_te, plot_type="bar",
                  max_display=10, show=True)

# Explicação local — uma predição individual
idx = 0
shap.waterfall_plot(shap.Explanation(
    values=shap_values[1][idx],
    base_values=explainer.expected_value[1],
    data=X_te.iloc[idx],
    feature_names=cancer.feature_names
))

# Dependence plot: interação entre features
shap.dependence_plot('worst radius', shap_values[1], X_te, show=True)
```

---

## 3. LIME — Local Interpretable Model-agnostic Explanations

LIME aproxima o modelo complexo localmente com um modelo linear interpretável:

```python
import lime
import lime.lime_tabular
import numpy as np

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    X_tr.values,
    feature_names=cancer.feature_names.tolist(),
    class_names=['Maligno', 'Benigno'],
    mode='classification'
)

# Explicar uma instância
instance = X_te.iloc[0].values
explanation = explainer_lime.explain_instance(
    instance, rf.predict_proba, num_features=10
)
explanation.show_in_notebook(show_table=True)
# ou
fig = explanation.as_pyplot_figure()
fig.tight_layout()
```

---

## 4. SHAP vs. LIME

| Característica | SHAP | LIME |
|---------------|------|------|
| Base teórica | Teoria dos jogos | Modelo local linear |
| Consistência | Garante consistência | Não garante |
| Velocidade | Rápido (Tree) / lento (Kernel) | Moderado |
| Escopo | Global + local | Principalmente local |
| Modelos | Qualquer | Qualquer |

---

## Questões para Reflexão
1. Por que explicabilidade local e global são ambas importantes?
2. Como SHAP pode ser usado para detectar viés em um modelo?
3. Um modelo explicável é sempre um modelo correto? Por quê não necessariamente?

## Referências
- Géron, cap. 19
- Faceli et al., cap. 13
- Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions" (2017)

---
*Próxima aula → [Aula 59: Privacidade e LGPD](aula-59-privacidade-lgpd.md)*
