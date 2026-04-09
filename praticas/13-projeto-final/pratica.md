# Prática 13 — Projeto Final: Pipeline End-to-End de ML

**Todos os Módulos** | **Duração:** ~8–16 horas | **Individual ou em dupla**

## Descrição

Desenvolva um projeto completo de Machine Learning desde a definição do problema até o deploy, aplicando as melhores práticas aprendidas ao longo do curso.

---

## 1. Escolha do Dataset

Selecione um dataset de pelo menos uma das fontes abaixo (ou outro de sua preferência):

| Fonte | Exemplos de datasets |
|-------|---------------------|
| Kaggle | House Prices, Titanic, Credit Card Fraud |
| UCI ML Repository | Wine Quality, Bank Marketing |
| HuggingFace Datasets | imdb, amazon_polarity (PT) |
| Dados Abertos Brasil | dados.gov.br (saúde, educação) |
| IBGE | Censo, PNAD, POF |

**Requisito mínimo:** >1000 amostras, >5 features, target definido.

---

## 2. Estrutura do Projeto

```
projeto-final/
├── data/
│   ├── raw/          # dados originais
│   └── processed/    # dados após pré-processamento
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-preprocessamento.ipynb
│   ├── 03-modelos.ipynb
│   └── 04-analise-final.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
├── app/
│   └── main.py       # FastAPI
├── reports/
│   └── model_card.md
├── requirements.txt
└── README.md
```

---

## 3. Etapas Obrigatórias

### Etapa 1: Entendimento do Problema (EDA)
- [ ] Definir o tipo de problema (classificação/regressão/clustering)
- [ ] Identificar métricas de sucesso de negócio
- [ ] EDA completo com pelo menos 8 visualizações
- [ ] Identificação de valores ausentes, outliers e distribuições
- [ ] Formular 3+ hipóteses e testá-las estatisticamente

### Etapa 2: Pré-processamento
- [ ] Construir pipeline com `ColumnTransformer` + `Pipeline`
- [ ] Tratar valores ausentes (SimpleImputer ou KNNImputer)
- [ ] Codificar variáveis categóricas
- [ ] Normalizar/padronizar variáveis numéricas
- [ ] Feature engineering: criar pelo menos 2 features novas
- [ ] Selecionar features relevantes

### Etapa 3: Modelagem
- [ ] Implementar baseline (DummyClassifier/Regressor)
- [ ] Treinar pelo menos 4 modelos diferentes
- [ ] Usar validação cruzada estratificada (10-fold)
- [ ] Ajustar hiperparâmetros com RandomizedSearchCV ou Optuna
- [ ] Reportar métricas de treino e validação

### Etapa 4: Avaliação
- [ ] Avaliar no conjunto de teste (nunca visto antes)
- [ ] Plotar curvas de aprendizado
- [ ] Comparar modelos estatisticamente (Wilcoxon)
- [ ] Analisar exemplos de erro
- [ ] Calcular SHAP values para o melhor modelo

### Etapa 5: Deploy (opcional, mas valorizado)
- [ ] Exportar modelo com joblib ou `model.save()`
- [ ] Criar API com FastAPI
- [ ] Documentar endpoints no Swagger (`/docs`)
- [ ] Containerizar com Docker

### Etapa 6: Documentação e Ética
- [ ] Escrever Model Card completo
- [ ] Análise de fairness (se atributos protegidos presentes)
- [ ] Limitações conhecidas do modelo
- [ ] Recomendações de uso responsável

---

## 4. Template de Model Card

```markdown
# Model Card — [Nome do Projeto]

## Sumário
[Breve descrição do problema e solução]

## Detalhes do Modelo
- Tipo: [Classificação binária / Regressão / ...]
- Algoritmo: [XGBoost / Random Forest / ...]
- Framework: [scikit-learn / TensorFlow / ...]
- Data de treinamento: ...

## Dados
- Dataset: ...
- Tamanho: ... amostras, ... features
- Período: ...
- Features usadas: ...

## Performance
| Métrica | Treino | Validação | Teste |
|---------|--------|-----------|-------|
| [Acurácia/RMSE/F1] | ... | ... | ... |

## Análise de Fairness
- Atributos protegidos: ...
- Métricas por grupo: ...

## Limitações
- ...

## Uso Responsável
- Casos de uso indicados: ...
- Casos de uso não indicados: ...
- Supervisão humana recomendada: [Sim/Não]
```

---

## 5. Critérios de Avaliação

| Critério | Peso | Descrição |
|---------|------|---------|
| EDA e visualizações | 20% | Profundidade, insights, hipóteses testadas |
| Pipeline de ML | 25% | Qualidade do código, uso de melhores práticas |
| Modelagem e avaliação | 25% | Seleção de modelos, métricas, comparação estatística |
| Interpretabilidade | 15% | SHAP, análise de erros, insights |
| Ética e documentação | 15% | Model Card, fairness, limitações |

---

## 6. Dicas

1. Escolha um problema que você **genuinamente ache interessante**
2. Comece simples (baseline) e itere
3. Documente todas as decisões no notebook
4. Versione o código com Git desde o início
5. Não busque acurácia máxima — busque **entendimento profundo**

---

## Entregáveis

- Repositório GitHub público com código e notebooks
- Apresentação de 10-15 minutos demonstrando o projeto
- Model Card no repositório
- (Opcional) Deploy em Hugging Face Spaces ou Railway

## Referências
- Géron, cap. 2 (Projeto End-to-End)
- CRISP-DM Guide
- Google Model Card Toolkit: modelcards.withgoogle.com
