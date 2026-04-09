# Práticas — Inteligência Artificial e Aprendizado de Máquina

## Visão Geral

Cada prática é um laboratório hands-on complementar ao conteúdo teórico. O código é executável no Google Colab, Jupyter Notebook ou ambiente local.

## Lista de Práticas

| # | Título | Módulos | Dataset |
|---|--------|---------|---------|
| 01 | [Python para Data Science](01-python-data-science/pratica.md) | 01 | Sintético |
| 02 | [Exploração de Dados (EDA)](02-exploracao-dados/pratica.md) | 02–03 | Titanic |
| 03 | [Pipeline de Preparação de Dados](03-preparacao-dados/pratica.md) | 03 | Adult Income |
| 04 | [KNN — Classificação](04-knn-classificacao/pratica.md) | 04 | Iris |
| 05 | [Regressão Linear](05-regressao-linear/pratica.md) | 05 | California Housing |
| 06 | [Regressão Logística e SVM](06-regressao-logistica-svm/pratica.md) | 05 | Breast Cancer |
| 07 | [Árvores e Random Forest](07-arvores-random-forest/pratica.md) | 06 | Titanic |
| 08 | [Gradient Boosting (XGBoost)](08-gradient-boosting/pratica.md) | 06 | Tabular tabular |
| 09 | [Redes Neurais com Keras](09-redes-neurais-keras/pratica.md) | 07 | MNIST |
| 10 | [Pipeline de Avaliação Completo](10-avaliacao-validacao/pratica.md) | 08 | Breast Cancer |
| 11 | [CNN — Classificação de Imagens](11-cnn-visao/pratica.md) | 10 | CIFAR-10 |
| 12 | [NLP com Transformers](12-nlp-transformers/pratica.md) | 10–11 | IMDb/IMDB PT |
| 13 | [Projeto Final — End-to-End](13-projeto-final/pratica.md) | Todos | Escolha livre |

## Como Usar

```bash
# Clone o repositório
git clone https://github.com/heliobentzen/ia.git
cd ia

# Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Abra o Jupyter
jupyter lab
```

## Ambiente Recomendado

```
Python >= 3.10
scikit-learn >= 1.4
pandas >= 2.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
tensorflow >= 2.15  (ou torch >= 2.0)
transformers >= 4.38
xgboost >= 2.0
lightgbm >= 4.0
shap >= 0.44
```
