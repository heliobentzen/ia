# 📂 Índice de Conteúdos

> **Curso:** Inteligência Artificial e Aprendizado de Máquina
> **Total:** 60 aulas · 12 módulos · 45 horas de aula (60 × 45 min)

Bem-vindo ao índice central de conteúdos do curso! Esta página é o seu **mapa de navegação**: aqui você encontra todos os módulos, todas as aulas e uma descrição do que será estudado em cada etapa. Use os links diretos para acessar qualquer aula, em qualquer ordem.

> 💡 **Dica de navegação:** se você está seguindo o curso sequencialmente, avance aula por aula. Se está usando como referência, vá direto ao módulo de interesse. Cada arquivo de aula é autocontido — teoria, exemplos em Python e exercícios.

---

## 🗺️ Visão Geral dos Módulos

| # | Módulo | Aulas | Período | Temas centrais |
|---|--------|:-----:|---------|----------------|
| 1 | [Fundamentos de IA](#-módulo-1--fundamentos-de-ia) | 5 | 01–05 | História, agentes, panorama |
| 2 | [Paradigmas de Aprendizado de Máquina](#-módulo-2--paradigmas-de-aprendizado-de-máquina) | 4 | 06–09 | Supervisionado, não-sup., RL |
| 3 | [Preparação e Análise de Dados](#-módulo-3--preparação-e-análise-de-dados) | 6 | 10–15 | EDA, limpeza, features, pipelines |
| 4 | [Aprendizado Supervisionado e Não Supervisionado](#-módulo-4--aprendizado-supervisionado-e-não-supervisionado) | 5 | 16–20 | Viés-variância, k-NN, PCA |
| 5 | [Regressão e Classificação](#-módulo-5--regressão-e-classificação) | 6 | 21–26 | Lin./Log. Reg., SVM, Naive Bayes |
| 6 | [Métodos baseados em Árvores e Ensembles](#-módulo-6--métodos-baseados-em-árvores-e-ensembles) | 5 | 27–31 | RF, XGBoost, LightGBM |
| 7 | [Redes Neurais Artificiais](#-módulo-7--redes-neurais-artificiais) | 5 | 32–36 | Perceptron, MLP, backprop |
| 8 | [Avaliação e Validação de Modelos](#-módulo-8--avaliação-e-validação-de-modelos) | 5 | 37–41 | ROC, CV, hiperparâmetros |
| 9 | [Overfitting e Regularização](#-módulo-9--overfitting-e-regularização) | 4 | 42–45 | L1/L2, Dropout, Early Stopping |
| 10 | [Introdução ao Aprendizado Profundo](#-módulo-10--introdução-ao-aprendizado-profundo) | 6 | 46–51 | CNN, RNN, Transformer, TL |
| 11 | [Aplicações de ML em Problemas Reais](#-módulo-11--aplicações-de-ml-em-problemas-reais) | 5 | 52–56 | NLP, visão, séries, MLOps |
| 12 | [Ética, Interpretabilidade e Uso Responsável](#-módulo-12--ética-interpretabilidade-e-uso-responsável) | 4 | 57–60 | Viés, SHAP, LGPD, IA responsável |

---

## 📋 Listagem Completa de Aulas

A seguir, todas as **60 aulas** do curso, com links diretos e descrição resumida de cada uma.

---

## 🔵 Módulo 1 — Fundamentos de IA

> **Aulas 01–05 · 5 aulas · ~3h45min**
>
> O primeiro módulo situa o estudante no universo da Inteligência Artificial: de onde viemos, onde estamos e para onde estamos indo. Você vai entender as definições precisas de IA, AM e Aprendizado Profundo, conhecer os tipos de agentes inteligentes e ter uma visão panorâmica das revoluções tecnológicas recentes — dos modelos de linguagem aos sistemas multimodais.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 01](01-fundamentos-ia/aula-01-historia-ia.md) | História e Evolução da Inteligência Artificial | Linha do tempo da IA: teste de Turing, invernos e primaveras da IA, deep learning revolution. |
| [Aula 02](01-fundamentos-ia/aula-02-definicoes.md) | Definições: IA, AM e Aprendizado Profundo | Hierarquia de conceitos; diferenças entre IA simbólica e conexionista; o que é aprender. |
| [Aula 03](01-fundamentos-ia/aula-03-tipos-ia.md) | Tipos de IA: Fraca, Forte e Geral (AGI) | IA estreita vs. AGI vs. superinteligência; debates filosóficos; estado atual da arte. |
| [Aula 04](01-fundamentos-ia/aula-04-agentes-inteligentes.md) | Agentes Inteligentes: PEAS e Ambientes | Modelo PEAS (Performance, Environment, Actuators, Sensors); tipos de agentes; racionalidade. |
| [Aula 05](01-fundamentos-ia/aula-05-panorama-atual.md) | Panorama Atual da IA | LLMs, modelos de difusão, visão computacional, multimodalidade; impacto social e econômico. |

---

## 🟢 Módulo 2 — Paradigmas de Aprendizado de Máquina

> **Aulas 06–09 · 4 aulas · ~3h**
>
> Aqui você aprende a **pensar em termos de AM**: qual é o tipo de problema que tenho? Quais dados estão disponíveis? O que quero que o modelo faça? O módulo classifica os paradigmas fundamentais — supervisionado, não supervisionado, semissupervisionado e por reforço — e apresenta a crescente área do self-supervised learning, que deu origem aos grandes modelos de linguagem.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 06](02-paradigmas-aprendizado/aula-06-supervisionado.md) | Aprendizado Supervisionado | Pares (entrada, saída); tarefas de classificação e regressão; exemplos de datasets. |
| [Aula 07](02-paradigmas-aprendizado/aula-07-nao-supervisionado.md) | Aprendizado Não Supervisionado | Clustering, redução de dimensionalidade, modelos generativos; quando usar. |
| [Aula 08](02-paradigmas-aprendizado/aula-08-semissupervisionado-reforco.md) | Semissupervisionado e Aprendizado por Reforço | Aproveitando poucos rótulos; agentes, ambiente, recompensa; aplicações em jogos e robótica. |
| [Aula 09](02-paradigmas-aprendizado/aula-09-self-supervised-transfer.md) | Self-Supervised Learning e Transfer Learning | Previsão de máscara, contraste, pretext tasks; como GPT e BERT são pré-treinados. |

---

## 🟡 Módulo 3 — Preparação e Análise de Dados

> **Aulas 10–15 · 6 aulas · ~4h30min**
>
> Dado que **80% do tempo em projetos de ML é gasto com dados**, este módulo é um dos mais práticos e importantes. Você vai aprender a explorar, limpar, transformar e versionar dados, construir pipelines reprodutíveis com Scikit-Learn e evitar a armadilha do *data leakage*. Nenhum modelo bom nasce de dados ruins.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 10](03-preparacao-dados/aula-10-eda-introducao.md) | Introdução à Análise Exploratória de Dados (EDA) | Estatísticas descritivas; distribuições; visualização com Pandas, Matplotlib e Seaborn. |
| [Aula 11](03-preparacao-dados/aula-11-qualidade-dados.md) | Qualidade de Dados: Faltantes e Outliers | Estratégias de imputação; detecção e tratamento de outliers; dados duplicados. |
| [Aula 12](03-preparacao-dados/aula-12-feature-engineering.md) | Feature Engineering: Criação de Atributos | Transformações matemáticas; interações; discretização; extração de datas e textos. |
| [Aula 13](03-preparacao-dados/aula-13-normalizacao-codificacao.md) | Normalização, Padronização e Codificação | Min-Max, Z-Score, RobustScaler; One-Hot Encoding, Label Encoding, Target Encoding. |
| [Aula 14](03-preparacao-dados/aula-14-selecao-features.md) | Seleção de Features | Filter methods, wrapper methods (RFE), embedded methods (Lasso); curse of dimensionality. |
| [Aula 15](03-preparacao-dados/aula-15-pipelines-dados.md) | Pipelines de Pré-Processamento | `Pipeline` e `ColumnTransformer` do Scikit-Learn; data leakage; versionamento de dados. |

---

## 🟠 Módulo 4 — Aprendizado Supervisionado e Não Supervisionado

> **Aulas 16–20 · 5 aulas · ~3h45min**
>
> Com os dados preparados, este módulo aprofunda os dois grandes paradigmas do AM. O foco está nos conceitos fundamentais que regem todo e qualquer modelo: o **dilema viés-variância**, a ideia de generalização e as heurísticas para escolher algoritmos. Você implementará k-NN, k-Means, DBSCAN e aprenderá a visualizar estruturas de alta dimensão com PCA e t-SNE.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 16](04-supervisionado-nao-supervisionado/aula-16-vies-variancia.md) | O Dilema Viés-Variância | Underfitting, overfitting e o sweet spot; curvas de aprendizado; capacidade do modelo. |
| [Aula 17](04-supervisionado-nao-supervisionado/aula-17-knn.md) | Algoritmo k-NN (k-Vizinhos Mais Próximos) | Classificação e regressão por proximidade; métricas de distância; escolha do k ótimo. |
| [Aula 18](04-supervisionado-nao-supervisionado/aula-18-kmeans.md) | Clustering com k-Means | Algoritmo de Lloyd; centróides; critério do cotovelo; k-Means++ para inicialização. |
| [Aula 19](04-supervisionado-nao-supervisionado/aula-19-dbscan-avaliacao-cluster.md) | DBSCAN e Avaliação de Clustering | Densidade; ruído e outliers; Silhouette Score, Davies-Bouldin; quando usar cada algoritmo. |
| [Aula 20](04-supervisionado-nao-supervisionado/aula-20-pca-tsne.md) | PCA e t-SNE: Redução de Dimensionalidade | Componentes principais; variância explicada; t-SNE para visualização; UMAP introdução. |

---

## 🔴 Módulo 5 — Regressão e Classificação

> **Aulas 21–26 · 6 aulas · ~4h30min**
>
> Este módulo cobre os algoritmos supervisionados clássicos com profundidade. Você vai derivar a regressão linear usando mínimos quadrados e gradiente descendente, entender a regressão logística como um modelo probabilístico, explorar os fundamentos geométricos do SVM e aprender quando cada algoritmo brilha. Métricas de avaliação adequadas para cada tipo de problema são tratadas com rigor.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 21](05-regressao-classificacao/aula-21-regressao-linear.md) | Regressão Linear: OLS e Gradiente Descendente | Mínimos quadrados; equação normal; gradiente descendente batch, mini-batch e estocástico. |
| [Aula 22](05-regressao-classificacao/aula-22-regressao-multipla.md) | Regressão Múltipla e Polinomial | Múltiplas variáveis; multicolinearidade; expansão polinomial; análise de resíduos. |
| [Aula 23](05-regressao-classificacao/aula-23-regressao-logistica.md) | Regressão Logística e Classificação Binária | Função sigmoid; odds ratio; log-loss; limiar de decisão e fronteiras de decisão. |
| [Aula 24](05-regressao-classificacao/aula-24-classificacao-multiclasse.md) | Classificação Multiclasse e Multilabel | One-vs-Rest, One-vs-One; softmax; classificação multilabel e métricas associadas. |
| [Aula 25](05-regressao-classificacao/aula-25-svm.md) | Máquinas de Vetores de Suporte (SVM) | Margem máxima; vetores de suporte; kernel trick (linear, RBF, polinomial); SVR. |
| [Aula 26](05-regressao-classificacao/aula-26-naive-bayes.md) | Naive Bayes e Métricas de Desempenho | Teorema de Bayes; variantes (Gaussiano, Multinomial, Bernoulli); acurácia, precisão, recall, F1. |

---

## 🟤 Módulo 6 — Métodos baseados em Árvores e Ensembles

> **Aulas 27–31 · 5 aulas · ~3h45min**
>
> Árvores de decisão são intuitivas e poderosas, mas sozinhas têm limitações. Combinadas em ensembles, tornam-se os algoritmos mais competitivos para dados tabulares. Neste módulo você vai entender como funcionam Random Forest, AdaBoost, Gradient Boosting e os campeões do Kaggle — XGBoost e LightGBM — além de aprender a interpretar a importância das features.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 27](06-arvores-ensembles/aula-27-arvores-decisao.md) | Árvores de Decisão: ID3, C4.5 e CART | Critérios de divisão (Gini, Entropia, MSE); poda; profundidade; interpretabilidade. |
| [Aula 28](06-arvores-ensembles/aula-28-random-forest.md) | Random Forest e Bagging | Bootstrap aggregating; aleatorização de features; out-of-bag error; feature importance. |
| [Aula 29](06-arvores-ensembles/aula-29-adaboost.md) | Boosting: AdaBoost | Combinação sequencial de aprendizes fracos; pesos adaptativos; análise de erros. |
| [Aula 30](06-arvores-ensembles/aula-30-gradient-boosting.md) | Gradient Boosting e GBDT | Ajuste de resíduos; learning rate; subsample; Shrinkage; comparação com RF. |
| [Aula 31](06-arvores-ensembles/aula-31-xgboost-lightgbm.md) | XGBoost, LightGBM e CatBoost | Otimizações computacionais; regularização; tratamento de categóricas; tuning avançado. |

---

## 🟣 Módulo 7 — Redes Neurais Artificiais

> **Aulas 32–36 · 5 aulas · ~3h45min**
>
> Das inspirações biológicas ao treinamento de redes profundas modernas. Neste módulo você vai entender o Perceptron e suas limitações, derivar o algoritmo de backpropagation passo a passo, estudar as funções de ativação modernas e implementar sua primeira rede neural com Keras. A jornada do neurônio artificial à arquitetura MLP completa.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 32](07-redes-neurais/aula-32-perceptron.md) | Neurônio Biológico e o Perceptron | Modelo de McCulloch-Pitts; Perceptron de Rosenblatt; limitações (XOR); regra de aprendizado. |
| [Aula 33](07-redes-neurais/aula-33-mlp-backpropagation.md) | Redes Multicamadas (MLP) e Backpropagation | Forward pass; função de perda; regra da cadeia; gradiente descendente em redes profundas. |
| [Aula 34](07-redes-neurais/aula-34-funcoes-ativacao.md) | Funções de Ativação | Sigmoid, tanh, ReLU, Leaky ReLU, ELU, Swish, GELU; problema do gradiente desvanecente. |
| [Aula 35](07-redes-neurais/aula-35-treinamento-keras.md) | Treinamento com Keras e TensorFlow | API Sequential e Functional; camadas, otimizadores (SGD, Adam, RMSProp); callbacks. |
| [Aula 36](07-redes-neurais/aula-36-arquiteturas-praticas.md) | Arquiteturas de Referência e Boas Práticas | Inicialização de pesos; Batch Normalization; Skip Connections; tensorboard; debugging. |

---

## ⚫ Módulo 8 — Avaliação e Validação de Modelos

> **Aulas 37–41 · 5 aulas · ~3h45min**
>
> Um modelo só é bom se conseguimos medir sua qualidade de forma honesta e robusta. Este módulo é dedicado inteiramente à avaliação: as métricas certas para cada problema, como evitar ser enganado por dados de teste, técnicas de validação cruzada, busca de hiperparâmetros e testes estatísticos para comparar modelos. Saber avaliar é tão importante quanto saber treinar.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 37](08-avaliacao-validacao/aula-37-metricas-classificacao.md) | Métricas de Classificação | Matriz de confusão; precisão, recall, F1; micro/macro/weighted average; quando usar cada uma. |
| [Aula 38](08-avaliacao-validacao/aula-38-roc-pr.md) | Curvas ROC, PR e AUC | Curva ROC e AUC-ROC; Precision-Recall e AUC-PR; calibração de probabilidades; Brier Score. |
| [Aula 39](08-avaliacao-validacao/aula-39-metricas-regressao.md) | Métricas de Regressão e Comparação | MAE, MSE, RMSE, R², MAPE; análise de resíduos; comparação entre modelos concorrentes. |
| [Aula 40](08-avaliacao-validacao/aula-40-validacao-cruzada.md) | Validação Cruzada | k-fold, stratified k-fold, leave-one-out, time-series split; escolha do k; nested CV. |
| [Aula 41](08-avaliacao-validacao/aula-41-selecao-hiperparametros.md) | Seleção de Hiperparâmetros | Grid Search, Random Search, Optuna (Bayesian Optimization); Halving Search; testes estatísticos. |

---

## 🔷 Módulo 9 — Overfitting e Regularização

> **Aulas 42–45 · 4 aulas · ~3h**
>
> Overfitting é o inimigo número um do AM. Este módulo apresenta um arsenal de técnicas para combatê-lo: regularização nos parâmetros do modelo (L1/L2), técnicas específicas para redes neurais (Dropout, Batch Normalization, Early Stopping) e estratégias para ampliar artificialmente o dataset (Data Augmentation). Você aprenderá a diagnosticar e tratar overfitting de forma sistemática.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 42](09-overfitting-regularizacao/aula-42-regularizacao-l1-l2.md) | Regularização L1 (Lasso) e L2 (Ridge) | Penalidades nos pesos; Elastic Net; efeito de esparsidade; path de regularização; seleção de λ. |
| [Aula 43](09-overfitting-regularizacao/aula-43-dropout-batchnorm.md) | Dropout e Batch Normalization | Dropout como ensemble implícito; BN: treinamento vs. inferência; Layer Normalization. |
| [Aula 44](09-overfitting-regularizacao/aula-44-early-stopping-lr.md) | Early Stopping e Learning Rate Schedules | Monitoramento de validação; paciência; Cosine Annealing, Step Decay, Warm Restarts. |
| [Aula 45](09-overfitting-regularizacao/aula-45-data-augmentation.md) | Data Augmentation e Técnicas de Expansão | Augmentation para imagens e texto; MixUp, CutMix; synthetic data; SMOTE para desbalanceamento. |

---

## 🔶 Módulo 10 — Introdução ao Aprendizado Profundo

> **Aulas 46–51 · 6 aulas · ~4h30min**
>
> O salto para o aprendizado profundo. Você vai entender como as CNNs "enxergam" imagens por meio de filtros e pooling, como as RNNs e LSTMs processam sequências com memória, e como o mecanismo de Atenção revolucionou o processamento de linguagem dando origem à arquitetura Transformer. Por fim, você aprenderá a aproveitar modelos pré-treinados de bilhões de parâmetros para suas próprias tarefas.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 46](10-aprendizado-profundo/aula-46-cnn-introducao.md) | Redes Convolucionais (CNNs) — Introdução | Filtros e feature maps; padding e stride; pooling; receptive field; arquiteturas LeNet e AlexNet. |
| [Aula 47](10-aprendizado-profundo/aula-47-cnn-avancado.md) | CNNs Avançadas: VGG, ResNet e Inception | Batch Normalization em CNNs; skip connections (ResNets); arquiteturas modernas (EfficientNet). |
| [Aula 48](10-aprendizado-profundo/aula-48-rnn-lstm.md) | RNNs, LSTMs e GRUs | Gradiente no tempo (BPTT); problema do gradiente desvanecente; LSTM: cell state e gates; GRU. |
| [Aula 49](10-aprendizado-profundo/aula-49-atencao-transformer.md) | Mecanismo de Atenção e Transformer | Self-attention; multi-head attention; positional encoding; encoder-decoder; BERT vs. GPT. |
| [Aula 50](10-aprendizado-profundo/aula-50-transfer-learning.md) | Transfer Learning e Fine-Tuning | Feature extraction vs. fine-tuning; congelamento de camadas; domain adaptation; few-shot. |
| [Aula 51](10-aprendizado-profundo/aula-51-modelos-fundacionais.md) | Modelos Fundacionais e LLMs | BERT, GPT-2/3/4, CLIP, SAM; prompting; in-context learning; limitações e riscos. |

---

## 🌐 Módulo 11 — Aplicações de ML em Problemas Reais

> **Aulas 52–56 · 5 aulas · ~3h45min**
>
> A teoria encontra a prática. Neste módulo você vai aplicar o que aprendeu em domínios reais: processamento de linguagem natural (classificação de sentimentos, NER), visão computacional (detecção de objetos), previsão de séries temporais, sistemas de recomendação e o ciclo completo de vida de um projeto de ML em produção — o chamado MLOps. Projetos integradores são apresentados em cada aula.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 52](11-aplicacoes-reais/aula-52-nlp-aplicado.md) | NLP Aplicado: Classificação e NER | Tokenização; TF-IDF; word embeddings; classificação de sentimentos; Named Entity Recognition. |
| [Aula 53](11-aplicacoes-reais/aula-53-visao-computacional.md) | Visão Computacional: Detecção e Segmentação | Object detection (YOLO, Faster R-CNN); segmentação semântica e de instâncias; OpenCV. |
| [Aula 54](11-aplicacoes-reais/aula-54-series-temporais.md) | Séries Temporais: Previsão e Anomalia | Stationarity; ARIMA; Prophet; LSTM para séries; detecção de anomalias; avaliação temporal. |
| [Aula 55](11-aplicacoes-reais/aula-55-sistemas-recomendacao.md) | Sistemas de Recomendação | Filtragem colaborativa; baseada em conteúdo; matrix factorization; cold start problem. |
| [Aula 56](11-aplicacoes-reais/aula-56-mlops.md) | MLOps: Deploy e Ciclo de Vida de Modelos | Versionamento (MLflow, DVC); serialização; APIs com FastAPI/Flask; monitoramento de drift. |

---

## ⚖️ Módulo 12 — Ética, Interpretabilidade e Uso Responsável

> **Aulas 57–60 · 4 aulas · ~3h**
>
> O módulo final — e talvez o mais importante para a formação do profissional de IA completo. Algoritmos de AM tomam decisões que afetam vidas humanas: crédito, saúde, justiça, emprego. Neste módulo você vai aprender a identificar e mitigar viés algorítmico, a usar ferramentas de explicabilidade (SHAP, LIME), a navegar pelas regulações de proteção de dados (LGPD, GDPR) e a adotar princípios de IA responsável em todos os seus projetos.

| Aula | Título | Descrição |
|------|--------|-----------|
| [Aula 57](12-etica-interpretabilidade/aula-57-vies-algoritmico.md) | Viés Algorítmico: Origens e Impactos | Viés nos dados, no modelo e no uso; estudos de caso (COMPAS, reconhecimento facial, saúde). |
| [Aula 58](12-etica-interpretabilidade/aula-58-explicabilidade-shap-lime.md) | Explicabilidade: SHAP e LIME | Modelos caixa-branca vs. caixa-preta; SHAP values; LIME; Integrated Gradients; Grad-CAM. |
| [Aula 59](12-etica-interpretabilidade/aula-59-lgpd-gdpr.md) | LGPD, GDPR e Regulações de IA | Lei Geral de Proteção de Dados (Brasil); GDPR (Europa); regulação da IA na UE (AI Act); direitos. |
| [Aula 60](12-etica-interpretabilidade/aula-60-ia-responsavel.md) | IA Responsável: FAT, Fairness e o Futuro | Fairness, Accountability, Transparency; checklist de IA responsável; tendências e carreira em IA. |

---

## 🔗 Navegação Rápida — Todas as 60 Aulas

Use esta lista para navegar rapidamente para qualquer aula:

<details>
<summary><strong>📂 Módulo 1 — Fundamentos de IA (aulas 01–05)</strong></summary>

- [Aula 01 — História e Evolução da IA](01-fundamentos-ia/aula-01-historia-ia.md)
- [Aula 02 — Definições: IA, AM e Aprendizado Profundo](01-fundamentos-ia/aula-02-definicoes.md)
- [Aula 03 — Tipos de IA: Fraca, Forte e AGI](01-fundamentos-ia/aula-03-tipos-ia.md)
- [Aula 04 — Agentes Inteligentes: PEAS e Ambientes](01-fundamentos-ia/aula-04-agentes-inteligentes.md)
- [Aula 05 — Panorama Atual da IA](01-fundamentos-ia/aula-05-panorama-atual.md)

</details>

<details>
<summary><strong>📂 Módulo 2 — Paradigmas de Aprendizado de Máquina (aulas 06–09)</strong></summary>

- [Aula 06 — Aprendizado Supervisionado](02-paradigmas-aprendizado/aula-06-supervisionado.md)
- [Aula 07 — Aprendizado Não Supervisionado](02-paradigmas-aprendizado/aula-07-nao-supervisionado.md)
- [Aula 08 — Semissupervisionado e Aprendizado por Reforço](02-paradigmas-aprendizado/aula-08-semissupervisionado-reforco.md)
- [Aula 09 — Self-Supervised Learning e Transfer Learning](02-paradigmas-aprendizado/aula-09-self-supervised-transfer.md)

</details>

<details>
<summary><strong>📂 Módulo 3 — Preparação e Análise de Dados (aulas 10–15)</strong></summary>

- [Aula 10 — Introdução à EDA](03-preparacao-dados/aula-10-eda-introducao.md)
- [Aula 11 — Qualidade de Dados: Faltantes e Outliers](03-preparacao-dados/aula-11-qualidade-dados.md)
- [Aula 12 — Feature Engineering: Criação de Atributos](03-preparacao-dados/aula-12-feature-engineering.md)
- [Aula 13 — Normalização, Padronização e Codificação](03-preparacao-dados/aula-13-normalizacao-codificacao.md)
- [Aula 14 — Seleção de Features](03-preparacao-dados/aula-14-selecao-features.md)
- [Aula 15 — Pipelines de Pré-Processamento](03-preparacao-dados/aula-15-pipelines-dados.md)

</details>

<details>
<summary><strong>📂 Módulo 4 — Aprendizado Supervisionado e Não Supervisionado (aulas 16–20)</strong></summary>

- [Aula 16 — O Dilema Viés-Variância](04-supervisionado-nao-supervisionado/aula-16-vies-variancia.md)
- [Aula 17 — Algoritmo k-NN](04-supervisionado-nao-supervisionado/aula-17-knn.md)
- [Aula 18 — Clustering com k-Means](04-supervisionado-nao-supervisionado/aula-18-kmeans.md)
- [Aula 19 — DBSCAN e Avaliação de Clustering](04-supervisionado-nao-supervisionado/aula-19-dbscan-avaliacao-cluster.md)
- [Aula 20 — PCA e t-SNE](04-supervisionado-nao-supervisionado/aula-20-pca-tsne.md)

</details>

<details>
<summary><strong>📂 Módulo 5 — Regressão e Classificação (aulas 21–26)</strong></summary>

- [Aula 21 — Regressão Linear: OLS e Gradiente Descendente](05-regressao-classificacao/aula-21-regressao-linear.md)
- [Aula 22 — Regressão Múltipla e Polinomial](05-regressao-classificacao/aula-22-regressao-multipla.md)
- [Aula 23 — Regressão Logística e Classificação Binária](05-regressao-classificacao/aula-23-regressao-logistica.md)
- [Aula 24 — Classificação Multiclasse e Multilabel](05-regressao-classificacao/aula-24-classificacao-multiclasse.md)
- [Aula 25 — Máquinas de Vetores de Suporte (SVM)](05-regressao-classificacao/aula-25-svm.md)
- [Aula 26 — Naive Bayes e Métricas de Desempenho](05-regressao-classificacao/aula-26-naive-bayes.md)

</details>

<details>
<summary><strong>📂 Módulo 6 — Métodos baseados em Árvores e Ensembles (aulas 27–31)</strong></summary>

- [Aula 27 — Árvores de Decisão: ID3, C4.5 e CART](06-arvores-ensembles/aula-27-arvores-decisao.md)
- [Aula 28 — Random Forest e Bagging](06-arvores-ensembles/aula-28-random-forest.md)
- [Aula 29 — Boosting: AdaBoost](06-arvores-ensembles/aula-29-adaboost.md)
- [Aula 30 — Gradient Boosting e GBDT](06-arvores-ensembles/aula-30-gradient-boosting.md)
- [Aula 31 — XGBoost, LightGBM e CatBoost](06-arvores-ensembles/aula-31-xgboost-lightgbm.md)

</details>

<details>
<summary><strong>📂 Módulo 7 — Redes Neurais Artificiais (aulas 32–36)</strong></summary>

- [Aula 32 — Neurônio Biológico e o Perceptron](07-redes-neurais/aula-32-perceptron.md)
- [Aula 33 — MLP e Backpropagation](07-redes-neurais/aula-33-mlp-backpropagation.md)
- [Aula 34 — Funções de Ativação](07-redes-neurais/aula-34-funcoes-ativacao.md)
- [Aula 35 — Treinamento com Keras e TensorFlow](07-redes-neurais/aula-35-treinamento-keras.md)
- [Aula 36 — Arquiteturas de Referência e Boas Práticas](07-redes-neurais/aula-36-arquiteturas-praticas.md)

</details>

<details>
<summary><strong>📂 Módulo 8 — Avaliação e Validação de Modelos (aulas 37–41)</strong></summary>

- [Aula 37 — Métricas de Classificação](08-avaliacao-validacao/aula-37-metricas-classificacao.md)
- [Aula 38 — Curvas ROC, PR e AUC](08-avaliacao-validacao/aula-38-roc-pr.md)
- [Aula 39 — Métricas de Regressão e Comparação](08-avaliacao-validacao/aula-39-metricas-regressao.md)
- [Aula 40 — Validação Cruzada](08-avaliacao-validacao/aula-40-validacao-cruzada.md)
- [Aula 41 — Seleção de Hiperparâmetros](08-avaliacao-validacao/aula-41-selecao-hiperparametros.md)

</details>

<details>
<summary><strong>📂 Módulo 9 — Overfitting e Regularização (aulas 42–45)</strong></summary>

- [Aula 42 — Regularização L1 (Lasso) e L2 (Ridge)](09-overfitting-regularizacao/aula-42-regularizacao-l1-l2.md)
- [Aula 43 — Dropout e Batch Normalization](09-overfitting-regularizacao/aula-43-dropout-batchnorm.md)
- [Aula 44 — Early Stopping e Learning Rate Schedules](09-overfitting-regularizacao/aula-44-early-stopping-lr.md)
- [Aula 45 — Data Augmentation e Técnicas de Expansão](09-overfitting-regularizacao/aula-45-data-augmentation.md)

</details>

<details>
<summary><strong>📂 Módulo 10 — Introdução ao Aprendizado Profundo (aulas 46–51)</strong></summary>

- [Aula 46 — CNNs: Introdução](10-aprendizado-profundo/aula-46-cnn-introducao.md)
- [Aula 47 — CNNs Avançadas: VGG, ResNet e Inception](10-aprendizado-profundo/aula-47-cnn-avancado.md)
- [Aula 48 — RNNs, LSTMs e GRUs](10-aprendizado-profundo/aula-48-rnn-lstm.md)
- [Aula 49 — Mecanismo de Atenção e Transformer](10-aprendizado-profundo/aula-49-atencao-transformer.md)
- [Aula 50 — Transfer Learning e Fine-Tuning](10-aprendizado-profundo/aula-50-transfer-learning.md)
- [Aula 51 — Modelos Fundacionais e LLMs](10-aprendizado-profundo/aula-51-modelos-fundacionais.md)

</details>

<details>
<summary><strong>📂 Módulo 11 — Aplicações de ML em Problemas Reais (aulas 52–56)</strong></summary>

- [Aula 52 — NLP Aplicado: Classificação e NER](11-aplicacoes-reais/aula-52-nlp-aplicado.md)
- [Aula 53 — Visão Computacional: Detecção e Segmentação](11-aplicacoes-reais/aula-53-visao-computacional.md)
- [Aula 54 — Séries Temporais: Previsão e Anomalia](11-aplicacoes-reais/aula-54-series-temporais.md)
- [Aula 55 — Sistemas de Recomendação](11-aplicacoes-reais/aula-55-sistemas-recomendacao.md)
- [Aula 56 — MLOps: Deploy e Ciclo de Vida de Modelos](11-aplicacoes-reais/aula-56-mlops.md)

</details>

<details>
<summary><strong>📂 Módulo 12 — Ética, Interpretabilidade e Uso Responsável (aulas 57–60)</strong></summary>

- [Aula 57 — Viés Algorítmico: Origens e Impactos](12-etica-interpretabilidade/aula-57-vies-algoritmico.md)
- [Aula 58 — Explicabilidade: SHAP e LIME](12-etica-interpretabilidade/aula-58-explicabilidade-shap-lime.md)
- [Aula 59 — LGPD, GDPR e Regulações de IA](12-etica-interpretabilidade/aula-59-lgpd-gdpr.md)
- [Aula 60 — IA Responsável: FAT, Fairness e o Futuro](12-etica-interpretabilidade/aula-60-ia-responsavel.md)

</details>

---

## 📌 Dicas de Navegação e Estudo

### Como cada arquivo de aula é organizado

Todos os arquivos de aula seguem uma estrutura padronizada para facilitar o estudo:

```
# Aula XX — Título

## 🎯 Objetivos de Aprendizagem
## 📖 Introdução
## 🔬 Desenvolvimento Teórico
## 💻 Código e Exemplos em Python
## 📊 Visualizações e Experimentos
## ✅ Resumo
## 🔗 Leituras Complementares
## 📝 Exercícios
```

### Estratégias de Estudo Recomendadas

| Perfil | Estratégia |
|--------|-----------|
| 🟢 **Iniciante** | Siga a ordem sequencial aula por aula; faça todos os exercícios |
| 🟡 **Intermediário** | Leia as introduções de cada módulo e aprofunde onde houver lacunas |
| 🔴 **Avançado** | Use como referência; explore os projetos integradores em `praticas/` |

### Convenções nos Arquivos

- 🔵 **Teoria:** blocos azuis com fundamentos matemáticos
- 💻 **Código:** blocos Python com `# comentários em português`
- ⚠️ **Atenção:** avisos sobre armadilhas e erros comuns
- 💡 **Dica:** boas práticas e truques úteis
- 📝 **Exercício:** atividades para fixação do conteúdo

---

<div align="center">

[← Voltar ao README principal](../README.md) · [Práticas →](../praticas/)

*Última atualização: 2024 · 60 aulas · 12 módulos · Curso de IA e AM*

</div>
