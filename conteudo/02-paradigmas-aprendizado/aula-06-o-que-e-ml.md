# Aula 06 — O Que é Aprendizado de Máquina?

> **Módulo 02 · Aula 06 · Carga horária: 1 hora-aula**

---

## Objetivos da Aula

- Compreender a definição formal de Aprendizado de Máquina.
- Distinguir ML da programação tradicional baseada em regras.
- Conhecer os marcos históricos que moldaram a área.
- Reconhecer aplicações reais e os tipos de tarefas de ML.
- Entender por que ML se tornou central na tecnologia atual.
- Implementar o primeiro modelo de ML em Python com scikit-learn.

---

## 1. Definição Formal de Aprendizado de Máquina

### 1.1 A Definição de Mitchell (1997)

A definição mais citada na literatura foi proposta por Tom Mitchell em seu livro clássico *Machine Learning* (1997):

> *"A computer program is said to learn from experience **E** with respect to some class of tasks **T** and performance measure **P**, if its performance at tasks in **T**, as measured by **P**, improves with experience **E**."*

**Tradução:**
> *"Diz-se que um programa de computador aprende a partir de uma experiência **E** com relação a uma classe de tarefas **T** e uma medida de desempenho **P**, se seu desempenho nas tarefas de **T**, conforme medido por **P**, melhora com a experiência **E**."*

**Exemplo prático — Filtro de spam:**

| Componente | Exemplo concreto |
|------------|-----------------|
| **E** (Experiência) | Milhares de e-mails rotulados como "spam" ou "não spam" |
| **T** (Tarefa) | Classificar novos e-mails como spam ou não spam |
| **P** (Desempenho) | Percentual de e-mails classificados corretamente (acurácia) |

O filtro **aprende** quando, ao ser exposto a mais exemplos de spam (E), sua acurácia (P) ao classificar novos e-mails (T) aumenta.

### 1.2 Outras Perspectivas

**Arthur Samuel (1959)** — pioneiro dos jogos de damas auto-aprendidos:
> *"Aprendizado de Máquina é o campo de estudo que dá aos computadores a capacidade de aprender sem serem explicitamente programados."*

**Pedro Domingos** (em *The Master Algorithm*, 2015):
> *"ML é como a automação da automação: em vez de programar computadores, você os programa para se programarem."*

---

## 2. ML vs. Programação Tradicional

### 2.1 Abordagem Tradicional (baseada em regras)

Na programação convencional, o desenvolvedor:
1. Analisa o problema.
2. Escreve regras explícitas (lógica `if/else`, expressões regulares, etc.).
3. O programa aplica essas regras a novos dados.

```
Dados + Regras Explícitas → Programa Tradicional → Respostas
```

**Exemplo — Detecção de spam por regras:**
```python
def eh_spam(email):
    palavras_spam = ["grátis", "clique aqui", "ganhe dinheiro", "oferta imperdível"]
    texto = email.lower()
    for palavra in palavras_spam:
        if palavra in texto:
            return True
    if email.count("!") > 5:
        return True
    return False
```

**Problemas:** spammers aprendem a contornar as regras. A lista de palavras precisa ser atualizada manualmente. Impossível cobrir todas as variações.

### 2.2 Abordagem de Machine Learning

Com ML, o fluxo é invertido:

```
Dados + Respostas (rótulos) → Algoritmo de ML → Modelo (regras aprendidas)
Novos Dados → Modelo → Previsões
```

O programa **descobre** as regras sozinho a partir dos exemplos. O modelo de spam pode detectar padrões que humanos nunca identificariam (combinação de horário de envio + domínio + estrutura HTML + frequência de maiúsculas).

### 2.3 Quando Usar ML vs. Programação Tradicional

| Critério | Programação Tradicional | Machine Learning |
|----------|------------------------|-----------------|
| Regras conhecidas e estáveis | ✅ Ideal | ❌ Desnecessário |
| Regras complexas ou desconhecidas | ❌ Difícil | ✅ Ideal |
| Volume de dados grande | ❌ Ineficiente | ✅ Necessário |
| Necessidade de explicabilidade total | ✅ Possível | ⚠️ Depende do modelo |
| Ambiente que muda com o tempo | ❌ Requer reescrita | ✅ Re-treinamento |
| Exemplos rotulados disponíveis | N/A | ✅ Necessário (supervisionado) |

---

## 3. Breve História do Aprendizado de Máquina

### 3.1 Linha do Tempo dos Marcos Fundamentais

**Décadas de 1940–1950 — As Origens**
- **1943** — McCulloch & Pitts propõem o primeiro modelo matemático de neurônio artificial.
- **1950** — Alan Turing publica "*Computing Machinery and Intelligence*" e propõe o Teste de Turing.
- **1952** — Arthur Samuel implementa o primeiro programa de xadrez/damas auto-aprendido.

**Década de 1950–1960 — O Perceptron**
- **1957** — Frank Rosenblatt cria o **Perceptron**, o primeiro algoritmo de aprendizado de rede neural, implementado em hardware (Mark I). O perceptron aprendia a classificar padrões simples.
- **1958–1969** — Entusiasmo inicial, financiamento militar (DARPA). Primeiras demonstrações de tradução automática.

**Décadas de 1970–1980 — Inverno e Renascimento**
- **1969** — Minsky & Papert publicam *Perceptrons*, demonstrando limitações dos perceptrons de uma camada (XOR problem). Primeiro "inverno da IA".
- **1986** — Rumelhart, Hinton & Williams redescobrem o algoritmo de **backpropagation**, tornando o treinamento de redes multicamadas viável.
- **1989** — LeCun aplica redes convolucionais (CNNs) ao reconhecimento de dígitos manuscritos (LeNet).

**Décadas de 1990–2000 — Consolidação**
- **1995** — Vapnik & Cortes publicam as **Support Vector Machines (SVM)**, com forte embasamento matemático (teoria VC).
- **1997** — Tom Mitchell publica o livro *Machine Learning*. IBM Deep Blue vence Kasparov no xadrez.
- **2001** — Leo Breiman propõe as **Random Forests**.
- **2001** — Friedman publica **Gradient Boosting Machines**.

**Décadas de 2010–Presente — Deep Learning e Revolução dos Dados**
- **2012** — **AlexNet** vence o ImageNet com margem expressiva, inaugurando a era moderna do Deep Learning com GPUs.
- **2014** — Goodfellow propõe as **GANs** (Generative Adversarial Networks).
- **2017** — Google publica "*Attention Is All You Need*", introduzindo os **Transformers**.
- **2018** — **BERT** (Google) e **GPT** (OpenAI) revolucionam o NLP.
- **2020** — **GPT-3**: 175 bilhões de parâmetros, few-shot learning.
- **2022** — **ChatGPT** atinge 100 milhões de usuários em 2 meses (produto de crescimento mais rápido da história).
- **2023** — **GPT-4**, **Llama 2**, **Gemini**: modelos multimodais e código aberto proliferam.

---

## 4. Aplicações de Machine Learning

### 4.1 Visão Computacional
- Reconhecimento facial (Face ID, vigilância)
- Diagnóstico por imagem médica (detecção de câncer em radiografias)
- Veículos autônomos (detecção de pedestres, semáforos, faixas)
- Inspeção de qualidade industrial (defeitos em produtos na linha de montagem)
- Realidade aumentada e filtros de câmera

### 4.2 Processamento de Linguagem Natural (NLP)
- Tradução automática (Google Translate, DeepL)
- Assistentes virtuais (Siri, Alexa, Google Assistant)
- Geração de texto (ChatGPT, GitHub Copilot)
- Análise de sentimento (monitoramento de redes sociais)
- Sumarização automática de documentos

### 4.3 Saúde e Medicina
- Predição de doenças (diabetes, doenças cardíacas)
- Descoberta de medicamentos (AlphaFold — estrutura de proteínas)
- Triagem de pacientes em UTIs
- Análise de sinais de ECG e EEG
- Medicina personalizada (farmacogenômica)

### 4.4 Finanças
- Detecção de fraudes em cartões de crédito (em tempo real)
- Crédito automatizado (credit scoring)
- Trading algorítmico (alta frequência)
- Análise de risco de portfólios
- Detecção de lavagem de dinheiro (AML)

### 4.5 Sistemas de Recomendação
- Recomendação de filmes (Netflix), músicas (Spotify), produtos (Amazon)
- Feed de redes sociais (Facebook, TikTok, Instagram)
- Publicidade direcionada (Google Ads, Meta Ads)
- Recomendação de cursos (Coursera, Udemy)

### 4.6 Outras Aplicações
- **Energia**: previsão de demanda elétrica, otimização de redes elétricas.
- **Agricultura**: detecção de doenças em plantas, previsão de colheita.
- **Logística**: otimização de rotas (UPS, FedEx), previsão de demanda.
- **Jogos**: AlphaGo (Go), AlphaStar (StarCraft II), OpenAI Five (Dota 2).

---

## 5. Tipos de Tarefas de Machine Learning

### 5.1 Classificação
Prever a qual **categoria discreta** uma instância pertence.

- **Binária**: spam/não-spam, fraude/legítimo, positivo/negativo.
- **Multiclasse**: reconhecimento de dígitos (0–9), classificação de espécies.
- **Multilabel**: tagger de artigos (um artigo pode ser "tecnologia" + "política" + "economia").

### 5.2 Regressão
Prever um **valor numérico contínuo**.

- Preço de imóveis, temperatura futura, vendas do próximo trimestre.
- Tempo de vida de um equipamento (manutenção preditiva).

### 5.3 Clustering (Agrupamento)
Descobrir **grupos naturais** nos dados sem rótulos pré-definidos.

- Segmentação de clientes, agrupamento de documentos, compressão de imagem.

### 5.4 Geração
Criar **novos dados** semelhantes aos de treinamento.

- Geração de imagens (DALL-E, Stable Diffusion), texto (GPT), música, código.

### 5.5 Detecção de Anomalias
Identificar instâncias **raras ou atípicas**.

- Detecção de falhas em equipamentos, transações fraudulentas, intrusões de rede.

### 5.6 Ranqueamento
Ordenar itens por relevância.

- Busca web (Google), recomendações, seleção de candidatos em RH.

---

## 6. Por Que ML Agora? Os Três Pilares

### 6.1 Dados (Big Data)
O volume de dados digitais cresce exponencialmente:
- Em 2023, a humanidade gerou ~120 zettabytes de dados.
- Redes sociais, sensores IoT, transações digitais e dispositivos móveis geram dados continuamente.
- Mais dados = modelos mais precisos. Lei de escalabilidade: quanto mais dados, melhor o desempenho.

### 6.2 Poder Computacional
- **GPUs** (Unidades de Processamento Gráfico): paralelismo massivo ideal para multiplicação de matrizes (operação central em redes neurais).
- **TPUs** (Tensor Processing Units, Google): chips especializados para ML.
- **Computação em nuvem**: AWS, GCP, Azure disponibilizam clusters de GPUs por demanda.
- O custo de treinamento caiu ~10× a cada 5 anos (Lei de Moore adaptada ao ML).

### 6.3 Algoritmos e Frameworks
- Redes neurais profundas (Deep Learning) tornaram-se viáveis com backpropagation eficiente.
- **Frameworks open-source**: TensorFlow (2015), PyTorch (2016) democratizaram o acesso.
- **Modelos pré-treinados** (transfer learning): fine-tuning em horas ao invés de semanas.
- Comunidade científica ativa: ~300.000 artigos de ML publicados em 2023 (arXiv).

---

## 7. Limitações e Desafios de ML

### 7.1 Dependência de Dados
- ML precisa de **grandes volumes** de dados de qualidade.
- **Garbage In, Garbage Out**: dados tendenciosos geram modelos tendenciosos.
- **Privacidade**: dados pessoais exigem conformidade com LGPD/GDPR.

### 7.2 Interpretabilidade (Black Box)
- Modelos complexos (redes neurais profundas) são difíceis de interpretar.
- Em domínios críticos (medicina, justiça, crédito), é exigida explicabilidade.
- Área emergente: **XAI** (Explainable AI) — LIME, SHAP, Grad-CAM.

### 7.3 Generalização
- **Overfitting**: modelo "decora" os dados de treinamento e falha em dados novos.
- **Underfitting**: modelo muito simples não captura padrões relevantes.
- **Data drift**: distribuição dos dados muda com o tempo (ex.: padrões de comportamento pós-pandemia).

### 7.4 Viés Algorítmico (Bias)
- Se os dados de treinamento refletem preconceitos históricos, o modelo os perpetua.
- Exemplos reais: sistemas de reconhecimento facial com menor acurácia para pessoas negras; algoritmos de crédito desfavorecendo minorias.

### 7.5 Custo Computacional e Ambiental
- Treinar o GPT-3 consumiu ~1.287 MWh de eletricidade (~552 toneladas de CO₂).
- Inferência em escala (milhões de requisições/segundo) tem custo significativo.

### 7.6 Segurança
- **Adversarial examples**: perturbações imperceptíveis em imagens enganam classificadores.
- **Model poisoning**: atacante contamina os dados de treinamento.
- **Model inversion/extraction**: extração de dados sensíveis do modelo.

---

## 8. Primeiro Modelo de ML em Python

### 8.1 O Dataset Iris

O dataset Iris é o "Hello, World!" do ML supervisionado. Contém 150 amostras de flores de íris com 4 features (comprimento e largura de sépalas e pétalas) e 3 classes (Setosa, Versicolor, Virginica).

```python
# =============================================================
# Aula 06 — Primeiro Modelo de ML com scikit-learn
# Dataset: Iris (classificação multiclasse)
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------------
# 1. CARREGAR E EXPLORAR OS DADOS
# ------------------------------------------------------------------
iris = load_iris()

# Criar DataFrame para melhor visualização
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['especie'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("=== Primeiras 5 amostras ===")
print(df.head())

print(f"\n=== Formato do dataset: {df.shape} ===")
print(f"Features: {iris.feature_names}")
print(f"Classes:  {iris.target_names}")

print("\n=== Distribuição das classes ===")
print(df['especie'].value_counts())

print("\n=== Estatísticas descritivas ===")
print(df.describe())

# ------------------------------------------------------------------
# 2. SEPARAR FEATURES E TARGET
# ------------------------------------------------------------------
X = iris.data    # Matriz de features: shape (150, 4)
y = iris.target  # Vetor de targets:   shape (150,)

print(f"\nX (features) shape: {X.shape}")
print(f"y (target)   shape: {y.shape}")
print(f"Classes: {iris.target_names}")

# ------------------------------------------------------------------
# 3. DIVIDIR EM TREINO E TESTE
# ------------------------------------------------------------------
# random_state garante reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para teste, 80% para treino
    random_state=42,    # semente para reprodutibilidade
    stratify=y          # manter proporção das classes
)

print(f"\n=== Divisão Treino/Teste ===")
print(f"Treino: {X_train.shape[0]} amostras")
print(f"Teste:  {X_test.shape[0]} amostras")

# ------------------------------------------------------------------
# 4. PRÉ-PROCESSAMENTO: NORMALIZAÇÃO
# ------------------------------------------------------------------
# StandardScaler: média=0, desvio_padrão=1
# IMPORTANTE: fit() apenas nos dados de treino! Evita data leakage.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # fit + transform
X_test_scaled  = scaler.transform(X_test)        # apenas transform

print(f"\n=== Após normalização (primeiras 3 amostras de treino) ===")
print(f"Antes: {X_train[:3].round(2)}")
print(f"Depois: {X_train_scaled[:3].round(2)}")

# ------------------------------------------------------------------
# 5. TREINAMENTO — K-Nearest Neighbors (KNN)
# ------------------------------------------------------------------
print("\n" + "="*50)
print("MODELO 1: K-Nearest Neighbors (KNN)")
print("="*50)

knn = KNeighborsClassifier(n_neighbors=5)

# fit() = treinamento (aprendizado a partir dos dados)
knn.fit(X_train_scaled, y_train)

# predict() = inferência em novos dados
y_pred_knn = knn.predict(X_test_scaled)

# Avaliação
acc_knn = accuracy_score(y_test, y_pred_knn)
print(f"\nAcurácia do KNN: {acc_knn:.4f} ({acc_knn*100:.1f}%)")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))

# ------------------------------------------------------------------
# 6. TREINAMENTO — Árvore de Decisão
# ------------------------------------------------------------------
print("="*50)
print("MODELO 2: Árvore de Decisão")
print("="*50)

tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train_scaled, y_train)
y_pred_tree = tree.predict(X_test_scaled)

acc_tree = accuracy_score(y_test, y_pred_tree)
print(f"\nAcurácia da Árvore de Decisão: {acc_tree:.4f} ({acc_tree*100:.1f}%)")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred_tree, target_names=iris.target_names))

# ------------------------------------------------------------------
# 7. MATRIZ DE CONFUSÃO
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (modelo_nome, y_pred) in zip(
    axes,
    [("KNN (k=5)", y_pred_knn), ("Árvore de Decisão", y_pred_tree)]
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Matriz de Confusão — {modelo_nome}")

plt.tight_layout()
plt.savefig("iris_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nMatriz de confusão salva em 'iris_confusion_matrices.png'")

# ------------------------------------------------------------------
# 8. VISUALIZAÇÃO — FRONTEIRAS DE DECISÃO (2 features)
# ------------------------------------------------------------------
# Usar apenas as 2 primeiras features para visualizar em 2D
X_2d = X[:, :2]  # comprimento e largura da sépala
X_train_2d, X_test_2d, _, _ = train_test_split(
    X_2d, y, test_size=0.2, random_state=42, stratify=y
)

knn_2d = KNeighborsClassifier(n_neighbors=5)
knn_2d.fit(X_train_2d, y_train)

# Grade de pontos para colorir as regiões de decisão
x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
cores = ['red', 'yellow', 'blue']
for classe, cor in zip([0, 1, 2], cores):
    mask = y == classe
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=cor, label=iris.target_names[classe],
                edgecolors='k', s=60, alpha=0.8)

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title("Fronteiras de Decisão — KNN (k=5) em 2D")
plt.legend()
plt.tight_layout()
plt.savefig("iris_decision_boundary.png", dpi=150, bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------
# 9. FAZENDO PREVISÕES EM NOVOS DADOS
# ------------------------------------------------------------------
print("\n=== Previsão em novas flores ===")
nova_flor = np.array([[5.1, 3.5, 1.4, 0.2]])  # provavelmente Setosa
nova_flor_scaled = scaler.transform(nova_flor)
previsao = knn.predict(nova_flor_scaled)
probabilidades = knn.predict_proba(nova_flor_scaled)

print(f"Nova flor: {nova_flor}")
print(f"Previsão:  {iris.target_names[previsao[0]]}")
print(f"Probabilidades por classe:")
for nome, prob in zip(iris.target_names, probabilidades[0]):
    print(f"  {nome}: {prob:.2f}")

# ------------------------------------------------------------------
# 10. RESUMO: ANATOMIA DO SCIKIT-LEARN
# ------------------------------------------------------------------
print("\n=== Anatomia do scikit-learn ===")
print("""
Todos os modelos seguem a mesma API:

1. Instanciar:    modelo = NomeDoModelo(hiperparametros)
2. Treinar:       modelo.fit(X_train, y_train)
3. Prever:        y_pred = modelo.predict(X_test)
4. Avaliar:       accuracy_score(y_test, y_pred)

Transformadores (pré-processadores):
1. Instanciar:    scaler = StandardScaler()
2. Ajustar:       scaler.fit(X_train)      ← apenas nos dados de treino!
3. Transformar:   X_scaled = scaler.transform(X)
   (ou combinado) X_scaled = scaler.fit_transform(X_train)
""")
```

### 8.2 Saída Esperada

```
=== Primeiras 5 amostras ===
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) especie
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
...

Acurácia do KNN: 0.9667 (96.7%)
Acurácia da Árvore de Decisão: 0.9333 (93.3%)
```

---

## 9. Conceitos-Chave da Aula

| Termo | Definição |
|-------|-----------|
| **Feature (atributo)** | Variável de entrada do modelo (coluna no dataset) |
| **Target (rótulo)** | Variável que o modelo tenta prever |
| **Instância (amostra)** | Uma linha no dataset |
| **Treinamento** | Processo de ajustar os parâmetros do modelo aos dados |
| **Inferência** | Aplicar o modelo treinado a novos dados |
| **Acurácia** | Proporção de previsões corretas |
| **Overfitting** | Modelo funciona bem no treino mas mal no teste |
| **Hiperparâmetro** | Configuração do modelo escolhida antes do treinamento (ex: `n_neighbors`) |

---

## 10. Exercícios

1. **Conceitual**: Identifique E, T e P (Mitchell) para o caso de um algoritmo de reconhecimento de voz.

2. **Prático**: Modifique o código acima para testar KNN com `k ∈ {1, 3, 5, 7, 10, 15}`. Plote um gráfico de acurácia vs. k. Qual valor de k é melhor?

3. **Pesquisa**: Pesquise um caso de viés algorítmico real e descreva: (a) o sistema afetado, (b) o tipo de viés, (c) as consequências e (d) possíveis soluções.

4. **Desafio**: Use o dataset `load_wine()` do scikit-learn. Compare os resultados de KNN e Árvore de Decisão. Qual modelo performa melhor? Por quê?

---

## 11. Referências

- MITCHELL, Tom M. *Machine Learning*. McGraw-Hill, 1997. Capítulo 1.
- FACELI, Katti et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. 2. ed. LTC, 2021. Cap. 1.
- GÉRON, Aurélien. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow*. 3. ed. Alta Books, 2023. Cap. 1.
- RUSSELL, Stuart; NORVIG, Peter. *Inteligência Artificial: uma abordagem moderna*. 4. ed. GEN LTC, 2022. Cap. 19.
- Scikit-learn Documentation. [https://scikit-learn.org](https://scikit-learn.org)

---

*Próxima aula: [Aula 07 — Tipos de Aprendizado](aula-07-tipos-de-aprendizado.md)*
