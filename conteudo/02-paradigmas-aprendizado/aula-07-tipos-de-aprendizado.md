# Aula 07 — Tipos de Aprendizado

> **Módulo 02 · Aula 07 · Carga horária: 1 hora-aula**

---

## Objetivos da Aula

- Compreender e distinguir os cinco paradigmas de aprendizado de máquina.
- Identificar quando usar cada tipo de aprendizado.
- Conhecer os principais algoritmos de cada paradigma.
- Implementar exemplos práticos em Python para cada tipo de aprendizado.

---

## 1. Visão Geral dos Paradigmas

```
                    ┌─────────────────────────────────────────┐
                    │       APRENDIZADO DE MÁQUINA            │
                    └─────────────────────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
    ┌──────▼──────┐           ┌────────▼────────┐         ┌───────▼───────┐
    │ Supervisionado│          │Não Supervisionado│         │  Por Reforço  │
    │  (com rótulos)│          │  (sem rótulos)  │         │  (recompensa) │
    └──────┬──────┘           └────────┬────────┘         └───────────────┘
           │                           │
    ┌──────┴──────┐           ┌────────┴────────┐
    │Classificação│           │    Clustering   │
    │  Regressão  │           │Redução de Dim.  │
    └─────────────┘           │Detec. Anomalias │
                              └─────────────────┘

         ┌────────────────┐         ┌──────────────────────┐
         │  Semi-Superv.  │         │   Auto-Supervisionado │
         │(poucos rótulos)│         │  (GPT, BERT, CLIP)    │
         └────────────────┘         └──────────────────────┘
```

---

## 2. Aprendizado Supervisionado

### 2.1 Conceito

No aprendizado supervisionado, o algoritmo aprende a partir de um conjunto de exemplos **rotulados**: pares (X, y) onde X são as features de entrada e y é o rótulo (resposta correta).

> *"Supervisionado"* porque é como se houvesse um professor fornecendo as respostas corretas durante o treinamento.

O objetivo é aprender uma função f tal que f(X) ≈ y para novos dados nunca vistos.

### 2.2 Classificação

**Objetivo**: prever a qual **categoria discreta** uma instância pertence.

#### 2.2.1 Classificação Binária
Dois possíveis resultados (0 ou 1, sim ou não):
- Spam vs. não-spam
- Fraude vs. legítimo
- Tumor maligno vs. benigno
- Churn vs. não-churn

#### 2.2.2 Classificação Multiclasse
Três ou mais categorias mutuamente exclusivas:
- Reconhecimento de dígitos (0–9): 10 classes
- Classificação de espécies de flores (Iris): 3 classes
- Sentiment analysis: positivo / neutro / negativo
- Tipo de documento: fatura / contrato / e-mail / relatório

#### 2.2.3 Classificação Multilabel
Cada instância pode pertencer a **múltiplas** categorias simultaneamente:
- Tags de um artigo de notícia: ["economia", "tecnologia", "startups"]
- Diagnóstico médico: paciente pode ter diabetes + hipertensão + obesidade
- Gêneros de um filme: ["ação", "comédia", "romance"]

**Algoritmos populares para classificação:**

| Algoritmo | Características |
|-----------|----------------|
| Regressão Logística | Linear, interpretável, rápido |
| K-Nearest Neighbors | Não paramétrico, simples, lento em teste |
| Árvore de Decisão | Interpretável, propenso a overfitting |
| Random Forest | Ensemble, robusto, menos interpretável |
| SVM | Eficaz em alta dimensão, kernel trick |
| Redes Neurais | Flexível, requer muitos dados |
| Naive Bayes | Rápido, bom para texto |
| XGBoost | Estado da arte em dados tabulares |

### 2.3 Regressão

**Objetivo**: prever um **valor numérico contínuo**.

- Preço de imóveis (m², localização, quartos → R$)
- Temperatura futura (histórico meteorológico → °C)
- Consumo de energia (hora, temperatura, dia → kWh)
- Salário (experiência, formação, área → R$)

**Algoritmos populares para regressão:**

| Algoritmo | Características |
|-----------|----------------|
| Regressão Linear | Simples, interpretável, baseline |
| Regressão Polinomial | Captura não-linearidades |
| Ridge / Lasso | Regularização para evitar overfitting |
| SVR | SVM para regressão |
| Random Forest Regressor | Robusto a outliers |
| Gradient Boosting | XGBoost, LightGBM, alta performance |
| Redes Neurais | Qualquer função não-linear |

### 2.4 Quando Usar Aprendizado Supervisionado

✅ Você tem dados históricos com rótulos confiáveis.
✅ O objetivo é fazer previsões ou classificações sobre novos dados.
✅ Existe uma relação aprendível entre features e target.
❌ Rotular dados é muito caro ou demorado (considere semi-supervisionado).
❌ Não existe variável target clara (considere não-supervisionado).

---

## 3. Aprendizado Não Supervisionado

### 3.1 Conceito

Neste paradigma, o algoritmo recebe apenas as features X, **sem rótulos y**. O objetivo é descobrir estruturas, padrões ou representações intrínsecas nos dados.

### 3.2 Clustering (Agrupamento)

Agrupar instâncias semelhantes em **clusters** sem conhecimento prévio dos grupos.

#### 3.2.1 K-Means
- Algoritmo mais popular e simples.
- Requer especificar k (número de clusters) antecipadamente.
- Minimiza a soma das distâncias intra-cluster (inertia).
- Sensível a outliers e à escala dos dados.
- Complexidade: O(n · k · i · d) onde i = iterações, d = dimensões.

#### 3.2.2 DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
- Descobre clusters de forma arbitrária (não apenas esférica).
- Não requer k pré-definido.
- Identifica automaticamente pontos de ruído (outliers).
- Parâmetros: `eps` (raio de vizinhança) e `min_samples`.

#### 3.2.3 Clustering Hierárquico (Agglomerative)
- Constrói uma árvore de clusters (dendrograma).
- Abordagem *bottom-up*: começa com cada ponto como cluster, vai mesclando.
- Não requer k antecipadamente (pode ser cortado em qualquer nível).
- Complexidade: O(n² log n) — lento para grandes datasets.

**Aplicações de Clustering:**
- Segmentação de clientes (marketing personalizado)
- Agrupamento de documentos (organização de notícias)
- Compressão de imagens (k-means na paleta de cores)
- Descoberta de comunidades em redes sociais
- Análise de perfis genômicos

### 3.3 Redução de Dimensionalidade

Reduz o número de features preservando informação relevante.

#### 3.3.1 PCA (Principal Component Analysis)
- Técnica linear que encontra as direções de maior variância.
- Projeta os dados em eixos ortogonais (componentes principais).
- Perde a interpretabilidade das features originais.
- Ideal para: visualização, remoção de ruído, pré-processamento.

#### 3.3.2 t-SNE (t-Distributed Stochastic Neighbor Embedding)
- Técnica não-linear, ideal para **visualização** em 2D/3D.
- Preserva a estrutura local (vizinhos próximos ficam próximos).
- Não determinístico (resultados variam com `random_state`).
- Não adequado para produção (sem novo mapeamento de pontos fora da amostra).

#### 3.3.3 UMAP (Uniform Manifold Approximation and Projection)
- Alternativa ao t-SNE: mais rápido, preserva estrutura global.
- Pode ser usado tanto para visualização quanto para pré-processamento.

### 3.4 Detecção de Anomalias Não Supervisionada

Encontrar instâncias raras sem rótulos de anomalia:
- **Isolation Forest**: isola anomalias por partições aleatórias.
- **LOF (Local Outlier Factor)**: mede a densidade local em relação aos vizinhos.
- **Autoencoder**: reconstrói dados normais; alto erro de reconstrução = anomalia.

---

## 4. Aprendizado Semi-Supervisionado

### 4.1 Conceito

Situação muito comum na prática: **poucos dados rotulados** + **muitos dados não rotulados**.

Rotular dados é caro (requer especialistas humanos: médicos para exames, advogados para contratos, etc.). O aprendizado semi-supervisionado usa ambos para melhorar o modelo.

### 4.2 Técnicas Principais

#### 4.2.1 Self-Training (Auto-treinamento)
1. Treinar modelo com os dados rotulados.
2. Fazer previsões nos dados não rotulados.
3. Adicionar ao conjunto de treino as previsões com **alta confiança**.
4. Repetir até não haver mais ganho.

#### 4.2.2 Label Propagation
- Propaga rótulos dos pontos rotulados para os não rotulados com base na similaridade (grafo de vizinhança).
- Implementado em `sklearn.semi_supervised.LabelPropagation`.

#### 4.2.3 Co-Training
- Treina dois modelos em **visões diferentes** dos dados (ex.: texto em dois idiomas).
- Cada modelo rotula dados para o outro.

### 4.3 Exemplos de Aplicação

- **Reconhecimento facial**: poucas fotos rotuladas + muitas fotos não rotuladas do mesmo rosto.
- **NLP médico**: alguns relatórios médicos anotados + milhares sem anotação.
- **Detecção de objetos**: imagens anotadas são caras; usar poucas + muitas não anotadas.
- **Classificação de genes**: poucos genes com função conhecida, milhares desconhecidos.

---

## 5. Aprendizado por Reforço

### 5.1 Conceito

O agente aprende por **tentativa e erro**, interagindo com um ambiente e recebendo **recompensas** ou **punições** conforme suas ações.

```
         ┌─────────────────────────────────────┐
         │                                     │
    Ação │                                     │ Estado + Recompensa
         ▼                                     │
    ┌────────┐                           ┌─────┴─────┐
    │ Agente │ ◄────── Política ────────►│  Ambiente │
    └────────┘                           └───────────┘
```

**Componentes:**
- **Agente**: entidade que toma decisões.
- **Ambiente**: o mundo com o qual o agente interage.
- **Estado (s)**: situação atual do ambiente.
- **Ação (a)**: o que o agente pode fazer.
- **Recompensa (r)**: sinal de feedback imediato.
- **Política (π)**: estratégia do agente (mapeamento estado → ação).
- **Valor (V)**: recompensa acumulada esperada a longo prazo.

### 5.2 Algoritmos Principais

#### 5.2.1 Q-Learning
- Aprende uma função Q(s, a) que estima a recompensa futura.
- Usa a equação de Bellman para atualização iterativa.
- Tabular: funciona bem para espaços de estados discretos e pequenos.
- **Deep Q-Network (DQN)**: usa rede neural para aproximar Q(s, a).

#### 5.2.2 Policy Gradient
- Aprende diretamente a política π(a|s) (probabilidade de ação dado o estado).
- **REINFORCE**: gradiente do logaritmo da política × recompensa acumulada.
- Funciona bem para espaços de ações contínuos.

#### 5.2.3 PPO (Proximal Policy Optimization)
- Algoritmo moderno e estável (OpenAI, 2017).
- Usado no RLHF (Reinforcement Learning from Human Feedback) do ChatGPT.
- Evita atualizações de política muito grandes (clipping do gradiente).

#### 5.2.4 Actor-Critic (A3C, A2C, SAC)
- Combina value-based e policy-based.
- **Actor**: define a política.
- **Critic**: estima o valor do estado para guiar o actor.

### 5.3 Exemplos de Aplicação

- **Jogos**: AlphaGo (Go), AlphaStar (StarCraft II), OpenAI Five (Dota 2), jogos Atari (DQN).
- **Robótica**: manipulação de objetos, locomoção bípede, drones.
- **Veículos autônomos**: planejamento de trajetória em tempo real.
- **Finanças**: otimização de portfólios, execução de ordens de trading.
- **Otimização de data centers**: Google usa RL para otimizar resfriamento.
- **NLP**: RLHF para alinhar LLMs com preferências humanas.

---

## 6. Aprendizado Auto-Supervisionado

### 6.1 Conceito

O modelo cria **suas próprias tarefas de supervisão** a partir dos dados brutos, sem anotação humana. É a base dos grandes modelos de linguagem e visão atuais.

### 6.2 Técnicas Principais

#### 6.2.1 Modelagem de Linguagem Mascarada (BERT)
- Mascara aleatoriamente 15% dos tokens do texto.
- Treina o modelo para **prever os tokens mascarados**.
- Resultado: representações contextuais ricas (embeddings).

#### 6.2.2 Modelagem de Linguagem Causal (GPT)
- Treina o modelo para **prever o próximo token** em uma sequência.
- Aprendizado não supervisionado em bilhões de textos da internet.
- Resultado: modelo capaz de gerar texto coerente e responder perguntas.

#### 6.2.3 Contrastive Learning (SimCLR, MoCo, CLIP)
- Cria pares de augmentações da mesma imagem (positivos) e de imagens diferentes (negativos).
- Treina o modelo para aproximar representações de pares positivos e afastar negativos.
- **CLIP** (OpenAI): aprende correspondência entre imagens e texto.

### 6.3 Por Que É Revolucionário?
- Não precisa de anotação humana cara.
- Escala para bilhões/trilhões de parâmetros.
- Transfer learning poderoso: fine-tuning em poucas amostras.
- Base de GPT-4, DALL-E, Stable Diffusion, AlphaFold 2.

---

## 7. Diagrama Comparativo

| Paradigma | Dados Necessários | Objetivo | Exemplos de Algoritmos |
|-----------|-------------------|----------|----------------------|
| **Supervisionado** | Rotulados (X, y) | Previsão/Classificação | Regressão Logística, SVM, Random Forest, XGBoost |
| **Não Supervisionado** | Sem rótulos (X) | Estrutura/Padrões | K-Means, DBSCAN, PCA, t-SNE |
| **Semi-Supervisionado** | Poucos rotulados + Muitos sem rótulo | Previsão com dados escassos | Self-Training, Label Propagation |
| **Por Reforço** | Recompensas do ambiente | Política ótima | Q-Learning, DQN, PPO, A3C |
| **Auto-Supervisionado** | Dados brutos (sem anotação) | Representações ricas | BERT, GPT, CLIP, SimCLR |

---

## 8. Exemplos em Python

### 8.1 Supervisionado — Classificação e Regressão

```python
# =============================================================
# Aprendizado Supervisionado: Classificação e Regressão
# =============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score

# --- CLASSIFICAÇÃO BINÁRIA: Diagnóstico de Câncer ---
print("=" * 55)
print("CLASSIFICAÇÃO BINÁRIA — Diagnóstico de Câncer de Mama")
print("=" * 55)

cancer = load_breast_cancer()
X_c, y_c = cancer.data, cancer.target
print(f"Classes: {cancer.target_names}  | Shape: {X_c.shape}")

X_tr, X_ts, y_tr, y_ts = train_test_split(X_c, y_c, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_ts_s  = scaler.transform(X_ts)

# Regressão Logística
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_tr_s, y_tr)
y_pred_lr = lr.predict(X_ts_s)
y_prob_lr = lr.predict_proba(X_ts_s)[:, 1]

print(f"Regressão Logística — Acurácia: {accuracy_score(y_ts, y_pred_lr):.4f} | AUC: {roc_auc_score(y_ts, y_prob_lr):.4f}")

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_tr_s, y_tr)
y_pred_rf = rf_clf.predict(X_ts_s)
y_prob_rf = rf_clf.predict_proba(X_ts_s)[:, 1]

print(f"Random Forest    — Acurácia: {accuracy_score(y_ts, y_pred_rf):.4f} | AUC: {roc_auc_score(y_ts, y_prob_rf):.4f}")

# --- REGRESSÃO: Previsão com dados sintéticos ---
print("\n" + "=" * 55)
print("REGRESSÃO — Previsão de Valor (dados sintéticos)")
print("=" * 55)

# Criar dataset sintético de regressão
np.random.seed(42)
n = 300
X_reg = np.random.randn(n, 5)
y_reg = 3*X_reg[:, 0] - 2*X_reg[:, 1] + 0.5*X_reg[:, 2]**2 + np.random.randn(n) * 0.5

X_rtr, X_rts, y_rtr, y_rts = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Regressão Linear
lin = LinearRegression()
lin.fit(X_rtr, y_rtr)
y_pred_lin = lin.predict(X_rts)
rmse_lin = np.sqrt(mean_squared_error(y_rts, y_pred_lin))
r2_lin = r2_score(y_rts, y_pred_lin)
print(f"Regressão Linear   — RMSE: {rmse_lin:.4f} | R²: {r2_lin:.4f}")

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_rtr, y_rtr)
y_pred_rfr = rf_reg.predict(X_rts)
rmse_rfr = np.sqrt(mean_squared_error(y_rts, y_pred_rfr))
r2_rfr = r2_score(y_rts, y_pred_rfr)
print(f"Random Forest Reg. — RMSE: {rmse_rfr:.4f} | R²: {r2_rfr:.4f}")
```

### 8.2 Não Supervisionado — Clustering e Redução de Dimensionalidade

```python
# =============================================================
# Aprendizado Não Supervisionado: K-Means, DBSCAN, PCA, t-SNE
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, load_digits
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# --- K-MEANS ---
print("=" * 50)
print("CLUSTERING — K-Means")
print("=" * 50)

# Dataset com 4 clusters naturais
X_blobs, y_true = make_blobs(
    n_samples=400, centers=4, cluster_std=0.8,
    random_state=42
)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_blobs)

# Escolha de k usando o Método do Cotovelo (Elbow Method)
inercias = []
silhuetas = []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inercias.append(km.inertia_)
    silhuetas.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(K_range, inercias, 'o-', color='steelblue')
ax1.set(xlabel='Número de Clusters (k)', ylabel='Inércia', title='Método do Cotovelo')
ax1.axvline(x=4, color='red', linestyle='--', label='k=4 (ideal)')
ax1.legend()

ax2.plot(K_range, silhuetas, 's-', color='darkorange')
ax2.set(xlabel='Número de Clusters (k)', ylabel='Coeficiente de Silhueta', title='Análise de Silhueta')
ax2.axvline(x=4, color='red', linestyle='--', label='k=4 (ideal)')
ax2.legend()
plt.tight_layout()
plt.savefig("kmeans_elbow.png", dpi=150, bbox_inches='tight')
plt.show()

# K-Means final com k=4
km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
labels_km = km_final.fit_predict(X_scaled)
ari_km = adjusted_rand_score(y_true, labels_km)
sil_km = silhouette_score(X_scaled, labels_km)
print(f"K-Means (k=4) — ARI: {ari_km:.4f} | Silhueta: {sil_km:.4f}")

# --- DBSCAN ---
print("\n" + "=" * 50)
print("CLUSTERING — DBSCAN (forma não esférica)")
print("=" * 50)

# Dataset com forma de lua crescente (não linearmente separável)
X_moons, y_moons = make_moons(n_samples=300, noise=0.05, random_state=42)
X_moons_s = StandardScaler().fit_transform(X_moons)

db = DBSCAN(eps=0.3, min_samples=5)
labels_db = db.fit_predict(X_moons_s)
n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = list(labels_db).count(-1)

print(f"DBSCAN — Clusters encontrados: {n_clusters} | Ruídos: {n_noise}")
if n_clusters > 1:
    sil_db = silhouette_score(X_moons_s, labels_db)
    print(f"DBSCAN — Coeficiente de Silhueta: {sil_db:.4f}")

# --- PCA ---
print("\n" + "=" * 50)
print("REDUÇÃO DE DIMENSIONALIDADE — PCA")
print("=" * 50)

# Dataset de dígitos manuscritos (64 features → 2D)
digits = load_digits()
X_digits = digits.data  # 1797 amostras, 64 features

# Variância explicada por componente
pca_full = PCA()
pca_full.fit(StandardScaler().fit_transform(X_digits))

variancia_acumulada = np.cumsum(pca_full.explained_variance_ratio_)
n_componentes_95 = np.argmax(variancia_acumulada >= 0.95) + 1
print(f"Componentes para 95% da variância: {n_componentes_95}")

# PCA para 2D (visualização)
pca_2d = PCA(n_components=2, random_state=42)
X_pca_2d = pca_2d.fit_transform(StandardScaler().fit_transform(X_digits))
print(f"Variância explicada pelos 2 primeiros PCs: {pca_2d.explained_variance_ratio_.sum():.2%}")

# --- t-SNE ---
print("\n" + "=" * 50)
print("REDUÇÃO DE DIMENSIONALIDADE — t-SNE")
print("=" * 50)

# t-SNE para visualização dos dígitos
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne_2d = tsne.fit_transform(StandardScaler().fit_transform(digits.data))

# Visualização PCA vs t-SNE
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
cores = plt.cm.tab10(np.linspace(0, 1, 10))

for digito in range(10):
    mask = digits.target == digito
    ax1.scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                c=[cores[digito]], label=str(digito), s=10, alpha=0.7)
    ax2.scatter(X_tsne_2d[mask, 0], X_tsne_2d[mask, 1],
                c=[cores[digito]], label=str(digito), s=10, alpha=0.7)

ax1.set(title="PCA — Dígitos Manuscritos (2D)", xlabel="PC1", ylabel="PC2")
ax2.set(title="t-SNE — Dígitos Manuscritos (2D)", xlabel="t-SNE 1", ylabel="t-SNE 2")
ax1.legend(title="Dígito", loc='upper right', markerscale=2)
ax2.legend(title="Dígito", loc='upper right', markerscale=2)
plt.tight_layout()
plt.savefig("pca_tsne_digits.png", dpi=150, bbox_inches='tight')
plt.show()
```

### 8.3 Semi-Supervisionado — Self-Training

```python
# =============================================================
# Aprendizado Semi-Supervisionado — Self-Training com Label Propagation
# =============================================================

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

digits = load_iris_data = load_digits()
X, y = digits.data, digits.target

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train_full)
X_test_s  = scaler.transform(X_test)

# --- Cenário: apenas 10% dos dados de treino são rotulados ---
n_rotulados = int(0.10 * len(X_train_full))
indices_rotulados = np.random.RandomState(42).choice(
    len(X_train_full), n_rotulados, replace=False
)

# Máscara: -1 indica "sem rótulo"
y_parcial = np.full_like(y_train_full, fill_value=-1)
y_parcial[indices_rotulados] = y_train_full[indices_rotulados]

print(f"Amostras de treino total: {len(X_train_full)}")
print(f"Amostras rotuladas:       {n_rotulados} ({100*n_rotulados/len(X_train_full):.0f}%)")
print(f"Amostras sem rótulo:      {(y_parcial == -1).sum()}")

# --- Modelo baseline: apenas com dados rotulados ---
lr_baseline = LogisticRegression(max_iter=1000, random_state=42)
lr_baseline.fit(X_train_s[indices_rotulados], y_train_full[indices_rotulados])
acc_baseline = accuracy_score(y_test, lr_baseline.predict(X_test_s))
print(f"\nBaseline (só rotulados): {acc_baseline:.4f}")

# --- Label Propagation (semi-supervisionado) ---
lp = LabelPropagation(kernel='knn', n_neighbors=7, max_iter=1000)
lp.fit(X_train_s, y_parcial)
acc_lp = accuracy_score(y_test, lp.predict(X_test_s))
print(f"Label Propagation:       {acc_lp:.4f}")

# --- Label Spreading ---
ls = LabelSpreading(kernel='knn', n_neighbors=7, alpha=0.2)
ls.fit(X_train_s, y_parcial)
acc_ls = accuracy_score(y_test, ls.predict(X_test_s))
print(f"Label Spreading:         {acc_ls:.4f}")

# --- Comparação ---
print(f"\nGanho (LP vs Baseline): +{(acc_lp - acc_baseline)*100:.1f} pontos percentuais")
```

### 8.4 Aprendizado por Reforço — Q-Learning no FrozenLake

```python
# =============================================================
# Aprendizado por Reforço — Q-Learning no ambiente FrozenLake
# (requer: pip install gymnasium)
# =============================================================

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Criar ambiente FrozenLake 4x4
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode=None)
n_estados  = env.observation_space.n   # 16 estados (4x4)
n_acoes    = env.action_space.n        # 4 ações: esquerda, baixo, direita, cima

# --- Inicializar Q-Table (todos os zeros) ---
Q = np.zeros((n_estados, n_acoes))

# --- Hiperparâmetros ---
alpha       = 0.8    # taxa de aprendizado
gamma       = 0.95   # fator de desconto
epsilon     = 1.0    # exploração inicial (epsilon-greedy)
eps_decay   = 0.995  # decaimento do epsilon por episódio
eps_min     = 0.01   # epsilon mínimo
n_episodios = 5000

recompensas = []

# --- Loop de treinamento ---
for ep in range(n_episodios):
    estado, _ = env.reset()
    recompensa_total = 0
    terminado = False

    while not terminado:
        # Política epsilon-greedy
        if np.random.random() < epsilon:
            acao = env.action_space.sample()  # exploração aleatória
        else:
            acao = np.argmax(Q[estado])       # exploração gananciosa

        prox_estado, recompensa, terminado, truncado, _ = env.step(acao)

        # Equação de Bellman: atualização da Q-Table
        Q[estado, acao] = Q[estado, acao] + alpha * (
            recompensa + gamma * np.max(Q[prox_estado]) - Q[estado, acao]
        )

        estado = prox_estado
        recompensa_total += recompensa

    recompensas.append(recompensa_total)
    epsilon = max(eps_min, epsilon * eps_decay)

# --- Avaliação da política aprendida ---
n_avaliacoes = 100
sucessos = 0
for _ in range(n_avaliacoes):
    estado, _ = env.reset()
    terminado = False
    while not terminado:
        acao = np.argmax(Q[estado])
        estado, recompensa, terminado, truncado, _ = env.step(acao)
    if recompensa == 1.0:
        sucessos += 1

taxa_sucesso = sucessos / n_avaliacoes
print(f"Q-Learning — Taxa de sucesso após {n_episodios} episódios: {taxa_sucesso:.0%}")

# Progresso de aprendizado
janela = 100
media_movel = [np.mean(recompensas[i:i+janela]) for i in range(0, len(recompensas)-janela)]
plt.figure(figsize=(10, 4))
plt.plot(media_movel, color='steelblue')
plt.axhline(y=0.9, color='red', linestyle='--', label='Meta: 90%')
plt.xlabel('Episódio')
plt.ylabel('Recompensa Média (janela=100)')
plt.title('Aprendizado do Agente Q-Learning — FrozenLake')
plt.legend()
plt.tight_layout()
plt.savefig("rl_qlearning.png", dpi=150, bbox_inches='tight')
plt.show()

env.close()
```

---

## 9. Quando Usar Cada Paradigma

```
Tenho dados rotulados?
├── Sim → Aprendizado SUPERVISIONADO
│         ├── Target contínuo? → REGRESSÃO
│         └── Target categórico? → CLASSIFICAÇÃO
│
├── Não → Aprendizado NÃO SUPERVISIONADO
│         ├── Agrupar instâncias? → CLUSTERING
│         ├── Reduzir dimensões?  → REDUÇÃO DE DIMENSIONALIDADE
│         └── Detectar anomalias? → DETECÇÃO DE ANOMALIAS
│
├── Poucos rotulados + Muitos sem rótulo?
│   └── Aprendizado SEMI-SUPERVISIONADO
│
├── Problema de decisão sequencial com recompensas?
│   └── Aprendizado POR REFORÇO
│
└── Dados brutos em larga escala, quero representações?
    └── Aprendizado AUTO-SUPERVISIONADO (pré-treino de LLMs)
```

---

## 10. Exercícios

1. **Conceitual**: Classifique cada problema abaixo em supervisionado, não supervisionado ou por reforço:
   - (a) Identificar grupos de clientes com comportamento similar.
   - (b) Treinar um robô para caminhar.
   - (c) Prever o salário de um profissional dadas suas características.
   - (d) Organizar automaticamente um arquivo de fotos.
   - (e) Detectar transações fraudulentas com base em dados históricos rotulados.

2. **Prático**: Use o dataset `make_blobs` para criar 3 clusters com 500 amostras. Teste K-Means com k=2, 3, 4, 5. Compare as métricas de silhueta. Visualize os resultados.

3. **Pesquisa**: O AlphaGo usou aprendizado por reforço combinado com outras técnicas. Pesquise e descreva o pipeline completo: quais tipos de aprendizado foram usados e como foram combinados?

4. **Desafio**: Implemente Self-Training manualmente: treine com 5% dos rótulos, faça previsões nos demais, adicione as previsões com probabilidade > 0.9 ao conjunto de treino, re-treine e meça a melhora.

---

## 11. Referências

- FACELI, Katti et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. 2. ed. LTC, 2021. Cap. 2.
- GÉRON, Aurélien. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow*. 3. ed. Alta Books, 2023. Cap. 1–3, 9.
- SUTTON, Richard S.; BARTO, Andrew G. *Reinforcement Learning: An Introduction*. 2. ed. MIT Press, 2018.
- RUSSELL, Stuart; NORVIG, Peter. *Inteligência Artificial: uma abordagem moderna*. 4. ed. GEN LTC, 2022. Caps. 19, 22.

---

*← [Aula 06 — O Que é ML?](aula-06-o-que-e-ml.md) | [Aula 08 — Fluxo de um Projeto de ML](aula-08-fluxo-projeto-ml.md) →*
