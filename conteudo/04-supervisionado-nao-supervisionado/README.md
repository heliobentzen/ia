# Módulo 04: Aprendizado Supervisionado e Não Supervisionado

## Visão Geral

Este módulo apresenta os dois grandes paradigmas do aprendizado de máquina: o **aprendizado supervisionado**, onde o modelo aprende a partir de exemplos rotulados, e o **aprendizado não supervisionado**, onde o modelo descobre estruturas ocultas nos dados sem rótulos explícitos.

**Carga horária:** 5 aulas (Aulas 16 a 20) — 10 horas-aula

---

## Posição no Curso

| Módulo | Tema | Aulas |
|--------|------|-------|
| 01 | Fundamentos de IA | 1–4 |
| 02 | Paradigmas de Aprendizado | 5–8 |
| 03 | Preparação de Dados | 9–15 |
| **04** | **Supervisionado e Não Supervisionado** | **16–20** |
| 05 | Regressão e Classificação | 21–26 |
| ... | ... | ... |

---

## Aulas do Módulo

| Aula | Título | Principais Tópicos |
|------|--------|--------------------|
| 16 | Fundamentos do Aprendizado Supervisionado | Formalização, ERM, bias-variance, VC Dimension |
| 17 | K-Vizinhos Mais Próximos (KNN) | Métricas de distância, KD-Tree, weighted KNN |
| 18 | Fundamentos do Aprendizado Não Supervisionado | Clustering, redução de dim., métricas internas |
| 19 | Algoritmos de Clustering | K-Means, DBSCAN, GMM, Hierárquico |
| 20 | Redução de Dimensionalidade | PCA, t-SNE, UMAP, LDA, Autoencoder |

---

## Objetivos de Aprendizagem

Ao concluir este módulo, o estudante será capaz de:

1. **Formalizar** problemas de aprendizado supervisionado utilizando a linguagem matemática adequada (espaço de hipóteses, função de perda, ERM).
2. **Compreender** o dilema bias-variance e sua relação com a capacidade do modelo.
3. **Implementar** e avaliar o algoritmo KNN para classificação e regressão, escolhendo K por validação cruzada.
4. **Distinguir** aprendizado supervisionado de não supervisionado, identificando quando cada paradigma é mais adequado.
5. **Aplicar** algoritmos de clustering (K-Means, DBSCAN, GMM, hierárquico) e interpretar os resultados com métricas internas.
6. **Realizar** redução de dimensionalidade com PCA, t-SNE e UMAP, compreendendo as diferenças entre métodos lineares e não-lineares.

---

## Pré-requisitos

- Módulos 01–03 concluídos
- Álgebra linear: vetores, matrizes, autovalores/autovetores
- Cálculo: derivadas parciais, gradientes
- Python: NumPy, Pandas, Matplotlib, Scikit-learn básico

---

## Bibliografia

- FACELI, Katti et al. *Inteligência Artificial: uma abordagem de aprendizado de máquina*. 2. ed. LTC, 2021. Cap. 4–7.
- GÉRON, Aurélien. *Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow*. 3. ed. Alta Books, 2023. Cap. 5, 8, 9.
- RUSSELL, Stuart; NORVIG, Peter. *Inteligência Artificial: uma abordagem moderna*. 4. ed. GEN LTC, 2022. Cap. 19–20.

---

## Avaliação do Módulo

- **Lista de Exercícios 04:** Implementação de KNN do zero + análise de clustering (peso 25%)
- **Projeto Parcial:** Pipeline completo supervisionado + não supervisionado em dataset real (peso 35%)
- **Prova Modular:** Questões teóricas e práticas (peso 40%)
