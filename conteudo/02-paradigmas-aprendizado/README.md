# Módulo 02 — Paradigmas de Aprendizado de Máquina

## Visão Geral

Este módulo apresenta os conceitos fundamentais do Aprendizado de Máquina (Machine Learning), explorando o que é ML, os diferentes paradigmas de aprendizado, o fluxo completo de um projeto de ML e o ecossistema de ferramentas amplamente utilizado pela comunidade científica e pela indústria.

| Item              | Detalhe                                      |
|-------------------|----------------------------------------------|
| **Módulo**        | 02 — Paradigmas de Aprendizado de Máquina    |
| **Aulas**         | 06, 07, 08 e 09                              |
| **Carga horária** | 4 horas-aula (2 semanas)                     |
| **Pré-requisitos**| Módulo 01 — Fundamentos de IA                |

---

## Aulas do Módulo

| Aula | Título                                        | Arquivo                          |
|------|-----------------------------------------------|----------------------------------|
| 06   | O Que é Aprendizado de Máquina?               | `aula-06-o-que-e-ml.md`          |
| 07   | Tipos de Aprendizado                          | `aula-07-tipos-de-aprendizado.md`|
| 08   | Fluxo de um Projeto de Machine Learning       | `aula-08-fluxo-projeto-ml.md`    |
| 09   | Ecossistema de Ferramentas de ML              | `aula-09-ecossistema-ferramentas.md` |

---

## Objetivos de Aprendizagem

Ao final deste módulo, o aluno será capaz de:

1. **Definir** formalmente o que é Aprendizado de Máquina e distingui-lo da programação tradicional.
2. **Identificar** e **classificar** os principais paradigmas de aprendizado (supervisionado, não supervisionado, semi-supervisionado, por reforço e auto-supervisionado).
3. **Descrever** as etapas de um projeto de ML seguindo a metodologia CRISP-DM.
4. **Reconhecer** as principais ferramentas do ecossistema Python para ML e selecionar a mais adequada para cada cenário.
5. **Implementar** exemplos básicos de ML utilizando scikit-learn e interpretar seus resultados.

---

## Pré-requisitos

- Conclusão do **Módulo 01 — Fundamentos de IA**.
- Conhecimentos básicos de **Python** (variáveis, funções, listas, dicionários).
- Noções de **Álgebra Linear** (vetores, matrizes) e **Estatística** (média, variância, distribuições).
- Ambiente Python configurado com: `numpy`, `pandas`, `matplotlib`, `scikit-learn`.

---

## Estrutura de Tópicos

```
Módulo 02
├── Aula 06 — O Que é ML?
│   ├── Definição formal (Mitchell, 1997)
│   ├── ML vs. Programação Tradicional
│   ├── Breve história do ML
│   ├── Aplicações reais
│   ├── Tipos de tarefas
│   ├── Por que ML agora?
│   └── Limitações e desafios
│
├── Aula 07 — Tipos de Aprendizado
│   ├── Aprendizado Supervisionado
│   │   ├── Classificação (binária, multiclasse, multilabel)
│   │   └── Regressão (contínua, ordinal)
│   ├── Aprendizado Não Supervisionado
│   │   ├── Clustering
│   │   └── Redução de dimensionalidade
│   ├── Aprendizado Semi-Supervisionado
│   ├── Aprendizado por Reforço
│   └── Aprendizado Auto-Supervisionado
│
├── Aula 08 — Fluxo de um Projeto de ML
│   ├── Metodologia CRISP-DM
│   ├── Definição do problema
│   ├── Coleta e entendimento dos dados
│   ├── Preparação dos dados
│   ├── Modelagem iterativa
│   ├── Avaliação e validação
│   ├── Deploy e monitoramento
│   └── Versionamento e boas práticas
│
└── Aula 09 — Ecossistema de Ferramentas
    ├── Scikit-learn
    ├── TensorFlow/Keras
    ├── PyTorch
    ├── HuggingFace
    ├── Gradient Boosting (XGBoost, LightGBM, CatBoost)
    ├── MLflow
    ├── Pandas, NumPy, Visualização
    ├── Ambientes de desenvolvimento
    └── Plataformas de nuvem
```

---

## Bibliografia do Módulo

- **FACELI, Katti et al.** Inteligência Artificial: uma abordagem de aprendizado de máquina. 2. ed. LTC, 2021. Capítulos 1 e 2.
- **GÉRON, Aurélien.** Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow. 3. ed. Alta Books, 2023. Capítulos 1 e 2.
- **RUSSELL, Stuart; NORVIG, Peter.** Inteligência Artificial: uma abordagem moderna. 4. ed. GEN LTC, 2022. Capítulo 19.

---

## Avaliação

| Instrumento            | Peso | Descrição                                           |
|------------------------|------|-----------------------------------------------------|
| Quiz Aulas 06–07       | 20%  | 10 questões conceituais sobre tipos de aprendizado  |
| Quiz Aulas 08–09       | 20%  | 10 questões sobre CRISP-DM e ferramentas            |
| Atividade Prática      | 60%  | Implementar pipeline básico de ML com scikit-learn  |

---

## Recursos Adicionais

- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Google Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Fast.ai Practical Deep Learning](https://course.fast.ai/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
