# Módulo 03 — Preparação e Análise de Dados

## Visão Geral

| Item              | Detalhe                                      |
|-------------------|----------------------------------------------|
| **Módulo**        | 03 — Preparação e Análise de Dados           |
| **Aulas**         | 10 a 15                                      |
| **Carga horária** | 6 horas-aula (3 semanas)                     |
| **Pré-requisitos**| Módulos 01 e 02                              |

---

## Aulas do Módulo

| Aula | Título | Arquivo |
|------|--------|---------|
| 10 | Coleta de Dados e Análise Exploratória (EDA) | `aula-10-coleta-e-eda.md` |
| 11 | Qualidade e Limpeza de Dados | `aula-11-qualidade-limpeza.md` |
| 12 | Feature Engineering | `aula-12-feature-engineering.md` |
| 13 | Transformações e Normalização | `aula-13-transformacoes-normalizacao.md` |
| 14 | Seleção de Atributos | `aula-14-selecao-atributos.md` |
| 15 | Pipeline de Dados com Scikit-learn | `aula-15-pipeline-dados.md` |

---

## Objetivos de Aprendizagem

Ao final deste módulo, o aluno será capaz de:

1. Coletar dados de diversas fontes e realizar EDA sistemática.
2. Identificar e tratar problemas de qualidade de dados.
3. Criar features relevantes a partir de dados brutos.
4. Aplicar transformações e normalizações adequadas.
5. Selecionar atributos usando métodos filter, wrapper e embedded.
6. Construir pipelines de pré-processamento reprodutíveis com scikit-learn.

---

## Pré-requisitos

- Módulos 01 e 02 concluídos.
- Python com: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

---

## Estrutura de Tópicos

```
Módulo 03
├── Aula 10 — EDA
│   ├── Fontes de dados
│   ├── Tipos de variáveis
│   ├── Estatísticas descritivas
│   └── Visualizações exploratórias
├── Aula 11 — Qualidade e Limpeza
│   ├── Valores ausentes (MCAR/MAR/MNAR)
│   ├── Outliers
│   └── Duplicatas e inconsistências
├── Aula 12 — Feature Engineering
│   ├── Encoding categórico
│   ├── Features temporais
│   └── Interações e domínio
├── Aula 13 — Normalização
│   ├── MinMax, StandardScaler, RobustScaler
│   └── Transformações log, Box-Cox, Yeo-Johnson
├── Aula 14 — Seleção de Atributos
│   ├── Métodos Filter, Wrapper, Embedded
│   └── VIF e colinearidade
└── Aula 15 — Pipelines
    ├── Pipeline + ColumnTransformer
    ├── Custom transformers
    └── Serialização
```

---

## Bibliografia

- **FACELI et al.** Inteligência Artificial: uma abordagem de aprendizado de máquina. 2. ed. LTC, 2021. Caps. 3–5.
- **GÉRON, Aurélien.** Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow. 3. ed. Alta Books, 2023. Caps. 2–3.
- **RUSSELL; NORVIG.** Inteligência Artificial: uma abordagem moderna. 4. ed. GEN LTC, 2022.
