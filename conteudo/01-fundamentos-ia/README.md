# Módulo 01 — Fundamentos de Inteligência Artificial

> **Curso:** Inteligência Artificial e Aprendizado de Máquina  
> **Carga horária do módulo:** 5 aulas × 45 minutos = 3h45min  
> **Posição no curso:** Módulo 1 de 12 (60 horas-aula totais)  
> **Nível:** Graduação / Pós-Graduação Lato Sensu

---

## Sumário

- [Apresentação do Módulo](#apresentação-do-módulo)
- [Objetivos de Aprendizagem](#objetivos-de-aprendizagem)
- [Pré-requisitos](#pré-requisitos)
- [Estrutura das Aulas](#estrutura-das-aulas)
- [Resultados Esperados](#resultados-esperados)
- [Metodologia](#metodologia)
- [Avaliação](#avaliação)
- [Recursos e Ferramentas](#recursos-e-ferramentas)
- [Referências Bibliográficas](#referências-bibliográficas)
- [Cronograma Sugerido](#cronograma-sugerido)

---

## Apresentação do Módulo

O **Módulo 01 — Fundamentos de Inteligência Artificial** constitui a base conceitual e prática sobre a qual todo o curso se apoia. Antes de mergulhar nos algoritmos de aprendizado de máquina, nas redes neurais profundas e nas aplicações modernas de IA, é imprescindível compreender:

1. **De onde viemos** — a história da IA, seus sucessos, fracassos e renascimentos.
2. **O que é** — definições precisas, distinções entre IA Estreita, Geral e Super-Humana.
3. **Como os agentes funcionam** — o modelo arquitetural que unifica todo sistema inteligente.
4. **Como representamos o conhecimento** — lógica, ontologias, redes bayesianas.
5. **Qual a ferramenta** — Python e seu ecossistema científico, indispensável para todas as aulas seguintes.

A Inteligência Artificial deixou de ser um campo exclusivamente acadêmico. Em 2024, sistemas como o GPT-4, Gemini Ultra, Claude 3 e Llama 3 são utilizados por centenas de milhões de pessoas diariamente. Veículos semi-autônomos percorrem estradas, algoritmos diagnosticam cânceres com precisão comparável à de radiologistas especialistas, e robôs industriais aprendem tarefas com poucos exemplos. Compreender os fundamentos que sustentam esses sistemas é o primeiro passo para criar, criticar e aprimorá-los.

Este módulo adota uma abordagem **histórica-conceitual-prática**: partimos da história para contextualizar, apresentamos os conceitos com rigor, e ancoramos tudo em código Python funcional que o estudante pode executar imediatamente.

---

## Objetivos de Aprendizagem

Ao final deste módulo, o estudante será capaz de:

### Conhecimento (Lembrar e Compreender)
- [ ] Descrever os principais marcos históricos da IA desde 1950 até os dias atuais.
- [ ] Definir os conceitos de racionalidade, agente, ambiente e percepção no contexto de IA.
- [ ] Distinguir entre IA Estreita (ANI), IA Geral (AGI) e IA Super-Humana (ASI).
- [ ] Explicar as quatro principais abordagens da IA: simbólica, conexionista, evolucionária e bayesiana.
- [ ] Descrever a estrutura PEAS de um agente inteligente.
- [ ] Enunciar os principais algoritmos de busca informada e não-informada.
- [ ] Listar as principais bibliotecas Python utilizadas em ciência de dados e IA.

### Aplicação (Aplicar e Analisar)
- [ ] Classificar diferentes sistemas de IA segundo o tipo de agente e de ambiente.
- [ ] Aplicar o framework PEAS para especificar agentes para problemas reais.
- [ ] Selecionar o algoritmo de busca apropriado dado um problema, justificando a escolha.
- [ ] Utilizar NumPy, Pandas, Matplotlib e Scikit-learn para tarefas básicas de análise de dados.
- [ ] Implementar um agente reflexivo simples em Python.
- [ ] Implementar algoritmos de busca BFS, DFS e A* em Python.

### Síntese e Avaliação
- [ ] Discutir as implicações éticas do desenvolvimento e implantação de sistemas de IA.
- [ ] Comparar abordagens simbólicas e conexionistas, identificando vantagens e limitações de cada uma.
- [ ] Avaliar criticamente afirmações sobre capacidades de sistemas de IA atuais.
- [ ] Propor soluções baseadas em agentes inteligentes para problemas do cotidiano.

---

## Pré-requisitos

### Conhecimentos Obrigatórios
- **Programação em Python** (nível básico-intermediário): variáveis, funções, classes, listas, dicionários, compreensão de listas, leitura/escrita de arquivos.
- **Matemática básica**: funções, conjuntos, probabilidade elementar (regra de Bayes, probabilidade condicional).
- **Lógica**: noções de proposição, conjunção, disjunção, negação, implicação.

### Conhecimentos Recomendados (não obrigatórios)
- Álgebra linear básica: vetores, matrizes, multiplicação matricial (será revisado na Aula 05).
- Estatística descritiva: média, mediana, desvio padrão, quartis.
- Experiência com ambientes Linux ou macOS (para configuração de ambiente).

### Ferramentas Necessárias
- Python 3.9+ instalado (recomendado via Anaconda ou Miniconda).
- Editor de código: VS Code, PyCharm ou Jupyter Lab.
- Conta no Google Colab (alternativa gratuita para quem não tem GPU/CPU adequada).

---

## Estrutura das Aulas

### Aula 01 — História da IA e Conceitos Fundamentais
**Arquivo:** [`aula-01-historia-e-conceitos.md`](./aula-01-historia-e-conceitos.md)  
**Duração:** 45 minutos  
**Tópicos principais:**
- O que é Inteligência Artificial? Definições e debates.
- Linha do tempo: de Turing (1950) aos LLMs (2024).
- Teste de Turing e Quarto Chinês de Searle.
- Tipos de IA: ANI, AGI, ASI.
- Abordagens: simbólica, conexionista, evolucionária, bayesiana.
- Invernos da IA, Deep Learning Revolution, Era dos Transformers.
- Aplicações atuais e questões éticas iniciais.

**Código prático:** Conceito de chatbot simples em Python (baseado em regras).

---

### Aula 02 — Agentes Inteligentes
**Arquivo:** [`aula-02-agentes-inteligentes.md`](./aula-02-agentes-inteligentes.md)  
**Duração:** 45 minutos  
**Tópicos principais:**
- Definição de agente: percepção, ação, função agente, racionalidade.
- Framework PEAS: Performance, Environment, Actuators, Sensors.
- Tipos de agentes: reflexivo simples, baseado em modelo, baseado em objetivos, baseado em utilidade, agente que aprende.
- Tipos de ambientes e suas propriedades.
- Exemplos: Roomba, assistente virtual, carro autônomo, xadrez.

**Código prático:** Implementação de agente reflexivo simples em Python.

---

### Aula 03 — Busca e Resolução de Problemas
**Arquivo:** [`aula-03-busca-e-resolucao-problemas.md`](./aula-03-busca-e-resolucao-problemas.md)  
**Duração:** 45 minutos  
**Tópicos principais:**
- Formulação de problemas como espaço de estados.
- Busca não-informada: BFS, DFS, Custo Uniforme, Aprofundamento Iterativo.
- Busca informada: Busca Gulosa, A*, IDA*.
- Heurísticas: admissibilidade e consistência.
- Aplicações: quebra-cabeça de 8 peças, labirinto, N-rainhas.
- Tabela comparativa de algoritmos.

**Código prático:** BFS, DFS e A* implementados em Python com exemplos.

---

### Aula 04 — Representação do Conhecimento
**Arquivo:** [`aula-04-representacao-conhecimento.md`](./aula-04-representacao-conhecimento.md)  
**Duração:** 45 minutos  
**Tópicos principais:**
- Lógica proposicional e de primeira ordem (FOL).
- Redes semânticas e frames.
- Ontologias: OWL, RDF, linked data.
- Sistemas baseados em regras: encadeamento para frente e para trás.
- Redes bayesianas: estrutura e inferência.
- Sistema especialista médico simplificado.

**Código prático:** Sistema baseado em regras em Python (motor de inferência simples).

---

### Aula 05 — Python para IA e Ciência de Dados
**Arquivo:** [`aula-05-python-para-ia.md`](./aula-05-python-para-ia.md)  
**Duração:** 45 minutos  
**Tópicos principais:**
- Por que Python? Vantagens e ecossistema.
- Configuração de ambiente: Anaconda, conda, pip, virtualenv.
- NumPy: arrays N-dimensionais, broadcasting, vetorização.
- Pandas: DataFrames, limpeza de dados, análise exploratória.
- Matplotlib e Seaborn: visualizações para ciência de dados.
- Scikit-learn: API de estimadores, pipelines, transformadores.
- Boas práticas com Jupyter Notebooks.

**Código prático:** Exemplos completos e executáveis para cada biblioteca.

---

## Resultados Esperados

### Competências Técnicas
Ao concluir este módulo, o estudante terá desenvolvido as seguintes competências:

1. **Pensamento Computacional Orientado a Agentes**: Capacidade de modelar qualquer problema do mundo real como um agente interagindo com um ambiente, especificando claramente sensores, atuadores, medida de desempenho e propriedades do ambiente.

2. **Domínio do Histórico e Contexto da IA**: Compreensão profunda de por que certas abordagens falharam ou tiveram sucesso em diferentes épocas, permitindo análise crítica de tendências atuais.

3. **Habilidade em Algoritmos de Busca**: Capacidade de selecionar e implementar o algoritmo de busca mais adequado para um problema dado, compreendendo as trocas entre completude, otimalidade e eficiência computacional.

4. **Base em Representação do Conhecimento**: Entendimento dos diferentes formalismos para representar conhecimento, habilitando o estudo de sistemas especialistas e IA simbólica.

5. **Proficiência no Ecossistema Python para IA**: Capacidade de utilizar NumPy, Pandas, Matplotlib, Seaborn e Scikit-learn para manipulação de dados, visualização e prototipagem de modelos de ML.

### Competências Transversais
- **Pensamento crítico** em relação a afirmações sobre capacidades de IA.
- **Consciência ética** sobre os impactos sociais de sistemas inteligentes.
- **Autonomia** para aprender novas ferramentas e bibliotecas do ecossistema Python.

---

## Metodologia

### Estrutura de Cada Aula (45 minutos)
| Fase | Tempo | Atividade |
|------|-------|-----------|
| Abertura | 5 min | Revisão da aula anterior, contextualização |
| Conteúdo teórico | 20 min | Exposição dialogada com slides/quadro |
| Demonstração prática | 10 min | Live coding ou análise de código |
| Atividade guiada | 7 min | Estudantes executam/adaptam o código |
| Síntese e reflexão | 3 min | Questões para casa, próximos passos |

### Abordagem Pedagógica
- **Aprendizado Ativo**: Intercalação de teoria e prática em cada aula.
- **Contextualização**: Todos os conceitos são ancorados em exemplos reais e contemporâneos.
- **Scaffolding**: Os conceitos são introduzidos gradualmente, cada aula construindo sobre as anteriores.
- **Metacognição**: Questões de reflexão ao final de cada aula para consolidar o aprendizado.

---

## Avaliação

### Avaliação Formativa (durante o módulo)
- **Participação nas reflexões** (perguntas ao final de cada aula): 20%
- **Exercícios práticos de código** (implementações guiadas): 30%

### Avaliação Somativa (ao final do módulo)
- **Mini-projeto**: Especificar e implementar um agente inteligente simples para um problema de sua escolha, utilizando Python. O agente deve ser especificado via PEAS e implementado com no mínimo dois dos tipos estudados (reflexivo e baseado em modelo). Incluir README, código comentado e breve relatório (2 páginas). **Peso: 50%**

### Critérios de Avaliação do Mini-Projeto
| Critério | Peso |
|----------|------|
| Especificação PEAS correta e completa | 20% |
| Qualidade e clareza do código Python | 25% |
| Funcionalidade do agente implementado | 30% |
| Qualidade do relatório e reflexões éticas | 15% |
| Criatividade e complexidade do problema escolhido | 10% |

---

## Recursos e Ferramentas

### Software (tudo gratuito)
- **Python 3.11+**: https://www.python.org/
- **Anaconda Distribution**: https://www.anaconda.com/download
- **VS Code**: https://code.visualstudio.com/
- **Google Colab**: https://colab.research.google.com/
- **Jupyter Lab**: instalado via `pip install jupyterlab`

### Bibliotecas Python (instalação via pip/conda)
```bash
conda create -n ia-curso python=3.11
conda activate ia-curso
pip install numpy pandas matplotlib seaborn scikit-learn jupyter notebook
```

### Recursos Online Complementares
- **Documentação NumPy**: https://numpy.org/doc/
- **Documentação Pandas**: https://pandas.pydata.org/docs/
- **Documentação Scikit-learn**: https://scikit-learn.org/stable/
- **AI Index Report (Stanford)**: https://aiindex.stanford.edu/
- **Papers With Code**: https://paperswithcode.com/ (para acompanhar o estado da arte)
- **Distill.pub**: https://distill.pub/ (artigos visuais sobre ML)

### Datasets para Prática
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/
- **Kaggle Datasets**: https://www.kaggle.com/datasets
- **Seaborn built-in datasets**: `seaborn.load_dataset('iris')`, `seaborn.load_dataset('titanic')`

---

## Referências Bibliográficas

### Bibliografia Básica

**[1]** FACELI, Katti; LORENA, Ana Carolina; GAMA, João; ALMEIDA, Tiago A. de; CARVALHO, André Carlos P. L. F. de. **Inteligência Artificial: uma abordagem de aprendizado de máquina**. 2. ed. Rio de Janeiro: LTC, 2021. ISBN: 978-85-216-3774-4.
> *Capítulos relevantes para este módulo: Cap. 1 (Introdução ao Aprendizado de Máquina), Cap. 2 (Conceitos Básicos).*

**[2]** RUSSELL, Stuart; NORVIG, Peter. **Inteligência Artificial: uma abordagem moderna**. 4. ed. Rio de Janeiro: GEN LTC, 2022. ISBN: 978-85-216-3428-6.
> *Capítulos relevantes: Cap. 1 (Introdução), Cap. 2 (Agentes Inteligentes), Cap. 3 (Resolução de Problemas por Busca), Cap. 7 (Agentes Lógicos), Cap. 12 (Raciocínio Probabilístico).*

**[3]** GÉRON, Aurélien. **Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow**. 3. ed. Rio de Janeiro: Alta Books, 2023. ISBN: 978-65-5520-428-4.
> *Capítulos relevantes: Cap. 1 (O Cenário do Aprendizado de Máquina), Cap. 2 (Projeto de Aprendizado de Máquina de Ponta a Ponta).*

### Bibliografia Complementar

**[4]** GOODFELLOW, Ian; BENGIO, Yoshua; COURVILLE, Aaron. **Deep Learning**. MIT Press, 2016. Disponível gratuitamente em: https://www.deeplearningbook.org/
> *Referência fundamental para os módulos de redes neurais. Cap. 1 fornece excelente contextualização histórica.*

**[5]** MITCHELL, Tom M. **Machine Learning**. McGraw-Hill, 1997.
> *Obra clássica. Cap. 1 define formalmente o que é aprendizado de máquina.*

**[6]** TURING, Alan M. **Computing Machinery and Intelligence**. Mind, v. 59, n. 236, p. 433-460, 1950.
> *O artigo original que definiu o Teste de Turing. Leitura histórica essencial.*

**[7]** SEARLE, John R. **Minds, Brains, and Programs**. Behavioral and Brain Sciences, v. 3, n. 3, p. 417-424, 1980.
> *Artigo original do Quarto Chinês. Disponível em: https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/minds-brains-and-programs/DC644B47A4299C637C89772FACC2706A*

**[8]** VASWANI, Ashish et al. **Attention Is All You Need**. NeurIPS, 2017. arXiv:1706.03762.
> *O artigo que introduziu a arquitetura Transformer, base dos LLMs modernos.*

**[9]** LECUN, Yann; BENGIO, Yoshua; HINTON, Geoffrey. **Deep Learning**. Nature, v. 521, p. 436-444, 2015.
> *Artigo seminal de revisão sobre deep learning pelos três "padrinhos" da área.*

---

## Cronograma Sugerido

```
SEMANA 1
├── Aula 01: História da IA e Conceitos Fundamentais
│   └── Tarefa casa: Ler Cap. 1 de Russell & Norvig
├── Aula 02: Agentes Inteligentes
│   └── Tarefa casa: Especificar via PEAS 3 sistemas do cotidiano

SEMANA 2
├── Aula 03: Busca e Resolução de Problemas
│   └── Tarefa casa: Implementar BFS para o problema do labirinto
├── Aula 04: Representação do Conhecimento
│   └── Tarefa casa: Modelar um domínio simples com lógica de primeira ordem

SEMANA 3
├── Aula 05: Python para IA e Ciência de Dados
│   └── Tarefa casa: Iniciar mini-projeto (escolher problema e especificar PEAS)
└── ENTREGA DO MINI-PROJETO (1 semana após a Aula 05)
```

---

## Conexão com os Próximos Módulos

Este módulo estabelece a fundação para todo o curso. Veja como os tópicos se conectam:

```
Módulo 01 (Fundamentos)
    │
    ├─→ Módulo 02 (Paradigmas de Aprendizado)
    │       ├── Supervisionado, Não-supervisionado, Por Reforço
    │       └── Usa: conceitos de agente, Python
    │
    ├─→ Módulo 03 (Preparação de Dados)
    │       └── Usa: Pandas, NumPy da Aula 05
    │
    ├─→ Módulo 04-08 (Algoritmos de ML)
    │       └── Usa: Scikit-learn, raciocínio de agente, busca heurística
    │
    └─→ Módulo 10 (Deep Learning)
            └── Usa: abordagem conexionista da Aula 01
```

---

*Módulo elaborado seguindo as diretrizes curriculares para cursos de graduação em Ciência da Computação, Engenharia de Software e áreas afins. Atualizado para refletir o estado da arte em IA em 2024.*

*Autor: Equipe Didática — Disciplina de Inteligência Artificial e Aprendizado de Máquina*  
*Última atualização: 2024*
