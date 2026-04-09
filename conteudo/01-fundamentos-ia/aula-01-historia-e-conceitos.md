# Aula 01 — História da IA e Conceitos Fundamentais

> **Módulo:** 01 — Fundamentos de Inteligência Artificial  
> **Duração:** 45 minutos  
> **Pré-requisitos:** Nenhum (aula introdutória)

---

## Objetivos de Aprendizagem

Ao final desta aula, o estudante será capaz de:

1. **Definir** Inteligência Artificial a partir de diferentes perspectivas (comportamental, racional, humana, ideal).
2. **Narrar** os principais marcos históricos da IA, desde Alan Turing (1950) até os grandes modelos de linguagem (2024).
3. **Distinguir** IA Estreita (ANI), IA Geral (AGI) e IA Super-Humana (ASI), com exemplos concretos de cada.
4. **Descrever** as quatro principais abordagens da IA: simbólica, conexionista, evolucionária e bayesiana.
5. **Explicar** o Teste de Turing e a crítica do Quarto Chinês de Searle.
6. **Identificar** aplicações atuais de IA e os questionamentos éticos que elas suscitam.

---

## 1. O Que É Inteligência Artificial?

### 1.1 O Problema das Definições

Definir "Inteligência Artificial" é surpreendentemente difícil — e isso nos diz algo importante sobre o campo. A dificuldade vem de duas fontes: não sabemos definir precisamente o que é **inteligência**, e a palavra **artificial** acrescenta outra camada de ambiguidade.

Russell e Norvig (2022) propõem um mapa de definições organizadas em dois eixos:

```
                    PROCESSO DE PENSAMENTO
                 ┌─────────────────────────────┐
                 │   Pensando como humanos      │   Pensando racionalmente
      CRITÉRIO   │   "o esforço para fazer      │   "o estudo das faculdades
      HUMANO     │   computadores pensarem,     │   mentais por meio de
                 │   no sentido literal"        │   modelos computacionais"
                 │   (Haugeland, 1985)          │   (Charniak, 1985)
                 ├─────────────────────────────┤
                 │   Agindo como humanos        │   Agindo racionalmente
      CRITÉRIO   │   "a arte de criar máquinas  │   "a computação que faz
      IDEAL      │   que executam funções que   │   o que é certo, dadas
                 │   exigem inteligência quando │   as crenças e objetivos"
                 │   feitas por humanos"        │   (Poole et al., 1998)
                 │   (Kurzweil, 1990)           │
                 └─────────────────────────────┘
```

**Neste curso, adotamos a perspectiva do Agente Racional**: um sistema de IA é aquele que percebe seu ambiente e toma ações que maximizam suas chances de atingir seus objetivos — não necessariamente agindo como um humano, mas agindo de forma ótima dado o que sabe.

Essa definição é mais operacional e nos permite construir sistemas práticos. Um GPS que encontra o melhor caminho é racional (encontra o ótimo) mesmo sem "pensar como humano".

### 1.2 Por que IA é Difícil?

Tarefas que parecem simples para humanos são extraordinariamente difíceis para computadores, e vice-versa. Isso é conhecido como o **Paradoxo de Moravec**:

> *"É relativamente fácil fazer computadores exibirem desempenho em nível adulto em testes de inteligência ou xadrez, e difícil ou impossível dar-lhes as habilidades de uma criança de um ano em termos de percepção e mobilidade."*  
> — Hans Moravec, 1988

| Tarefa | Humano | Computador |
|--------|--------|------------|
| Cálculo de 1000 × 3.14159 | Difícil (lento) | Trivial (nanosegundos) |
| Jogar xadrez em nível mundial | Muito difícil | Fácil (Deep Blue, 1997) |
| Reconhecer o rosto da mãe | Trivial (milissegundos) | Difícil (décadas de pesquisa) |
| Andar em terreno irregular | Trivial | Extremamente difícil |
| Compreender sarcasmo | Fácil (contexto social) | Muito difícil |
| Traduzir "banco" (assento vs. financeiro) | Trivial (contexto) | Difícil (requer contexto) |

---

## 2. Uma Breve História da Inteligência Artificial

### 2.1 Os Precursores (antes de 1950)

A ideia de máquinas inteligentes antecede os computadores modernos por milênios:

- **Autômatos da Antiguidade**: Herão de Alexandria (séc. I d.C.) construiu autômatos mecânicos movidos a vapor.
- **Golem da tradição judaica**: A ideia de criar vida artificial a partir de matéria inanimada.
- **Leibniz (1646-1716)**: Propôs uma *calculus ratiocinator* — um cálculo universal que resolveria qualquer questão pelo raciocínio.
- **George Boole (1815-1864)**: Formalizou a lógica em álgebra booleana, base da computação digital.
- **Ada Lovelace (1815-1852)**: Escreveu o primeiro algoritmo para a Máquina Analítica de Babbage. Questionou se máquinas poderiam "pensar".
- **Alan Turing (1912-1954)**: Criou o modelo teórico da Máquina de Turing (1936), definiu computabilidade e, em 1950, propôs o "Jogo da Imitação".

### 2.2 O Nascimento Formal (1950-1956)

```
1950 ──── Alan Turing publica "Computing Machinery and Intelligence"
          → Propõe o "Jogo da Imitação" (Teste de Turing)
          → Discute se máquinas podem aprender

1951 ──── Marvin Minsky e Dean Edmonds constroem o SNARC
          → Primeira rede neural de aprendizado (40 neurônios)

1956 ──── Conferência de Dartmouth (verão de 1956)
          → John McCarthy, Marvin Minsky, Claude Shannon, Nathaniel Rochester
          → McCarthy cunha o termo "Inteligência Artificial"
          → Proposta: "cada aspecto da aprendizagem pode ser descrito com
            precisão suficiente para que uma máquina possa simulá-lo"
          ★ MARCO: nascimento oficial da IA como disciplina
```

A **Conferência de Dartmouth** (1956) é considerada o marco de nascimento da IA. A proposta original era otimista demais: em 2 meses de trabalho intenso, 10 pesquisadores resolveriam os problemas fundamentais da IA. Como sabemos, isso não aconteceu — e a subestimação da dificuldade do problema seria um padrão que se repetiria.

### 2.3 A Era de Ouro (1956-1974)

O período inicial foi marcado por entusiasmo excessivo e resultados impressionantes em domínios limitados:

**Programas notáveis:**
- **Logic Theorist (1956)** — Newell, Shaw, Simon: provou 38 dos 52 teoremas do *Principia Mathematica* de Russell e Whitehead.
- **General Problem Solver (1957)** — Newell & Simon: tentativa de criar um solucionador de problemas genérico por meios-fins.
- **ELIZA (1966)** — Joseph Weizenbaum (MIT): primeiro chatbot. Simulava um psicoterapeuta Rogeriano. Pessoas se apegavam emocionalmente ao programa — o que preocupou o próprio criador.
- **SHRDLU (1970)** — Terry Winograd: sistema de linguagem natural que manipulava blocos coloridos em um mundo virtual.
- **Perceptron (1958)** — Frank Rosenblatt: primeiro modelo formal de neurônio artificial com aprendizado.

**A promessa inflacionada:**
> *"Dentro de 10 anos, um computador digital será o campeão mundial de xadrez."*  
> — Herbert Simon, 1957

> *"Dentro de 20 anos, as máquinas serão capazes de fazer qualquer trabalho que um homem possa fazer."*  
> — Herbert Simon, 1965

Nenhuma dessas previsões se cumpriu no prazo. Isso levou ao primeiro "Inverno da IA".

### 2.4 Primeiro Inverno da IA (1974-1980)

O entusiasmo desmoronou quando ficou claro que as abordagens iniciais não escalavam:

**Causas principais:**
1. **Explosão combinatória**: Os algoritmos de busca funcionavam bem para problemas pequenos, mas eram inviáveis para problemas reais.
2. **Crítica de Minsky e Papert (1969)**: O livro *Perceptrons* demonstrou matematicamente que perceptrons de camada única não podiam aprender funções XOR — generalizando erroneamente para multilayer networks.
3. **Relatório Lighthill (1973)**: O governo britânico encomendou uma avaliação da IA. Sir James Lighthill concluiu que a IA havia falhado em seus objetivos e recomendou corte de financiamento.
4. **Limitações de hardware**: Computadores da época eram ordens de magnitude lentos demais para as tarefas pretendidas.

O financiamento governamental (especialmente DARPA nos EUA e Science Research Council no Reino Unido) foi drasticamente reduzido.

### 2.5 Sistemas Especialistas e o Boom do Conhecimento (1980-1987)

O campo renasceu com uma nova abordagem: **sistemas especialistas** — programas que codificavam o conhecimento de especialistas humanos em regras `SE-ENTÃO`.

**Exemplos notáveis:**
- **MYCIN (1974, Stanford)**: Diagnosticava doenças infecciosas do sangue e recomendava antibióticos. Desempenho comparável ao de médicos especialistas, mas superior a médicos generalistas. Nunca usado clinicamente por questões regulatórias.
- **DENDRAL (1965-1983, Stanford)**: Identificava estruturas moleculares a partir de espectros de massa.
- **R1/XCON (DEC, 1980)**: Configurava pedidos de sistemas de computadores VAX. Economizou ~$40 milhões/ano para a DEC.
- **CADUCEUS**: Sistema de diagnóstico geral com ~4000 doenças e ~5000 achados clínicos.

Em 1985, empresas gastavam **mais de $1 bilhão/ano** em sistemas especialistas. A IA era um negócio.

**Limitações dos sistemas especialistas:**
- Frágeis fora do domínio (sem senso comum).
- Difíceis de manter (o conhecimento muda).
- Gargalo do especialista (obter o conhecimento era custoso).
- Sem capacidade de aprender.

### 2.6 Segundo Inverno da IA (1987-1993)

O mercado de hardware de IA (máquinas LISP) entrou em colapso em 1987. O DARPA cortou financiamento novamente. Muitas empresas de IA faliram.

Paralelamente, ocorreu um desenvolvimento silencioso mas crucial: a **redescoberta do backpropagation** (Rumelhart, Hinton, Williams, 1986) mostrou que redes neurais multicamadas podiam ser treinadas. Mas o hardware ainda era lento demais.

### 2.7 Renascimento e Era dos Agentes (1993-2011)

A IA voltou, mas com expectativas mais realistas e abordagens mais rigorosas:

```
1997 ──── Deep Blue (IBM) vence Garry Kasparov no xadrez
          → Primeiro sistema computacional a derrotar o campeão mundial
          → Combinação de busca com avaliação heurística

1997 ──── LSTM proposta por Hochreiter & Schmidhuber
          → Resolve o problema do gradiente desvanecente em RNNs
          → Base para processamento de sequências por décadas

2002 ──── Roomba (iRobot) lançado
          → Primeiro robô doméstico comercialmente bem-sucedido

2004 ──── DARPA Grand Challenge (deserto do Mojave)
          → Carros autônomos: nenhum completou o percurso de 240km
          → Mas plantou a semente do que viria

2005 ──── DARPA Grand Challenge 2: 5 veículos completam o percurso
          → Stanford's Stanley venceu com aprendizado de máquina

2007 ──── DARPA Urban Challenge: carros autônomos em ambiente urbano
          → Veículo de CMU (Boss) venceu

2008 ──── Google lança Street View (veículo coletou 5M km de imagens)
```

### 2.8 A Revolução do Deep Learning (2012-2017)

O evento que mudou tudo:

```
2012 ──── AlexNet (Krizhevsky, Sutskever, Hinton) vence ImageNet
          → Taxa de erro: 15.3% vs. 26.2% do segundo lugar
          → GPU training com CUDA
          ★ MARCO: início da era moderna do deep learning

2013 ──── Word2Vec (Mikolov, Google)
          → Representações vetoriais densas de palavras
          → Semântica emerge de estatísticas de coocorrência

2014 ──── GANs (Goodfellow et al.)
          → Redes Adversariais Generativas
          → Primeiras imagens sintéticas convincentes

2015 ──── ResNet (He et al., Microsoft)
          → 152 camadas, vence humanos no ImageNet (3.57% vs. 5.1%)
          → Conexões residuais resolvem degradação de gradiente

2016 ──── AlphaGo (DeepMind) vence Lee Sedol no Go (4-1)
          → Go tem mais configurações que átomos no universo observável
          → Combinação de MCTS + redes neurais profundas
          ★ MARCO: IA supera humanos em Go, antes considerado impossível

2017 ──── AlphaZero aprende xadrez, Go e shogi do zero em horas
          → Sem conhecimento humano, apenas regras do jogo
```

### 2.9 A Era dos Transformers e LLMs (2017-presente)

```
2017 ──── "Attention Is All You Need" (Vaswani et al., Google)
          → Arquitetura Transformer: self-attention sem recorrência
          ★ MARCO: base de praticamente todos os LLMs modernos

2018 ──── BERT (Devlin et al., Google)
          → Pré-treinamento bidirecional em texto
          → Estado da arte em 11 tarefas de NLP simultaneamente

2018 ──── GPT-1 (OpenAI): 117M parâmetros
2019 ──── GPT-2 (OpenAI): 1.5B parâmetros
          → OpenAI inicialmente se recusou a lançar por "risco de mal uso"

2020 ──── GPT-3 (OpenAI): 175B parâmetros
          → Primeiros sinais de "emergência": capacidades inesperadas

2021 ──── DALL-E (OpenAI): geração de imagens a partir de texto
          → GitHub Copilot (baseado em Codex): código assistido por IA

2022 ──── ChatGPT (OpenAI, novembro 2022)
          → 1 milhão de usuários em 5 dias; 100 milhões em 2 meses
          ★ MARCO: IA generativa alcança o público geral

2022 ──── Stable Diffusion: geração de imagens open-source
          → Midjourney: imagens de qualidade artística

2023 ──── GPT-4 (OpenAI): multimodal, capacidades próximas de especialistas
          → Gemini Ultra (Google): supera humanos em MMLU
          → Llama (Meta): LLM open-source competitivo
          → Claude (Anthropic): foco em segurança e helpfulness

2024 ──── GPT-4o, Gemini 1.5 Pro, Claude 3 Opus
          → Raciocínio multimodal (texto, imagem, áudio, vídeo)
          → Janelas de contexto de 1M+ tokens
          → Modelos especializados para código, matemática, medicina
```

---

## 3. Tipos de Inteligência Artificial

### 3.1 IA Estreita (ANI — Artificial Narrow Intelligence)

**Definição**: Sistemas que executam uma única tarefa específica com desempenho igual ou superior ao humano, mas que não generalizam para outras tarefas.

**Características:**
- Toda IA existente hoje é ANI.
- Desempenho pode superar humanos em domínios específicos.
- Sem consciência, compreensão ou adaptação a tarefas não treinadas.
- Necessita de muitos dados e treinamento específico para cada tarefa.

**Exemplos:**
| Sistema | Tarefa | Desempenho |
|---------|--------|------------|
| DeepBlue/Stockfish | Xadrez | >> Humano |
| AlphaGo/AlphaZero | Go, Xadrez, Shogi | >> Humano |
| GPT-4, Gemini, Claude | Linguagem natural | ≈ Especialista humano (contexto) |
| Tesla Autopilot | Direção em rodovias | ≈ Motorista humano |
| AlphaFold2 | Predição de estrutura proteica | >> Humano |
| DALL-E, Midjourney | Geração de imagens | Diferente do humano |
| Filtro de spam | Classificação de emails | >> Humano |

### 3.2 IA Geral (AGI — Artificial General Intelligence)

**Definição**: Hipotético sistema que possuiria inteligência comparável à humana em qualquer domínio cognitivo — capacidade de aprender qualquer tarefa intelectual que um humano possa aprender.

**Características hipotéticas:**
- Transferência de conhecimento entre domínios.
- Aprendizado com poucos exemplos (few-shot learning).
- Senso comum e raciocínio causal.
- Autoconhecimento e planejamento de longo prazo.
- Criatividade genuína (não recombinação de exemplos).

**Estamos próximos de AGI?** — O debate é intenso:
- **Ceticismo (Gary Marcus, Yann LeCun)**: LLMs são "papagaios estocásticos" — recombinadores sofisticados sem compreensão real.
- **Otimismo moderado (Sam Altman)**: "AGI é mais próximo do que pensamos."
- **Perspectiva técnica**: Há lacunas claras — raciocínio causal, aritmética confiável, planejamento de longo prazo.

### 3.3 IA Super-Humana (ASI — Artificial Superintelligence)

**Definição**: Sistema hipotético que supera a inteligência humana em **todas** as dimensões — criatividade, sabedoria, raciocínio social, científico e artístico.

**Conceitos relacionados:**
- **Explosão de Inteligência (I.J. Good, 1965)**: Uma AGI poderia criar uma versão melhorada de si mesma, que criaria outra ainda melhor, em um loop positivo.
- **Singularidade Tecnológica (Vinge, 1993; Kurzweil, 2005)**: Ponto em que a IA supera a inteligência humana coletiva, tornando o futuro imprevisível.
- **Prazo estimado**: Kurzweil previu 2029 para AGI e 2045 para a Singularidade. Outros pesquisadores são muito mais céticos.

**Perspectiva responsável**: A ASI é atualmente especulação filosófica, mas as questões que ela levanta — alinhamento de valores, controle — já são relevantes para sistemas ANI atuais.

---

## 4. As Quatro Abordagens da IA

### 4.1 IA Simbólica (Good Old-Fashioned AI — GOFAI)

**Premissa**: A inteligência pode ser capturada pela manipulação explícita de símbolos segundo regras formais.

**Como funciona:**
- Conhecimento representado como regras, lógica, ontologias.
- Raciocínio = aplicação de regras a símbolos.
- O programador codifica o conhecimento explicitamente.

**Exemplos**: Sistemas especialistas, planejamento lógico, provadores de teoremas, motores de busca baseados em regras.

**Vantagens:**
- Interpretável: podemos explicar cada decisão.
- Funciona com poucos dados.
- Pode raciocinar com precisão em domínios bem definidos.

**Limitações:**
- Não escala para o mundo real (conhecimento é demais para codificar manualmente).
- Frágil a incerteza e ambiguidade.
- O gargalo do especialista.

### 4.2 IA Conexionista (Redes Neurais / Deep Learning)

**Premissa**: A inteligência emerge de redes de unidades simples (neurônios artificiais) interconectadas que aprendem a partir de dados.

**Como funciona:**
- Inspirado no neurônio biológico.
- Redes com milhões/bilhões de parâmetros ajustados por backpropagation.
- Aprende representações hierárquicas a partir de exemplos.

**Exemplos**: CNNs para visão, RNNs/LSTMs para sequências, Transformers para linguagem, GANs para geração.

**Vantagens:**
- Escala magnificamente com dados e compute.
- Aprende representações automaticamente.
- Desempenho estado da arte em percepção e linguagem.

**Limitações:**
- Caixa preta: difícil de interpretar.
- Requer muitos dados rotulados.
- Não raciocina causalmente.
- Pode falhar catastroficamente em inputs fora da distribuição.

### 4.3 IA Evolucionária (Computação Evolutiva)

**Premissa**: A inteligência pode evoluir através de processos análogos à seleção natural darwiniana.

**Como funciona:**
- Algoritmos genéticos: população de soluções candidatas evolui por mutação, crossover e seleção.
- Programação genética: evolui programas de computador.
- Neuroevolução: evolui a topologia de redes neurais.

**Exemplos**: Otimização de rotas, design de antenas (NASA), arquitetura neural (NEAT), jogos (OpenAI Five com evolução de políticas).

**Vantagens:**
- Não requer gradiente: funciona para espaços de busca não-diferenciáveis.
- Pode encontrar soluções inovadoras e contra-intuitivas.
- Paralelizável.

**Limitações:**
- Lento para problemas de alta dimensão.
- Difícil de escalar para linguagem/visão complexa.
- Convergência prematura.

### 4.4 IA Bayesiana (Raciocínio Probabilístico)

**Premissa**: A inteligência requer raciocinar sobre incerteza de forma coerente, usando probabilidade.

**Como funciona:**
- Conhecimento representado como distribuições de probabilidade.
- Raciocínio = atualização de crenças via Teorema de Bayes.
- Inferência em redes bayesianas, modelos de Markov ocultos.

**Exemplos**: Filtros de Kalman (navegação), diagnóstico médico probabilístico, reconhecimento de fala (HMMs), modelos de tópicos (LDA).

**Vantagens:**
- Princípios matemáticos sólidos.
- Quantifica incerteza explicitamente.
- Funciona bem com poucos dados (priors fortes).

**Limitações:**
- Intratável para problemas de alta dimensão (sem aproximações).
- Escolha de prior pode ser subjetiva.
- Nem sempre escala bem.

---

## 5. O Teste de Turing e Seus Críticos

### 5.1 O Teste de Turing (1950)

Alan Turing propôs o "Jogo da Imitação" em seu artigo seminal como uma forma operacional de responder à pergunta "Podem máquinas pensar?":

**Configuração original:**
```
Interrogador (humano, em sala separada)
    │
    │ texto escrito
    ├──────────────────→ Jogador A (humano ou máquina?)
    │
    └──────────────────→ Jogador B (o outro)
```

Se o interrogador não consegue distinguir, com confiabilidade acima do acaso, qual é a máquina, a máquina passou no teste.

**O Teste de Turing Total (Harnad, 1991)**: Versão expandida que inclui percepção visual e manipulação de objetos físicos — requerendo robótica avançada.

**Capacidades necessárias para passar no Teste de Turing:**
1. **Processamento de linguagem natural**: Compreender e gerar linguagem humana.
2. **Representação de conhecimento**: Armazenar e usar informações sobre o mundo.
3. **Raciocínio automatizado**: Responder perguntas e tirar conclusões.
4. **Aprendizado de máquina**: Adaptar-se a novos padrões e situações.

**Sistemas que passaram (ou quase) no Teste de Turing:**
- **ELIZA (1966)**: Enganou pessoas, mas por razões psicológicas (efeito ELIZA).
- **Eugene Goostman (2014)**: Chatbot simulando um menino ucraniano de 13 anos "enganou" 33% dos juízes. Debate sobre validade.
- **GPT-4 (2023)**: Em alguns experimentos, avaliadores humanos não conseguem distinguir. Mas isso revela limitações do teste, não AGI.

### 5.2 O Quarto Chinês de Searle (1980)

John Searle propôs um experimento mental para criticar a ideia de que um programa que passa no Teste de Turing realmente "compreende":

**O experimento:**
```
Fora do quarto:          Dentro do quarto:
Falante nativo         ┌─────────────────────┐
de chinês escreve  →   │ Pessoa que não fala  │
"你好吗？"             │ chinês + livros de   │  →  "我很好，谢谢"
(Como vai você?)       │ regras de combinação │    (Estou bem, obrigado)
                       │ de símbolos          │
                       └─────────────────────┘
```

**Argumento de Searle:**
- A pessoa dentro do quarto manipula símbolos sem compreendê-los.
- Do exterior, parece que o quarto compreende chinês.
- Mas há apenas manipulação sintática, não semântica (compreensão real).
- Portanto, mesmo que um computador passe no Teste de Turing, isso não prova que ele "compreende".

**Contra-argumentos:**
- **Resposta do sistema**: Talvez o "sistema inteiro" (pessoa + regras + sala) compreenda, mesmo que a pessoa não compreenda individualmente.
- **Resposta do robô**: Se conectado a sensores e atuadores, o sistema teria fundamentação semântica no mundo real.
- **Resposta do cérebro**: O cérebro também "manipula símbolos" — neurônios disparando. Por que seria diferente?

**Relevância atual**: O debate Turing vs. Searle continua vivo na era dos LLMs. GPT-4 "conversa" de forma convincente — isso é compreensão ou manipulação sofisticada de padrões? Essa questão tem implicações éticas e práticas profundas.

---

## 6. Aplicações Atuais e o Estado da Arte

### 6.1 Linguagem e Geração de Texto
- **ChatGPT / GPT-4o (OpenAI)**: Assistente de escrita, programação, análise.
- **Gemini (Google)**: Multimodal, integrado ao Google Workspace.
- **Claude (Anthropic)**: Foco em segurança e tarefas de raciocínio.
- **Llama 3 (Meta)**: Open-source, pode rodar localmente.

### 6.2 Visão Computacional e Geração de Imagens
- **DALL-E 3 (OpenAI)**: Geração de imagens fotorrealistas a partir de texto.
- **Midjourney**: Geração de arte de alta qualidade.
- **Stable Diffusion**: Open-source, altamente customizável.
- **GPT-4V, Gemini Vision**: Análise de imagens e resposta a perguntas visuais.

### 6.3 Jogos e Raciocínio Estratégico
- **AlphaGo/AlphaZero (DeepMind)**: Mestre no Go, xadrez, shogi.
- **OpenAI Five**: Derrotou campeões mundiais no Dota 2 (jogo em equipe).
- **AlphaStar (DeepMind)**: Nível Grandmaster em StarCraft II.
- **MuZero**: Aprende as regras e a estratégia simultaneamente.

### 6.4 Ciência e Saúde
- **AlphaFold2 (DeepMind)**: Previu estruturas de ~200 milhões de proteínas.
- **Med-PaLM 2 (Google)**: Respostas médicas de nível especialista.
- **Diagnóstico por imagem**: CNNs com desempenho comparável a radiologistas.
- **Descoberta de drogas**: IA design de moléculas (Insilico Medicine).

### 6.5 Veículos Autônomos
- **Tesla Autopilot / Full Self-Driving**: Assistência à direção em rodovias e cidades.
- **Waymo**: Táxi autônomo sem motorista em San Francisco e Phoenix.
- **Cruise (GM)**: Serviço de táxi autônomo (suspenso após incidente, 2023).

---

## 7. Questões Éticas Iniciais

A IA traz consigo desafios éticos que permeiam todo o curso:

### 7.1 Viés e Discriminação
Sistemas treinados em dados históricos podem perpetuar e amplificar preconceitos:
- **COMPAS (Correctional Offender Management Profiling)**: Sistema de avaliação de risco criminal que mostrou viés racial.
- **Amazon Recruiting Tool (2018)**: Sistema de triagem de currículos discriminava mulheres (treinado em currículos históricos, majoritariamente masculinos).
- **Reconhecimento facial**: Taxas de erro até 35% maiores para mulheres negras do que para homens brancos.

### 7.2 Privacidade e Vigilância
- Reconhecimento facial em larga escala (China, mas também EUA/Europa).
- Coleta massiva de dados para treinamento.
- "Data exhaust" — dados gerados como subproduto de atividade online.

### 7.3 Automação e Emprego
- Estudos estimam que 47% dos empregos nos EUA estão em risco de automação (Frey & Osborne, 2013).
- O impacto não é uniforme: trabalhadores de baixa renda e rotineiros são mais vulneráveis.
- Mas IA também cria novos empregos e aumenta produtividade.

### 7.4 Responsabilidade e Transparência
- Quem é responsável quando um carro autônomo causa um acidente?
- Como contestar uma decisão tomada por algoritmo?
- "Right to Explanation" (GDPR na Europa).

### 7.5 Concentração de Poder
- Poucas empresas (Google, Microsoft, OpenAI, Meta, Amazon) controlam os maiores modelos.
- Assimetria de recursos: treinar GPT-4 custou ~$100 milhões.

---

## 8. Exemplo de Código: Chatbot Simples Baseado em Regras

```python
"""
Chatbot simples baseado em regras — conceito de IA Simbólica
Demonstra a abordagem mais básica de IA conversacional (pré-LLMs)
"""

import re
from typing import Optional


class ChatbotSimples:
    """
    Chatbot básico usando correspondência de padrões (pattern matching).
    Inspirado no ELIZA de Weizenbaum (1966), mas muito simplificado.
    
    Esta é IA Simbólica: o conhecimento é codificado explicitamente
    pelo programador como regras SE-ENTÃO.
    """

    def __init__(self):
        # Cada regra é uma tupla (padrão_regex, lista_de_respostas)
        # O chatbot escolhe aleatoriamente entre as respostas disponíveis
        self.regras = [
            (
                r"olá|oi|hey|e aí",
                [
                    "Olá! Como posso ajudar?",
                    "Oi! Em que posso ser útil?",
                    "Olá! Que bom te ver por aqui.",
                ]
            ),
            (
                r"(meu nome é|me chamo|sou o|sou a) (\w+)",
                [
                    "Que nome bonito, {2}! Prazer em te conhecer.",
                    "Olá, {2}! Como posso ajudar você hoje?",
                ]
            ),
            (
                r"o que (é|são) (inteligência artificial|ia|machine learning|ml)",
                [
                    "Inteligência Artificial é o campo da ciência da computação "
                    "que estuda como criar sistemas capazes de realizar tarefas "
                    "que normalmente requerem inteligência humana.",
                    "IA é a ciência e engenharia de criar máquinas inteligentes, "
                    "especialmente programas de computador inteligentes. "
                    "(John McCarthy, 1956)",
                ]
            ),
            (
                r"(como|o que) (você (é|faz)|és|você pode)",
                [
                    "Sou um chatbot simples baseado em regras. "
                    "Estou aqui para demonstrar como a IA Simbólica funciona!",
                    "Sou um programa que usa correspondência de padrões para responder. "
                    "Nada de LLM aqui — apenas boas e velhas regras SE-ENTÃO!",
                ]
            ),
            (
                r"(tchau|até logo|até mais|adeus|bye)",
                [
                    "Até logo! Foi um prazer conversar.",
                    "Tchau! Volte quando quiser.",
                    "Adeus! Espero ter ajudado.",
                ]
            ),
            (
                r"(obrigado|obrigada|valeu|thanks)",
                [
                    "De nada! Posso ajudar com mais alguma coisa?",
                    "Fico feliz em ajudar!",
                ]
            ),
            (
                r"(qual|quem) (é|foi) (alan turing|turing)",
                [
                    "Alan Turing (1912-1954) foi um matemático e cientista da computação "
                    "britânico. É considerado o pai da computação e da IA. Em 1950, "
                    "propôs o famoso 'Teste de Turing' no artigo "
                    "'Computing Machinery and Intelligence'.",
                ]
            ),
            (
                r"(inverno da ia|ai winter)",
                [
                    "Os 'Invernos da IA' foram períodos de redução drástica de "
                    "financiamento e interesse em IA (1974-1980 e 1987-1993), "
                    "causados por expectativas exageradas que não foram cumpridas.",
                ]
            ),
        ]
        # Resposta padrão quando nenhuma regra corresponde
        self.resposta_padrao = [
            "Interessante! Mas ainda não aprendi sobre isso. "
            "Tente perguntar sobre IA, Alan Turing ou o que sou.",
            "Hmm, não tenho uma resposta para isso ainda. "
            "Sou um chatbot simples — tente algo relacionado à IA!",
            "Ainda estou aprendendo. Pode reformular sua pergunta?",
        ]
        self._resposta_idx = 0  # Para rotação determinística (sem random)

    def _escolher_resposta(self, respostas: list) -> str:
        """Rotaciona entre as respostas disponíveis de forma determinística."""
        resposta = respostas[self._resposta_idx % len(respostas)]
        self._resposta_idx += 1
        return resposta

    def _substituir_grupos(self, template: str, match: re.Match) -> str:
        """Substitui {1}, {2}, etc. pelos grupos capturados no match."""
        resultado = template
        for i, grupo in enumerate(match.groups(), 1):
            if grupo:
                resultado = resultado.replace(f"{{{i}}}", grupo.capitalize())
        return resultado

    def responder(self, entrada: str) -> str:
        """
        Processa a entrada do usuário e retorna uma resposta.
        
        Args:
            entrada: Texto do usuário
            
        Returns:
            Resposta do chatbot baseada nas regras definidas
        """
        entrada_normalizada = entrada.lower().strip()
        
        # Tenta cada regra em ordem
        for padrao, respostas in self.regras:
            match = re.search(padrao, entrada_normalizada)
            if match:
                resposta_template = self._escolher_resposta(respostas)
                return self._substituir_grupos(resposta_template, match)
        
        # Nenhuma regra correspondeu
        return self._escolher_resposta(self.resposta_padrao)

    def iniciar_conversa(self):
        """Inicia um loop de conversa interativa no terminal."""
        print("=" * 60)
        print("  CHATBOT SIMBÓLICO — Demonstração de IA baseada em Regras")
        print("  (Digite 'tchau' para encerrar)")
        print("=" * 60)
        
        while True:
            try:
                entrada = input("\nVocê: ").strip()
                if not entrada:
                    continue
                    
                resposta = self.responder(entrada)
                print(f"Bot:  {resposta}")
                
                # Verifica se o usuário quer sair
                if re.search(r"tchau|até logo|até mais|adeus|bye", entrada.lower()):
                    break
                    
            except (KeyboardInterrupt, EOFError):
                print("\nBot: Até logo!")
                break


def demonstrar_chatbot():
    """Demonstra o chatbot com uma conversa pré-definida."""
    print("\n" + "="*60)
    print("DEMONSTRAÇÃO DO CHATBOT SIMBÓLICO")
    print("="*60)
    
    bot = ChatbotSimples()
    
    conversas_teste = [
        "Olá!",
        "Meu nome é Maria",
        "O que é Inteligência Artificial?",
        "Quem foi Alan Turing?",
        "O que é inverno da IA?",
        "Como você funciona?",
        "O que é blockchain?",  # Fora do domínio — resposta padrão
        "Obrigada pela ajuda",
        "Tchau!",
    ]
    
    for entrada in conversas_teste:
        print(f"\nUsuário: {entrada}")
        resposta = bot.responder(entrada)
        print(f"Bot:     {resposta}")


def analisar_limitacoes():
    """
    Demonstra as limitações da abordagem simbólica baseada em regras.
    Isso motiva a necessidade de abordagens de ML nas próximas aulas.
    """
    print("\n" + "="*60)
    print("ANÁLISE DAS LIMITAÇÕES — Por que precisamos de ML?")
    print("="*60)
    
    bot = ChatbotSimples()
    
    casos_problematicos = [
        # Variações que humanos entenderiam mas o bot não
        ("O que é a IA?", "Deveria responder sobre IA (tem artigo 'a')"),
        ("Oi gente", "Saudação não reconhecida"),
        ("Quem inventou a inteligência artificial?", "Resposta sobre IA não capturada"),
        ("Explica o teste de turing pra mim", "Deveria explicar Turing"),
    ]
    
    for entrada, esperado in casos_problematicos:
        resposta = bot.responder(entrada)
        print(f"\nEntrada:  '{entrada}'")
        print(f"Esperado: {esperado}")
        print(f"Bot:      {resposta}")
        falhou = "resposta padrão" in resposta.lower() or "ainda não aprendi" in resposta
        print(f"Status:   {'❌ Falhou' if falhou else '✓ OK'}")
    
    print("\n" + "-"*60)
    print("CONCLUSÃO:")
    print("Um chatbot baseado em regras precisa de uma regra para CADA")
    print("variação possível. Para linguagem natural, isso é inviável.")
    print("Isso motivou o desenvolvimento de abordagens de ML (NLP).")
    print("Um LLM moderno (GPT-4, etc.) lida com todas essas variações")
    print("porque foi treinado em bilhões de exemplos de linguagem.")


# Ponto de entrada
if __name__ == "__main__":
    demonstrar_chatbot()
    analisar_limitacoes()
    
    # Descomente para modo interativo:
    # bot = ChatbotSimples()
    # bot.iniciar_conversa()
```

**Saída esperada (parcial):**
```
DEMONSTRAÇÃO DO CHATBOT SIMBÓLICO
============================================================

Usuário: Olá!
Bot:     Olá! Como posso ajudar?

Usuário: Meu nome é Maria
Bot:     Que nome bonito, Maria! Prazer em te conhecer.

Usuário: O que é Inteligência Artificial?
Bot:     Inteligência Artificial é o campo da ciência da computação que...

Usuário: O que é blockchain?
Bot:     Interessante! Mas ainda não aprendi sobre isso.
```

---

## 9. Linha do Tempo Resumida

```
1950  ● Turing — "Computing Machinery and Intelligence" / Teste de Turing
1956  ● Conferência de Dartmouth — nasce o termo "Inteligência Artificial"
1966  ● ELIZA — primeiro chatbot
1969  ● Crítica ao Perceptron (Minsky & Papert) → 1° Inverno se aproxima
1974  ● 1° Inverno da IA (1974-1980)
1980  ● Sistemas especialistas dominam (MYCIN, R1/XCON)
1986  ● Redescoberta do Backpropagation (Rumelhart, Hinton, Williams)
1987  ● 2° Inverno da IA (1987-1993)
1997  ● Deep Blue vence Kasparov no xadrez
2006  ● Hinton reintroduz "Deep Learning" com pré-treinamento de RBMs
2012  ● AlexNet vence ImageNet — início da era moderna do deep learning
2016  ● AlphaGo vence Lee Sedol no Go
2017  ● "Attention Is All You Need" — arquitetura Transformer
2018  ● BERT (Google) — estado da arte em NLP
2020  ● GPT-3 — 175B parâmetros, capacidades emergentes
2022  ● ChatGPT — IA generativa para o grande público
2023  ● GPT-4, Gemini, Llama 2, Claude 2 — corrida dos LLMs
2024  ● GPT-4o, Gemini 1.5, Claude 3 — multimodalidade e raciocínio
```

---

## 10. Questões para Reflexão

1. **Definição**: Ao comparar as quatro perspectivas de Russell & Norvig para definir IA (pensar/agir como humano vs. pensar/agir racionalmente), qual você considera mais adequada para guiar o desenvolvimento de sistemas práticos? Por quê?

2. **Teste de Turing**: O Teste de Turing ainda é um critério válido para avaliar inteligência em 2024, considerando que o ChatGPT pode enganar muitos humanos? O que o teste mede de fato?

3. **Quarto Chinês**: A crítica de Searle ao Teste de Turing é válida? Um LLM como o GPT-4 "compreende" o que diz, ou apenas manipula símbolos? O que seria necessário para uma "compreensão genuína"?

4. **Invernos da IA**: Quais lições dos invernos da IA do passado deveríamos aplicar às expectativas atuais sobre LLMs e AGI? Estamos em um novo ciclo de hype?

5. **Ética**: Cite uma aplicação atual de IA que você usa cotidianamente. Quais são os benefícios e os riscos éticos dessa aplicação? Como você acha que ela deveria ser regulada?

6. **Abordagens**: Por que a abordagem simbólica (sistemas especialistas) falhou em substituir completamente o conhecimento humano no século XX, mas a abordagem conexionista (deep learning) conseguiu resultados tão impressionantes no século XXI? O que mudou?

---

## Referências

**[1]** TURING, A. M. Computing Machinery and Intelligence. *Mind*, v. 59, n. 236, p. 433–460, 1950.

**[2]** SEARLE, J. R. Minds, Brains, and Programs. *Behavioral and Brain Sciences*, v. 3, n. 3, p. 417–424, 1980.

**[3]** RUSSELL, S.; NORVIG, P. **Inteligência Artificial: uma abordagem moderna**. 4. ed. Rio de Janeiro: GEN LTC, 2022. Cap. 1 (Introdução).

**[4]** FACELI, K. et al. **Inteligência Artificial: uma abordagem de aprendizado de máquina**. 2. ed. LTC, 2021. Cap. 1.

**[5]** GÉRON, A. **Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow**. 3. ed. Alta Books, 2023. Cap. 1.

**[6]** GOODFELLOW, I.; BENGIO, Y.; COURVILLE, A. **Deep Learning**. MIT Press, 2016. Cap. 1. Disponível em: https://www.deeplearningbook.org/

**[7]** VASWANI, A. et al. Attention Is All You Need. *NeurIPS*, 2017. arXiv:1706.03762.

**[8]** LECUN, Y.; BENGIO, Y.; HINTON, G. Deep Learning. *Nature*, v. 521, p. 436–444, 2015.

**[9]** KURZWEIL, R. **The Singularity Is Near**. Viking Press, 2005.

**[10]** WEIZENBAUM, J. ELIZA — A Computer Program for the Study of Natural Language Communication Between Man and Machine. *Communications of the ACM*, v. 9, n. 1, p. 36–45, 1966.

---

*Próxima aula: [Aula 02 — Agentes Inteligentes](./aula-02-agentes-inteligentes.md)*
