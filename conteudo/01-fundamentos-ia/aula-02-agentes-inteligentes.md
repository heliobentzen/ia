# Aula 02 — Agentes Inteligentes

> **Módulo:** 01 — Fundamentos de Inteligência Artificial  
> **Duração:** 45 minutos  
> **Pré-requisitos:** Aula 01 (História da IA e Conceitos Fundamentais)

---

## Objetivos de Aprendizagem

Ao final desta aula, o estudante será capaz de:

1. **Definir** formalmente o conceito de agente inteligente e seus componentes essenciais.
2. **Aplicar** o framework PEAS para especificar qualquer sistema de IA.
3. **Classificar** agentes nos cinco tipos principais, explicando as diferenças estruturais entre eles.
4. **Caracterizar** ambientes segundo suas propriedades (observabilidade, determinismo, etc.).
5. **Analisar** sistemas reais (Roomba, GPS, ChatGPT, carro autônomo) como agentes inteligentes.
6. **Implementar** um agente reflexivo simples em Python.

---

## 1. O Conceito de Agente

### 1.1 Definição Formal

> *"Um agente é qualquer coisa que pode ser vista como percebendo seu ambiente por meio de sensores e agindo sobre esse ambiente por meio de atuadores."*  
> — Russell & Norvig, 2022

Esta definição é deliberadamente ampla. Ela engloba:
- **Humanos**: sensores = olhos, ouvidos, pele, língua; atuadores = mãos, pernas, voz.
- **Robôs**: sensores = câmeras, microfones, LIDAR; atuadores = motores, braços.
- **Agentes de software**: sensores = teclado, mouse, dados de rede; atuadores = tela, arquivos, chamadas de API.

```
                    ┌─────────────────────────────────────┐
                    │              AGENTE                 │
                    │                                     │
   Percepções  ───→ │  [Sensores] → [Função Agente] → [Atuadores] ──→  Ações
                    │                                     │
                    └─────────────────────────────────────┘
                                      ↕
                              AMBIENTE (Mundo)
```

### 1.2 Percepção, Ação e Racionalidade

**Percepção**: A entrada atual que os sensores do agente fornecem. Uma única observação do ambiente em um dado momento.

**Sequência de percepções**: O histórico completo de tudo que o agente percebeu até agora. A função agente pode depender de toda essa sequência.

**Ação**: A saída que o agente produz para afetar o ambiente.

**Função agente**: `f: Sequência_de_percepções → Ação` — o mapeamento matemático que define o comportamento do agente.

**Programa agente**: A implementação concreta (código) da função agente.

### 1.3 O Que É Racionalidade?

> *"Um agente racional é aquele que faz a coisa certa — no sentido de que a ação que se espera que maximize sua medida de desempenho, dado o histórico de percepções e todo conhecimento embutido no agente."*  
> — Russell & Norvig, 2022

**Quatro fatores que determinam racionalidade:**
1. A medida de desempenho que define o critério de sucesso.
2. O conhecimento prévio do agente sobre o ambiente.
3. As ações que o agente pode executar.
4. A sequência de percepções do agente até o momento atual.

**Racionalidade ≠ Perfeição**: Um agente racional maximiza o desempenho **esperado** dado o que sabe. Ele não é omnisciente. Um agente que atravessa uma rua olhando nos dois lados age racionalmente, mesmo que possa ser atropelado por um carro que surgiu de trás.

**Racionalidade ≠ Onisciência**: Requer exploração (gathering information) quando útil.

**Agente Autônomo**: Um agente cuja capacidade de aprender e adaptar-se compensa o conhecimento incompleto do ambiente inicial.

---

## 2. Framework PEAS

PEAS é um framework para especificar completamente um agente inteligente. A sigla vem do inglês:

| Letra | Inglês | Português | Pergunta-chave |
|-------|--------|-----------|----------------|
| **P** | Performance Measure | Medida de Desempenho | Como sabemos se o agente está indo bem? |
| **E** | Environment | Ambiente | Onde o agente opera? |
| **A** | Actuators | Atuadores | Como o agente afeta o mundo? |
| **S** | Sensors | Sensores | Como o agente percebe o mundo? |

### 2.1 Exemplos de Especificação PEAS

#### Motorista de Táxi Autônomo

| Componente | Descrição |
|------------|-----------|
| **Desempenho** | Chegada segura no destino, minimizar tempo/combustível, obedecer leis de trânsito, maximizar conforto do passageiro |
| **Ambiente** | Ruas, tráfego, pedestres, outros veículos, semáforos, tempo (chuva/sol), rodovias e vias urbanas |
| **Atuadores** | Volante, acelerador, freio, buzina, seta, luz alta/baixa, comunicação |
| **Sensores** | Câmeras (360°), LIDAR, RADAR, GPS, velocímetro, acelerômetro, microfone |

#### Assistente Virtual (tipo Siri/Alexa)

| Componente | Descrição |
|------------|-----------|
| **Desempenho** | Entendimento correto da intenção, resposta útil e precisa, tempo de resposta, satisfação do usuário |
| **Ambiente** | Conversa com humanos, acesso à internet, dispositivos conectados (IoT), calendário, listas |
| **Atuadores** | Síntese de voz, envio de mensagens/emails, chamadas telefônicas, controle de dispositivos IoT |
| **Sensores** | Microfone, câmera (visual), APIs de internet, notificações do sistema, localização GPS |

#### Robô Aspirador (Roomba)

| Componente | Descrição |
|------------|-----------|
| **Desempenho** | % do chão limpo, tempo para limpar, bateria consumida, evitar danos/quedas |
| **Ambiente** | Cômodos de casa: chão plano, tapetes, obstáculos (cadeiras, brinquedos), bordas, degraus |
| **Atuadores** | Motores de rodas, escova rotativa, ventilador de sucção, descarga de sujeira |
| **Sensores** | Sensor de colisão (bumper), sensor de degrau (cliff), sensor de parede, sensor de sujeira, câmera |

#### Sistema de Recomendação (Netflix)

| Componente | Descrição |
|------------|-----------|
| **Desempenho** | Taxa de clique, tempo de visualização, taxa de cancelamento (churn), satisfação do usuário |
| **Ambiente** | Catálogo de conteúdo, histórico de usuário, dados demográficos, horário, dispositivo |
| **Atuadores** | Interface de usuário: ranking de filmes/séries, thumbnails, trailers automáticos |
| **Sensores** | Histórico de visualização, avaliações, pesquisas, tempo assistido, pausas, avançar/retroceder |

#### Agente de Xadrez (como Stockfish)

| Componente | Descrição |
|------------|-----------|
| **Desempenho** | Vencer a partida; maximizar vantagem posicional |
| **Ambiente** | Tabuleiro de xadrez 8×8, peças, regras do jogo, adversário humano/computador |
| **Atuadores** | Seleção e execução de movimentos válidos |
| **Sensores** | Estado atual do tabuleiro (posição de todas as peças) |

---

## 3. Tipos de Agentes

Russell & Norvig identificam cinco tipos de agentes, em ordem crescente de sofisticação e capacidade:

### 3.1 Agente Reflexivo Simples

**Princípio**: Age com base apenas na percepção **atual** — ignora o histórico.

```
Percepção atual → [Regras SE-ENTÃO] → Ação
```

**Estrutura interna:**
```python
def agente_reflexivo_simples(percepção):
    for condição, ação in tabela_de_regras:
        if condição(percepção):
            return ação
    return ação_padrão
```

**Características:**
- Muito simples e eficiente.
- Funciona bem apenas em ambientes **completamente observáveis**.
- Sem memória: trata cada instante independentemente.
- Vulnerável a loops (pode ficar girando em círculos).

**Exemplos reais:**
- Termostato: SE temperatura < 20°C → LIGAR aquecedor.
- Controle de semáforo simples: SE timer = 0 → TROCAR sinal.
- Filtro de spam básico: SE contém "GRÁTIS" E "CLIQUE AQUI" → SPAM.

### 3.2 Agente Baseado em Modelo (Model-Based Reflex Agent)

**Princípio**: Mantém um **estado interno** que representa aspectos do mundo que não são diretamente observáveis. Usa um modelo do mundo para interpretar percepções incompletas.

```
Percepção atual + Estado interno → [Modelo do mundo] → Ação
                        ↑
               (atualizado a cada passo)
```

**Dois modelos necessários:**
1. **Modelo de transição**: Como o mundo muda independentemente do agente? (Ex: outros carros se movem)
2. **Modelo sensor**: O que a percepção atual diz sobre o estado do mundo? (Ex: câmera mostra parede à esquerda)

**Características:**
- Lida com ambientes **parcialmente observáveis**.
- Mantém histórico relevante.
- Mais eficiente que manter toda a sequência de percepções.

**Exemplos reais:**
- Roomba com mapa mental dos cômodos já limpos.
- Carro autônomo rastreando pedestres que saem temporariamente do campo de visão.
- Sistema de detecção de fraude que considera histórico de transações.

### 3.3 Agente Baseado em Objetivos

**Princípio**: Além do estado atual, tem informação sobre situações **desejáveis** (objetivos/metas). Raciocina sobre ações futuras para alcançar o objetivo.

```
Estado atual + Objetivo + [Busca/Planejamento] → Sequência de ações
```

**Características:**
- Mais flexível: o mesmo agente com objetivos diferentes comporta-se diferentemente.
- Requer busca (como estudado na Aula 03) ou planejamento.
- Pode raciocinar "se eu fizer X, chegarei ao objetivo Y?".

**Exemplos reais:**
- GPS: objetivo = chegar ao destino. Planeja rota completa.
- Agente de xadrez: objetivo = checkmate. Busca sequências de movimentos.
- Robot planejador: objetivo = montar peça. Planeja sequência de manipulações.

**Limitação**: Objetivos são binários (atingido/não-atingido). Não lida bem com trade-offs.

### 3.4 Agente Baseado em Utilidade

**Princípio**: Substitui objetivos binários por uma **função de utilidade** — uma medida numérica de quão desejável é cada estado.

```
Estados possíveis + Utilidade de cada estado → Ação que maximiza utilidade esperada
```

**Por que utilidade é melhor que objetivo?**
- Objetivos conflitantes: rapidez vs. segurança vs. conforto (táxi autônomo).
- Objetivos parcialmente atingíveis: "chegar ao destino" vs. "chegar ao destino mais cedo".
- Incerteza: ação A leva ao objetivo com 90% de probabilidade; ação B com 60%.

**Teoria da Utilidade Esperada (von Neumann, 1944):**
```
EU(ação) = Σ P(resultado_i | ação) × U(resultado_i)
```

Um agente racional escolhe a ação que maximiza a utilidade esperada.

**Exemplos reais:**
- Portfólio de investimentos: maximiza retorno ajustado ao risco.
- Sistema de recomendação: maximiza engajamento esperado (cliques × retenção).
- Robô logístico: minimiza tempo de entrega considerando incerteza no tráfego.

### 3.5 Agente que Aprende (Learning Agent)

**Princípio**: Começa com conhecimento parcial ou nulo e melhora seu desempenho com a experiência.

**Arquitetura de quatro componentes:**

```
┌───────────────────────────────────────────────────────────────────┐
│                      AGENTE QUE APRENDE                           │
│                                                                   │
│  Crítico ──── avalia desempenho ────→ Elemento de Aprendizado     │
│     ↑                                        │                    │
│     │ percepções                     muda regras/weights          │
│     │                                        ↓                    │
│  Padrão de                         Elemento de Desempenho         │
│  Desempenho                         (executa ações)               │
│                                        │                          │
│                             Gerador de Problemas                  │
│                             (sugere ações exploratórias)          │
└───────────────────────────────────────────────────────────────────┘
```

**Quatro componentes:**
1. **Elemento de desempenho**: Seleciona ações (os tipos de agentes anteriores).
2. **Crítico**: Avalia quão bem o agente está se saindo segundo o padrão de desempenho.
3. **Elemento de aprendizado**: Usa feedback do crítico para melhorar o elemento de desempenho.
4. **Gerador de problemas**: Sugere ações exploratórias para aprender mais.

**Exemplos reais:**
- AlphaGo: começa com exemplos humanos e melhora jogando contra si mesmo.
- Sistema de recomendação que aprende preferências com o tempo.
- Filtro de spam que aprende com emails marcados pelo usuário.
- Carro autônomo que melhora com quilômetros rodados.

---

## 4. Propriedades dos Ambientes

A escolha do tipo de agente adequado depende das propriedades do ambiente. Russell & Norvig identificam seis dimensões:

### 4.1 Completamente vs. Parcialmente Observável

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Completamente observável** | O agente tem acesso ao estado completo do mundo a cada instante | Xadrez (vê todas as peças) |
| **Parcialmente observável** | Sensores dão visão incompleta ou com ruído | Pôquer (cartas escondidas), direção (veículos atrás) |

**Implicação**: Ambientes parcialmente observáveis requerem agentes baseados em modelo para manter estado interno.

### 4.2 Determinístico vs. Estocástico (vs. Não-determinístico)

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Determinístico** | O próximo estado é completamente determinado pelo estado atual + ação | Xadrez (regras fixas) |
| **Estocástico** | Há incerteza quantificável (probabilidades) no próximo estado | Jogar dados, diagnóstico médico |
| **Não-determinístico** | Incerteza não quantificada | Negociação humana |

**Implicação**: Ambientes estocásticos requerem raciocínio probabilístico (agentes bayesianos).

### 4.3 Episódico vs. Sequencial

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Episódico** | Cada episódio é independente — ações atuais não afetam episódios futuros | Classificação de imagens (cada foto é independente) |
| **Sequencial** | Ações atuais afetam decisões futuras | Xadrez, direção autônoma, negociação |

**Implicação**: Ambientes sequenciais são muito mais complexos — requerem planejamento de longo prazo.

### 4.4 Estático vs. Dinâmico (vs. Semi-dinâmico)

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Estático** | O ambiente não muda enquanto o agente delibera | Palavras cruzadas |
| **Dinâmico** | O ambiente muda independentemente enquanto o agente pensa | Trânsito urbano, mercado financeiro |
| **Semi-dinâmico** | O ambiente não muda, mas o desempenho do agente muda com o tempo | Xadrez cronometrado |

**Implicação**: Ambientes dinâmicos exigem decisão rápida — não há tempo para busca exaustiva.

### 4.5 Discreto vs. Contínuo

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Discreto** | Número finito de estados e ações claramente distintos | Xadrez (posições discretas) |
| **Contínuo** | Estados e/ou ações são valores contínuos | Direção (ângulo do volante ∈ ℝ) |

**Implicação**: Ambientes contínuos geralmente requerem técnicas diferentes (controle, otimização).

### 4.6 Agente Único vs. Multiagente

| Tipo | Descrição | Exemplo |
|------|-----------|---------|
| **Agente único** | Um único agente opera no ambiente | Palavras cruzadas, Minesweeper |
| **Multiagente cooperativo** | Múltiplos agentes com objetivo comum | Robótica em enxame, formação de times |
| **Multiagente competitivo** | Múltiplos agentes com objetivos conflitantes | Xadrez, pôquer, leilão, trânsito |

**Implicação**: Ambientes multiagente requerem raciocínio estratégico sobre outros agentes (Teoria dos Jogos).

### 4.7 Resumo por Aplicação

| Ambiente | Observável | Determinístico | Episódico | Estático | Discreto | Agentes |
|----------|-----------|----------------|-----------|----------|----------|---------|
| Palavras cruzadas | Completamente | Determinístico | Sequencial | Estático | Discreto | Único |
| Xadrez cronometrado | Completamente | Determinístico | Sequencial | Semi-din. | Discreto | Multi |
| Pôquer | Parcialmente | Estocástico | Sequencial | Estático | Discreto | Multi |
| Táxi autônomo | Parcialmente | Estocástico | Sequencial | Dinâmico | Contínuo | Multi |
| Diagnóstico médico | Parcialmente | Estocástico | Sequencial | Estático | Discreto | Único |
| Análise de imagens | Completamente | Determinístico | Episódico | Estático | Contínuo | Único |

---

## 5. Exemplos Práticos de Agentes

### 5.1 Roomba — Agente Baseado em Modelo

```
PEAS:
- Desempenho: % chão limpo, eficiência energética, sem danos
- Ambiente: cômodo 2D, poeira, obstáculos, bordas, tapetes
- Atuadores: motores (avançar/girar/recuar), aspirador
- Sensores: bumper (colisão), cliff (queda), dirt detector, tempo

TIPO: Baseado em modelo (Roomba com mapeamento)
AMBIENTE: Parcialmente observável, determinístico, sequencial, dinâmico, contínuo, único
```

**Comportamento típico de um Roomba com mapeamento:**
1. Mapeamento inicial do cômodo (exploração em espiral).
2. Manutenção de mapa interno com áreas limpas/sujas/bloqueadas.
3. Planejamento de rota para cobertura eficiente.
4. Retorno à base quando bateria baixa.

### 5.2 Assistente Virtual (Siri/Google Assistant/Alexa)

```
PEAS:
- Desempenho: precisão da resposta, satisfação do usuário, tempo de resposta
- Ambiente: conversação, dispositivos IoT, serviços web, calendários
- Atuadores: voz sintetizada, mensagens, controle de dispositivos, pesquisa web
- Sensores: microfone, APIs de serviços web, notificações do sistema

TIPO: Agente que aprende (personalização ao usuário ao longo do tempo)
AMBIENTE: Parcialmente observável, estocástico, sequencial, dinâmico, discreto, multi
```

### 5.3 Carro Autônomo (Nível 4 — Waymo)

```
PEAS:
- Desempenho: chegar ao destino sem acidentes, tempo de trajeto, conforto, legalidade
- Ambiente: vias públicas, outros veículos, pedestres, semáforos, sinalização, clima
- Atuadores: volante, acelerador, freio, buzina, seta, comunicação V2X
- Sensores: 29 câmeras, 5 LIDARs, 6 RADARs, 4 sonares, GPS/IMU, velocímetro

TIPO: Agente baseado em utilidade + aprendizado
AMBIENTE: Parcialmente observável, estocástico, sequencial, dinâmico, contínuo, multi
```

**Pilha de software do Waymo (simplificada):**
```
1. Percepção: CNN detecta e classifica objetos (veículo, pedestre, etc.)
2. Previsão: RNN prevê trajetórias de objetos detectados
3. Planejamento: otimização de trajetória via MPC + RL
4. Controle: PID converte trajetória em comandos de volante/aceleração
```

### 5.4 Agente de Xadrez (Stockfish + NNUE)

```
PEAS:
- Desempenho: vencer a partida (maximizar probabilidade de vitória)
- Ambiente: tabuleiro 8×8, 32 peças, 20 movimentos iniciais possíveis, ~10^44 posições
- Atuadores: escolha e execução de movimento válido
- Sensores: posição completa do tabuleiro

TIPO: Baseado em utilidade (minimax com poda α-β + avaliação por rede neural)
AMBIENTE: Completamente observável, determinístico, sequencial, estático, discreto, multi
```

---

## 6. Implementação em Python

```python
"""
Implementação de diferentes tipos de agentes em Python.
Demonstra a hierarquia: reflexivo simples → baseado em modelo → objetivo → utilidade.

Domínio de exemplo: Mundo do Aspirador (Vacuum World)
- Grade 2D simples (N × M)
- Cada célula: limpa ou suja
- Agente percebe: localização atual + status (limpo/sujo)
- Ações: aspirar, mover_esquerda, mover_direita, mover_cima, mover_baixo
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict
from enum import Enum
import random


# ─── Definições do domínio ────────────────────────────────────────────────────

class Acao(Enum):
    ASPIRAR = "aspirar"
    ESQUERDA = "esquerda"
    DIREITA = "direita"
    CIMA = "cima"
    BAIXO = "baixo"
    PARAR = "parar"


@dataclass
class Percepcao:
    """O que o agente percebe em cada passo."""
    localizacao: Tuple[int, int]  # (linha, coluna)
    status: str                   # "sujo" ou "limpo"


@dataclass
class AmbienteAspirador:
    """
    Ambiente 2D para o problema do aspirador de pó.
    Demonstra o ambiente em que os agentes operam.
    """
    linhas: int = 2
    colunas: int = 2
    pos_agente: Tuple[int, int] = (0, 0)
    # True = sujo, False = limpo
    grade: Dict[Tuple[int, int], bool] = field(default_factory=dict)

    def __post_init__(self):
        if not self.grade:
            # Inicializa com sujeira aleatória
            random.seed(42)
            self.grade = {
                (r, c): random.random() < 0.7  # 70% de chance de sujo
                for r in range(self.linhas)
                for c in range(self.colunas)
            }

    @classmethod
    def from_grid(cls, grade_inicial: List[List[bool]],
                  pos_inicial: Tuple[int, int] = (0, 0)) -> "AmbienteAspirador":
        """Cria ambiente a partir de uma grade explícita."""
        linhas = len(grade_inicial)
        colunas = len(grade_inicial[0])
        grade = {
            (r, c): grade_inicial[r][c]
            for r in range(linhas)
            for c in range(colunas)
        }
        return cls(linhas=linhas, colunas=colunas, pos_agente=pos_inicial, grade=grade)

    def perceber(self) -> Percepcao:
        """Retorna o que o agente percebe na posição atual."""
        r, c = self.pos_agente
        sujo = self.grade.get((r, c), False)
        return Percepcao(
            localizacao=self.pos_agente,
            status="sujo" if sujo else "limpo"
        )

    def executar(self, acao: Acao) -> bool:
        """
        Executa uma ação no ambiente.
        
        Returns:
            True se a ação foi válida, False caso contrário
        """
        r, c = self.pos_agente
        
        if acao == Acao.ASPIRAR:
            self.grade[(r, c)] = False  # limpa
            return True
        elif acao == Acao.ESQUERDA and c > 0:
            self.pos_agente = (r, c - 1)
            return True
        elif acao == Acao.DIREITA and c < self.colunas - 1:
            self.pos_agente = (r, c + 1)
            return True
        elif acao == Acao.CIMA and r > 0:
            self.pos_agente = (r - 1, c)
            return True
        elif acao == Acao.BAIXO and r < self.linhas - 1:
            self.pos_agente = (r + 1, c)
            return True
        
        return False  # ação inválida (tentou sair dos limites)

    def esta_limpo(self) -> bool:
        """Verifica se todo o ambiente está limpo."""
        return not any(self.grade.values())

    def celulas_sujas(self) -> List[Tuple[int, int]]:
        """Retorna lista de células sujas."""
        return [pos for pos, sujo in self.grade.items() if sujo]

    def exibir(self, titulo: str = ""):
        """Exibe o estado atual do ambiente."""
        if titulo:
            print(f"\n{titulo}")
        for r in range(self.linhas):
            linha = ""
            for c in range(self.colunas):
                sujo = self.grade.get((r, c), False)
                eh_agente = self.pos_agente == (r, c)
                if eh_agente and sujo:
                    linha += "[A*]"  # agente em célula suja
                elif eh_agente:
                    linha += "[A ]"  # agente em célula limpa
                elif sujo:
                    linha += "[ *]"  # célula suja
                else:
                    linha += "[  ]"  # célula limpa
            print(linha)
        print(f"  (* = sujo, A = agente)")


# ─── Tipo 1: Agente Reflexivo Simples ────────────────────────────────────────

class AgenteReflexivoSimples:
    """
    Agente mais básico: decide apenas com base na percepção atual.
    Não tem memória, não planeja, não aprende.
    
    Regras:
    - SE sujo → aspirar
    - SE limpo E pode ir direita → ir direita
    - SE limpo E pode ir baixo → ir baixo
    - SENÃO → parar
    """
    
    def __init__(self, linhas: int, colunas: int):
        self.linhas = linhas
        self.colunas = colunas
    
    def agir(self, percepcao: Percepcao) -> Acao:
        """
        Função agente: percepção → ação (sem memória).
        
        Esta é a implementação da função f: P → A
        onde P é a percepção atual e A é a ação.
        """
        r, c = percepcao.localizacao
        
        # Regra 1: SE sujo → limpar (prioridade máxima)
        if percepcao.status == "sujo":
            return Acao.ASPIRAR
        
        # Regra 2: mover para varrer todo o espaço (estratégia zigzag)
        # linha par: vai para direita
        if r % 2 == 0:
            if c < self.colunas - 1:
                return Acao.DIREITA
            elif r < self.linhas - 1:
                return Acao.BAIXO
        # linha ímpar: vai para esquerda
        else:
            if c > 0:
                return Acao.ESQUERDA
            elif r < self.linhas - 1:
                return Acao.BAIXO
        
        return Acao.PARAR


# ─── Tipo 2: Agente Baseado em Modelo ────────────────────────────────────────

class AgenteBaseadoEmModelo:
    """
    Agente com estado interno (memória).
    Mantém um mapa das células já visitadas e limpas,
    evitando re-visitar células desnecessariamente.
    
    MODELO DO MUNDO: grade interna que rastreia o estado de cada célula.
    """
    
    def __init__(self, linhas: int, colunas: int):
        self.linhas = linhas
        self.colunas = colunas
        
        # ESTADO INTERNO: o que o agente "sabe" sobre o mundo
        self.celulas_limpas: set = set()
        self.celulas_visitadas: set = set()
        self.historico_acoes: List[Acao] = []
        self.passos = 0
    
    def _atualizar_modelo(self, percepcao: Percepcao):
        """Atualiza o estado interno com base na nova percepção."""
        self.celulas_visitadas.add(percepcao.localizacao)
        if percepcao.status == "limpo":
            self.celulas_limpas.add(percepcao.localizacao)
        else:
            # Sabe que está suja, mas após aspirar será limpa
            self.celulas_limpas.discard(percepcao.localizacao)
    
    def _proxima_celula_nao_visitada(self, pos: Tuple[int, int]) -> Optional[Acao]:
        """
        Encontra uma ação para ir a uma célula não visitada.
        Usa busca simples (BFS) internamente.
        """
        r, c = pos
        movimentos = [
            (Acao.DIREITA, (r, c + 1)),
            (Acao.BAIXO, (r + 1, c)),
            (Acao.ESQUERDA, (r, c - 1)),
            (Acao.CIMA, (r - 1, c)),
        ]
        
        for acao, (nr, nc) in movimentos:
            if (0 <= nr < self.linhas and
                0 <= nc < self.colunas and
                (nr, nc) not in self.celulas_visitadas):
                return acao
        
        # Nenhuma célula não-visitada adjacente
        return None
    
    def agir(self, percepcao: Percepcao) -> Acao:
        """
        Função agente com estado interno (memória do modelo).
        A decisão usa percepção atual + estado interno.
        """
        self.passos += 1
        
        # 1. Atualizar o modelo interno
        self._atualizar_modelo(percepcao)
        
        # 2. Se sujo → aspirar e registrar
        if percepcao.status == "sujo":
            self.celulas_limpas.add(percepcao.localizacao)
            self.historico_acoes.append(Acao.ASPIRAR)
            return Acao.ASPIRAR
        
        # 3. Se há célula não visitada → ir para ela
        proxima = self._proxima_celula_nao_visitada(percepcao.localizacao)
        if proxima:
            self.historico_acoes.append(proxima)
            return proxima
        
        # 4. Todas as células foram visitadas → parar
        return Acao.PARAR
    
    def estatisticas(self) -> dict:
        """Retorna estatísticas sobre o desempenho do agente."""
        return {
            "passos_totais": self.passos,
            "celulas_visitadas": len(self.celulas_visitadas),
            "celulas_limpas": len(self.celulas_limpas),
            "acoes_por_tipo": {
                acao.value: self.historico_acoes.count(acao)
                for acao in Acao
            }
        }


# ─── Simulação ────────────────────────────────────────────────────────────────

def simular_agente(agente, ambiente: AmbienteAspirador,
                   max_passos: int = 50, verbose: bool = True) -> dict:
    """
    Executa a simulação de um agente em um ambiente.
    
    Args:
        agente: Instância de qualquer tipo de agente
        ambiente: Ambiente a ser limpo
        max_passos: Limite de passos para evitar loops infinitos
        verbose: Se True, exibe estado a cada passo
    
    Returns:
        Dicionário com métricas de desempenho
    """
    nome_agente = type(agente).__name__
    passos = 0
    acoes_realizadas = []
    celulas_sujas_inicial = len(ambiente.celulas_sujas())
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"SIMULAÇÃO: {nome_agente}")
        print(f"{'='*50}")
        ambiente.exibir(f"Estado inicial ({celulas_sujas_inicial} células sujas):")
    
    while not ambiente.esta_limpo() and passos < max_passos:
        # 1. Agente percebe o ambiente
        percepcao = ambiente.perceber()
        
        # 2. Agente decide ação
        acao = agente.agir(percepcao)
        
        if acao == Acao.PARAR:
            if verbose:
                print(f"\nAgente decidiu PARAR no passo {passos}")
            break
        
        # 3. Ambiente executa a ação
        valido = ambiente.executar(acao)
        acoes_realizadas.append(acao)
        passos += 1
        
        if verbose and passos <= 10:
            status_icone = "💧" if percepcao.status == "sujo" else "✓"
            print(f"Passo {passos:2d}: {status_icone} {percepcao.localizacao} → {acao.value}"
                  f"{'(inválido)' if not valido else ''}")
    
    celulas_sujas_final = len(ambiente.celulas_sujas())
    celulas_limpas = celulas_sujas_inicial - celulas_sujas_final
    
    if verbose:
        ambiente.exibir(f"\nEstado final (após {passos} passos):")
        print(f"\n📊 MÉTRICAS:")
        print(f"   Passos realizados:    {passos}")
        print(f"   Células limpas:       {celulas_limpas}/{celulas_sujas_inicial}")
        print(f"   Eficiência:           {celulas_limpas/max(passos,1):.2f} células/passo")
        print(f"   Ambiente limpo:       {'SIM ✓' if ambiente.esta_limpo() else 'NÃO ✗'}")
    
    return {
        "nome": nome_agente,
        "passos": passos,
        "celulas_limpas": celulas_limpas,
        "celulas_sujas_inicial": celulas_sujas_inicial,
        "ambiente_limpo": ambiente.esta_limpo(),
        "eficiencia": celulas_limpas / max(passos, 1),
    }


def comparar_agentes():
    """
    Compara os dois tipos de agentes no mesmo ambiente.
    Demonstra como o estado interno melhora a eficiência.
    """
    # Grade com sujeira específica para comparação justa
    grade = [
        [True,  False, True,  True],
        [False, True,  False, True],
        [True,  True,  False, False],
        [False, False, True,  True],
    ]
    
    print("\n" + "="*60)
    print("COMPARAÇÃO DE AGENTES — Mesmo ambiente, abordagens diferentes")
    print("="*60)
    
    resultados = []
    
    for AgenteTipo in [AgenteReflexivoSimples, AgenteBaseadoEmModelo]:
        # Recria o ambiente para que ambos comecem igual
        ambiente = AmbienteAspirador.from_grid(grade, pos_inicial=(0, 0))
        
        if AgenteTipo == AgenteReflexivoSimples:
            agente = AgenteTipo(linhas=4, colunas=4)
        else:
            agente = AgenteTipo(linhas=4, colunas=4)
        
        resultado = simular_agente(agente, ambiente, max_passos=100, verbose=True)
        resultados.append(resultado)
    
    # Tabela comparativa
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO")
    print("="*60)
    print(f"{'Métrica':<30} {'Reflexivo':>12} {'Modelo':>12}")
    print("-"*55)
    for metrica in ["passos", "celulas_limpas", "ambiente_limpo"]:
        v1 = resultados[0][metrica]
        v2 = resultados[1][metrica]
        print(f"{metrica:<30} {str(v1):>12} {str(v2):>12}")
    print(f"{'eficiencia (células/passo)':<30} "
          f"{resultados[0]['eficiencia']:>12.3f} "
          f"{resultados[1]['eficiencia']:>12.3f}")
    
    print("\n💡 O agente baseado em modelo é mais eficiente porque:")
    print("   - Não re-visita células que já sabe estar limpas")
    print("   - Tem um plano de cobertura baseado no mapa interno")
    print("   - A vantagem cresce com o tamanho do ambiente")


def demonstrar_peas():
    """Demonstra o framework PEAS para o ambiente do aspirador."""
    print("\n" + "="*60)
    print("ESPECIFICAÇÃO PEAS — Agente Aspirador de Pó")
    print("="*60)
    
    peas = {
        "Medida de Desempenho": [
            "Pontuação = +10 por célula limpa, -1 por passo realizado",
            "Penalidade por passar em célula já limpa",
            "Bônus por concluir antes do limite de passos"
        ],
        "Ambiente": [
            "Grade 2D N×M de células",
            "Cada célula: limpa (False) ou suja (True)",
            "Estático: sujeira não aparece durante operação",
            "Determinístico: ações têm efeito garantido"
        ],
        "Atuadores": [
            "Motor: ESQUERDA, DIREITA, CIMA, BAIXO",
            "Aspirador: ASPIRAR (limpa célula atual)"
        ],
        "Sensores": [
            "Localizador: coordenada (linha, coluna) atual",
            "Sensor de sujeira: 'sujo' ou 'limpo' na célula atual"
        ]
    }
    
    for componente, descricoes in peas.items():
        print(f"\n[{componente[0]}] {componente}:")
        for desc in descricoes:
            print(f"   • {desc}")
    
    print("\n🌍 Propriedades do Ambiente:")
    props = [
        ("Observabilidade",    "Completamente observável (vê localização + status atual)"),
        ("Determinismo",       "Determinístico (aspirar sempre limpa, mover sempre move)"),
        ("Episodicidade",      "Sequencial (estado anterior importa)"),
        ("Dinamismo",          "Estático (sujeira não muda durante operação)"),
        ("Dimensionalidade",   "Discreto (grade finita de células)"),
        ("Número de agentes",  "Único agente no ambiente"),
    ]
    for prop, desc in props:
        print(f"   {prop:<22}: {desc}")


# ─── Ponto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    demonstrar_peas()
    comparar_agentes()
```

---

## 7. Questões para Reflexão

1. **PEAS na prática**: Escolha um sistema tecnológico que você usa diariamente (app de banco, GPS, feed de redes sociais) e elabore a especificação PEAS completa. Quais aspectos do PEAS foram mais difíceis de especificar? Por quê?

2. **Racionalidade vs. Perfeição**: Um médico que segue os melhores protocolos médicos disponíveis, mas cujo paciente morre por uma condição raríssima que os protocolos não cobriam, agiu racionalmente? Relacione isso ao conceito de agente racional de Russell & Norvig.

3. **Tipo de agente**: Por que um carro autônomo não pode ser apenas um agente reflexivo simples? Cite ao menos três situações de direção que demonstram a necessidade de memória (estado interno) e/ou planejamento.

4. **Ambientes**: Compare o ambiente de um agente de xadrez com o de um assistente virtual (Alexa). Para cada uma das seis dimensões (observabilidade, determinismo, etc.), explique como elas diferem e como isso afeta a arquitetura necessária para cada agente.

5. **Limites da racionalidade**: Herbert Simon introduziu o conceito de "racionalidade limitada" (bounded rationality) — os agentes tomam decisões "boas o suficiente" (satisficing) em vez de ótimas, devido a limitações cognitivas e computacionais. Como esse conceito se aplica a agentes de IA? Você consegue pensar em um exemplo de sistema de IA que usa racionalidade limitada conscientemente?

6. **Ética de agentes**: Um agente de crédito automático (sistema que aprova ou rejeita pedidos de empréstimo) pode ser especificado via PEAS. Qual seria a medida de desempenho "correta"? Apenas lucratividade? E inclusão financeira, equidade racial, acesso a crédito para populações marginalizadas? Como conflitos entre esses objetivos devem ser resolvidos?

---

## Referências

**[1]** RUSSELL, S.; NORVIG, P. **Inteligência Artificial: uma abordagem moderna**. 4. ed. Rio de Janeiro: GEN LTC, 2022. Cap. 2 (Agentes Inteligentes).

**[2]** FACELI, K. et al. **Inteligência Artificial: uma abordagem de aprendizado de máquina**. 2. ed. LTC, 2021. Cap. 1.

**[3]** GÉRON, A. **Mãos à Obra: Aprendizado de Máquina com Scikit-Learn, Keras & TensorFlow**. 3. ed. Alta Books, 2023. Cap. 1.

**[4]** WOOLDRIDGE, M. **An Introduction to MultiAgent Systems**. 2. ed. Wiley, 2009. Cap. 1-2.

**[5]** SIMON, H. A. **The Sciences of the Artificial**. 3. ed. MIT Press, 1996.

**[6]** BROOKS, R. A. Intelligence Without Representation. *Artificial Intelligence*, v. 47, n. 1-3, p. 139-159, 1991.
> *Artigo provocativo argumentando que inteligência não requer representação interna explícita.*

**[7]** SUTTON, R. S.; BARTO, A. G. **Reinforcement Learning: An Introduction**. 2. ed. MIT Press, 2018. Cap. 3.
> *Define formalmente o framework de agente-ambiente para aprendizado por reforço.*

---

*Aula anterior: [Aula 01 — História da IA e Conceitos Fundamentais](./aula-01-historia-e-conceitos.md)*  
*Próxima aula: [Aula 03 — Busca e Resolução de Problemas](./aula-03-busca-e-resolucao-problemas.md)*
