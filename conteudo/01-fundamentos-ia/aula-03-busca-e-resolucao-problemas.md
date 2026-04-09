# Aula 03 — Busca e Resolução de Problemas

> **Módulo:** 01 — Fundamentos de Inteligência Artificial  
> **Duração:** 45 minutos  
> **Pré-requisitos:** Aulas 01 e 02 (Conceitos de IA e Agentes Inteligentes)

---

## Objetivos de Aprendizagem

Ao final desta aula, o estudante será capaz de:

1. **Formular** um problema de resolução de problemas como um espaço de estados.
2. **Implementar** os algoritmos BFS, DFS, Busca de Custo Uniforme e Aprofundamento Iterativo.
3. **Implementar** os algoritmos de busca informada: Busca Gulosa e A*.
4. **Verificar** se uma heurística é admissível e/ou consistente.
5. **Comparar** algoritmos de busca segundo completude, otimalidade, complexidade temporal e espacial.
6. **Aplicar** os algoritmos ao problema do labirinto, quebra-cabeça de 8 peças e N-rainhas.

---

## 1. Formulação de Problemas como Espaço de Estados

### 1.1 O Paradigma da Resolução de Problemas

Muitas tarefas de IA podem ser modeladas como **busca em espaço de estados**:

> *"Dado um estado inicial e um objetivo, encontre uma sequência de ações que leve do estado inicial ao objetivo."*

Este framework é surpreendentemente geral:
- Jogar xadrez = busca pela sequência de movimentos que leva ao xeque-mate.
- GPS = busca pela rota que leva do ponto A ao ponto B.
- Robô planejador = busca pela sequência de movimentos que monta uma peça.
- Resolver um cubo mágico = busca pela sequência de rotações que soluciona o cubo.

### 1.2 Componentes Formais de um Problema

Um problema de busca é definido por cinco componentes:

```
Problema = (S₀, AÇÕES, RESULTADO, OBJETIVO?, CUSTO)
```

| Componente | Notação | Descrição | Exemplo (GPS) |
|------------|---------|-----------|---------------|
| **Estado inicial** | S₀ | Estado em que o agente começa | Rua A, posição GPS atual |
| **Ações** | AÇÕES(s) | Conjunto de ações disponíveis no estado s | Seguir reto, virar esquerda, virar direita |
| **Modelo de transição** | RESULTADO(s, a) | Estado resultante de aplicar ação a ao estado s | Após virar esquerda na Rua A → Av. B |
| **Teste de objetivo** | OBJETIVO?(s) | Verifica se s é o estado objetivo | s == destino escolhido? |
| **Função de custo** | CUSTO(s, a, s') | Custo de aplicar ação a no estado s | Distância em km, tempo em min |

**Espaço de estados**: O grafo implícito formado por todos os estados alcançáveis a partir de S₀ via AÇÕES e RESULTADO.

**Solução**: Uma sequência de ações que leva de S₀ a algum estado que satisfaz OBJETIVO?.

**Solução ótima**: A solução com menor custo total de caminho.

### 1.3 Exemplo: O Problema do Labirinto

```
Labirinto 5×5:
┌─────────────────────┐
│ S  .  #  .  .  .   │   S = início (0,0)
│ .  #  #  .  #  .   │   G = objetivo (4,4)
│ .  .  .  .  #  .   │   # = parede (bloqueado)
│ #  #  .  #  .  .   │   . = passagem livre
│ .  .  .  .  .  G   │
└─────────────────────┘

Estado: (linha, coluna)
Ações: {cima, baixo, esquerda, direita} (se não houver parede)
Custo: 1 por passo
```

**Espaço de estados do labirinto**: Grafo com até N×M nós (células livres), cada um conectado a seus vizinhos livres.

### 1.4 Exemplo: Quebra-Cabeça de 8 Peças

```
Estado inicial:    Estado objetivo:
┌───────────┐      ┌───────────┐
│ 7 │ 2 │ 4 │      │ 1 │ 2 │ 3 │
├───┼───┼───┤      ├───┼───┼───┤
│ 5 │   │ 6 │      │ 4 │ 5 │ 6 │
├───┼───┼───┤      ├───┼───┼───┤
│ 8 │ 3 │ 1 │      │ 7 │ 8 │   │
└───────────┘      └───────────┘
```

- **Estado**: tupla de 9 elementos (posições das peças + espaço vazio).
- **Ações**: mover a peça adjacente ao espaço vazio (equivalente a mover o espaço).
- **Espaço de estados**: 9! = 362.880 estados possíveis (metade alcançáveis de qualquer estado inicial).
- **Profundidade ótima típica**: 20-30 movimentos.

---

## 2. Busca Não-Informada (Cega)

Algoritmos de busca não-informada **não usam informação sobre a posição do objetivo** — apenas exploram o espaço de estados sistematicamente.

### 2.1 Busca em Largura (BFS — Breadth-First Search)

**Ideia**: Expande todos os nós de profundidade d antes de expandir qualquer nó de profundidade d+1. Usa uma **fila** (FIFO).

```
Fronteira (fila): [S₀]
Passo 1: expande S₀ → adiciona filhos [A, B]
Passo 2: expande A  → adiciona filhos [C, D]
Passo 3: expande B  → adiciona filhos [E]
...
(visita em camadas: profundidade 0, depois 1, depois 2, ...)
```

**Propriedades:**

| Propriedade | Valor | Explicação |
|-------------|-------|------------|
| Completo? | **Sim** | Se existe solução com profundidade finita, BFS a encontra |
| Ótimo? | **Sim (se custo = 1 por passo)** | Encontra a solução mais rasa (menos passos) |
| Complexidade temporal | **O(b^d)** | b = fator de ramificação, d = profundidade da solução |
| Complexidade espacial | **O(b^d)** | Mantém toda a fronteira na memória |

**Limitação crítica**: Para b=10, d=10 → 10^10 = 10 bilhões de nós. BFS é impraticável para problemas profundos.

### 2.2 Busca em Profundidade (DFS — Depth-First Search)

**Ideia**: Sempre expande o nó mais profundo da fronteira. Usa uma **pilha** (LIFO).

```
Fronteira (pilha): [S₀]
Passo 1: expande S₀ → push [A, B] → fronteira: [A, B]
Passo 2: expande A  → push [C, D] → fronteira: [C, D, B]
Passo 3: expande C  → sem filhos → fronteira: [D, B]
...
(vai fundo em um ramo antes de explorar outros)
```

**Propriedades:**

| Propriedade | Valor | Explicação |
|-------------|-------|------------|
| Completo? | **Não** (em espaços infinitos) | Pode descer infinitamente por um ramo sem solução |
| Ótimo? | **Não** | Não garante o caminho mais curto |
| Complexidade temporal | **O(b^m)** | m = profundidade máxima da árvore |
| Complexidade espacial | **O(bm)** | Só mantém o caminho atual + irmãos |

**Vantagem**: Uso de memória **linear** — muito eficiente em espaço!

### 2.3 Busca de Custo Uniforme (UCS — Uniform Cost Search)

**Ideia**: Expande o nó com **menor custo total do caminho** g(n). Usa uma **fila de prioridade** ordenada por g(n).

Equivalente ao algoritmo de Dijkstra para encontrar o caminho de menor custo.

**Propriedades:**

| Propriedade | Valor | Explicação |
|-------------|-------|------------|
| Completo? | **Sim** (se custos > 0) | Garantido encontrar solução se existir |
| Ótimo? | **Sim** | Garante o caminho de menor custo total |
| Complexidade | **O(b^(C*/ε))** | C* = custo da solução ótima, ε = custo mínimo de ação |

### 2.4 Busca de Aprofundamento Iterativo (IDA — Iterative Deepening)

**Ideia**: Executa DFS repetidamente com limite de profundidade crescente (0, 1, 2, ...) até encontrar a solução.

**Propriedades:**

| Propriedade | Valor |
|-------------|-------|
| Completo? | **Sim** |
| Ótimo? | **Sim** (se custo = 1 por passo) |
| Complexidade temporal | **O(b^d)** |
| Complexidade espacial | **O(bd)** — melhor que BFS! |

**Por que é bom?** Combina o melhor dos dois mundos: completude/otimalidade de BFS com eficiência de memória de DFS.

---

## 3. Busca Informada (Heurística)

Busca informada usa **conhecimento adicional sobre o problema** para guiar a busca em direção ao objetivo.

### 3.1 Funções Heurísticas

> *"Uma heurística é uma estimativa do custo para chegar ao objetivo a partir de um nó."*

**Notação**: `h(n)` = estimativa do custo do nó n até o objetivo mais próximo.

**Requisito mínimo**: `h(objetivo) = 0`

**Exemplos de heurísticas:**

*Para labirinto (distância em grade):*
- **h₁ = Distância de Manhattan**: |x₁-x₂| + |y₁-y₂| (movimento em 4 direções)
- **h₂ = Distância Euclidiana**: √((x₁-x₂)² + (y₁-y₂)²) (pode mover em diagonal)

*Para quebra-cabeça de 8 peças:*
- **h₁ = Peças fora do lugar**: conta quantas peças não estão na posição objetivo.
- **h₂ = Distância de Manhattan**: soma das distâncias de Manhattan de cada peça à sua posição objetivo.
- **h₃ = Conflito linear** (mais sofisticada): conta conflitos em linhas/colunas.

### 3.2 Heurísticas Admissíveis

> *"Uma heurística h(n) é admissível se NUNCA superestima o custo real para atingir o objetivo."*

```
h(n) ≤ h*(n)    para todo nó n
```

onde `h*(n)` é o custo real do caminho ótimo de n ao objetivo.

**Por que importa?** Se h(n) é admissível, o algoritmo A* é **garantidamente ótimo**.

**Exemplo**: No quebra-cabeça de 8 peças:
- `h₁` (peças fora do lugar) é admissível: cada peça fora do lugar precisa de no mínimo 1 movimento.
- `h₂` (Manhattan) é admissível: cada peça precisa de no mínimo a distância de Manhattan de movimentos.
- `h₂` domina `h₁` (h₂(n) ≥ h₁(n) para todo n) → h₂ é mais informada.

**Contra-exemplo de inadmissibilidade**: `h(n) = 2 × (distância de Manhattan)` seria **não admissível** — superestimaria o custo.

### 3.3 Heurísticas Consistentes (Monótonas)

> *"Uma heurística é consistente se h(n) ≤ c(n, a, n') + h(n') para todo nó n e seu sucessor n' via ação a."*

```
Condição de consistência:
h(n) ≤ custo_do_arco(n → n') + h(n')
```

Consistência implica admissibilidade (mas não o contrário). Se h é consistente, os valores de f(n) ao longo de qualquer caminho são não-decrescentes.

**Importância prática**: Com heurística consistente, o A* nunca precisa reexpandir um nó — cada nó expandido já tem seu custo ótimo definido.

### 3.4 Busca Gulosa (Greedy Best-First Search)

**Ideia**: Sempre expande o nó com **menor valor heurístico** h(n) — o que parece mais próximo do objetivo.

**f(n) = h(n)** (ignora o custo já gasto)

**Propriedades:**

| Propriedade | Valor | Explicação |
|-------------|-------|------------|
| Completo? | **Não** (pode entrar em loops) | Pode ciclar se heurística for enganosa |
| Ótimo? | **Não** | Ignora o custo do caminho percorrido |
| Complexidade | **O(b^m)** no pior caso | Mas muito rápida na prática com boa heurística |

**Analogia**: Guiar-se sempre "na direção que parece mais próxima" — pode funcionar, mas pode também levar a becos sem saída.

### 3.5 Algoritmo A*

> *"A* combina o custo real do caminho até agora (g) com a estimativa heurística (h)."*

**f(n) = g(n) + h(n)**

onde:
- `g(n)` = custo total do caminho de S₀ até n (custo real acumulado)
- `h(n)` = estimativa do custo de n até o objetivo (heurística)
- `f(n)` = estimativa do custo total do caminho completo passando por n

**Propriedades (com h admissível):**

| Propriedade | Valor | Condição |
|-------------|-------|----------|
| Completo? | **Sim** | Se b é finito e custos > 0 |
| Ótimo? | **Sim** | Se h é admissível |
| Complexidade temporal | **O(b^d)** no pior caso | Exponencial, mas heurística reduz dramaticamente |
| Complexidade espacial | **O(b^d)** | Guarda todos os nós na memória |

**Intuição**: O A* equilibra exploração (não vai tão fundo quanto DFS) com foco no objetivo (não explora em todas as direções como BFS). É o "melhor dos dois mundos" quando a heurística é boa.

---

## 4. Tabela Comparativa de Algoritmos

| Algoritmo | Estrutura | Completo? | Ótimo? | Tempo | Espaço | Usa h? |
|-----------|-----------|-----------|--------|-------|--------|--------|
| BFS | Fila | Sim | Sim* | O(b^d) | O(b^d) | Não |
| DFS | Pilha | Não** | Não | O(b^m) | O(bm) | Não |
| UCS | Fila de prioridade (g) | Sim | Sim | O(b^(C*/ε)) | O(b^(C*/ε)) | Não |
| IDA | Pilha (limite) | Sim | Sim* | O(b^d) | O(bd) | Não |
| Gulosa | Fila de prioridade (h) | Não | Não | O(b^m) | O(b^m) | Sim |
| A* | Fila de prioridade (f) | Sim | Sim*** | O(b^d) | O(b^d) | Sim |

*Ótimo apenas se custo = 1 por passo  
**Completo em espaços finitos com checagem de estados visitados  
***Ótimo apenas se h é admissível

---

## 5. Implementações em Python

```python
"""
Implementações completas de BFS, DFS, Busca de Custo Uniforme e A*.
Aplicados ao problema do labirinto e ao quebra-cabeça de 8 peças.
"""

import heapq
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set, Callable
import time


# ─── Estrutura de Nó ─────────────────────────────────────────────────────────

@dataclass
class No:
    """
    Nó da árvore de busca.
    Contém estado, pai, ação que gerou este nó, e custo acumulado.
    """
    estado: any
    pai: Optional["No"] = field(default=None, repr=False)
    acao: Optional[str] = None
    custo: float = 0.0
    profundidade: int = 0

    def __lt__(self, outro: "No") -> bool:
        """Necessário para comparação em heapq."""
        return self.custo < outro.custo

    def caminho(self) -> List["No"]:
        """Reconstrói o caminho do nó raiz até este nó."""
        no, caminho = self, []
        while no is not None:
            caminho.append(no)
            no = no.pai
        return list(reversed(caminho))

    def acoes_caminho(self) -> List[str]:
        """Retorna a sequência de ações do caminho."""
        return [no.acao for no in self.caminho() if no.acao is not None]


# ─── Problema do Labirinto ────────────────────────────────────────────────────

class ProblemaLabirinto:
    """
    Problema de busca para navegar em um labirinto 2D.
    
    Grade:
    - 0 = passagem livre
    - 1 = parede
    - 'S' = início
    - 'G' = objetivo
    """
    
    ACOES = {
        "cima":    (-1,  0),
        "baixo":   ( 1,  0),
        "esquerda":(  0, -1),
        "direita": (  0,  1),
    }
    
    def __init__(self, grade: List[List]):
        self.grade = grade
        self.linhas = len(grade)
        self.colunas = len(grade[0])
        self.inicio = self._encontrar("S")
        self.objetivo = self._encontrar("G")
    
    def _encontrar(self, simbolo: str) -> Tuple[int, int]:
        for r in range(self.linhas):
            for c in range(self.colunas):
                if self.grade[r][c] == simbolo:
                    return (r, c)
        raise ValueError(f"Símbolo '{simbolo}' não encontrado na grade.")
    
    def estado_inicial(self) -> Tuple[int, int]:
        return self.inicio
    
    def eh_objetivo(self, estado: Tuple[int, int]) -> bool:
        return estado == self.objetivo
    
    def acoes(self, estado: Tuple[int, int]) -> List[str]:
        r, c = estado
        acoes_validas = []
        for nome_acao, (dr, dc) in self.ACOES.items():
            nr, nc = r + dr, c + dc
            if (0 <= nr < self.linhas and
                0 <= nc < self.colunas and
                self.grade[nr][nc] != 1):  # não é parede
                acoes_validas.append(nome_acao)
        return acoes_validas
    
    def resultado(self, estado: Tuple[int, int], acao: str) -> Tuple[int, int]:
        r, c = estado
        dr, dc = self.ACOES[acao]
        return (r + dr, c + dc)
    
    def custo_acao(self, estado, acao, estado_resultado) -> float:
        return 1.0  # custo uniforme: 1 por passo
    
    def heuristica_manhattan(self, estado: Tuple[int, int]) -> float:
        """Distância de Manhattan até o objetivo — heurística admissível."""
        r, c = estado
        gr, gc = self.objetivo
        return abs(r - gr) + abs(c - gc)
    
    def exibir_solucao(self, caminho: List[Tuple[int, int]]):
        """Exibe o labirinto com o caminho da solução marcado."""
        grade_copia = [list(str(c) for c in linha) for linha in self.grade]
        for pos in caminho:
            r, c = pos
            if grade_copia[r][c] not in ("S", "G"):
                grade_copia[r][c] = "·"
        
        print("\nLabirinto com solução:")
        print("  " + " ".join(str(c) for c in range(self.colunas)))
        for r, linha in enumerate(grade_copia):
            cel_str = " ".join(
                "█" if c == "1" else
                "S" if c == "S" else
                "G" if c == "G" else
                "·" if c == "·" else " "
                for c in linha
            )
            print(f"{r} {cel_str}")


# ─── Algoritmos de Busca ──────────────────────────────────────────────────────

def busca_generica(problema, fronteira_tipo: str,
                   heuristica: Optional[Callable] = None) -> Tuple[Optional[No], dict]:
    """
    Implementação genérica dos algoritmos de busca.
    Muda apenas a estrutura de dados da fronteira.
    
    Args:
        problema: Problema de busca (com estado_inicial, eh_objetivo, acoes, resultado)
        fronteira_tipo: "bfs", "dfs", "ucs", "gulosa", "astar"
        heuristica: Função h(estado) para buscas informadas
    
    Returns:
        (nó_solução, estatísticas) ou (None, estatísticas) se sem solução
    """
    estado_inicial = problema.estado_inicial()
    no_raiz = No(estado=estado_inicial, custo=0.0, profundidade=0)
    
    # Inicializar fronteira segundo o tipo de busca
    if fronteira_tipo == "bfs":
        fronteira = deque([no_raiz])
        def pop_fronteira(): return fronteira.popleft()
        def push_fronteira(no): fronteira.append(no)
    elif fronteira_tipo == "dfs":
        fronteira = [no_raiz]
        def pop_fronteira(): return fronteira.pop()
        def push_fronteira(no): fronteira.append(no)
    else:
        # Fila de prioridade (UCS, Gulosa, A*)
        fronteira = [(0.0, 0, no_raiz)]  # (prioridade, contador, nó)
        contador = [1]
        def pop_fronteira():
            _, _, no = heapq.heappop(fronteira)
            return no
        def push_fronteira(no):
            if fronteira_tipo == "ucs":
                prioridade = no.custo
            elif fronteira_tipo == "gulosa":
                prioridade = heuristica(no.estado) if heuristica else 0
            else:  # astar
                h = heuristica(no.estado) if heuristica else 0
                prioridade = no.custo + h
            heapq.heappush(fronteira, (prioridade, contador[0], no))
            contador[0] += 1
    
    explorados: Set = set()
    nos_gerados = 1
    nos_expandidos = 0
    inicio = time.time()
    
    while fronteira:
        no_atual = pop_fronteira()
        
        # Teste de objetivo
        if problema.eh_objetivo(no_atual.estado):
            elapsed = time.time() - inicio
            return no_atual, {
                "nos_gerados": nos_gerados,
                "nos_expandidos": nos_expandidos,
                "profundidade": no_atual.profundidade,
                "custo": no_atual.custo,
                "tempo_ms": elapsed * 1000,
            }
        
        # Evitar re-expansão
        estado_hash = (no_atual.estado if isinstance(no_atual.estado, tuple)
                       else str(no_atual.estado))
        if estado_hash in explorados:
            continue
        explorados.add(estado_hash)
        nos_expandidos += 1
        
        # Expandir: gerar sucessores
        for acao in problema.acoes(no_atual.estado):
            estado_filho = problema.resultado(no_atual.estado, acao)
            custo_filho = no_atual.custo + problema.custo_acao(
                no_atual.estado, acao, estado_filho
            )
            filho = No(
                estado=estado_filho,
                pai=no_atual,
                acao=acao,
                custo=custo_filho,
                profundidade=no_atual.profundidade + 1,
            )
            nos_gerados += 1
            push_fronteira(filho)
    
    # Fronteira vazia sem encontrar objetivo
    return None, {
        "nos_gerados": nos_gerados,
        "nos_expandidos": nos_expandidos,
        "profundidade": -1,
        "custo": float("inf"),
        "tempo_ms": (time.time() - inicio) * 1000,
    }


def bfs(problema) -> Tuple[Optional[No], dict]:
    """Busca em Largura."""
    return busca_generica(problema, "bfs")

def dfs(problema) -> Tuple[Optional[No], dict]:
    """Busca em Profundidade."""
    return busca_generica(problema, "dfs")

def ucs(problema) -> Tuple[Optional[No], dict]:
    """Busca de Custo Uniforme."""
    return busca_generica(problema, "ucs")

def busca_gulosa(problema, heuristica) -> Tuple[Optional[No], dict]:
    """Busca Gulosa (Greedy Best-First)."""
    return busca_generica(problema, "gulosa", heuristica)

def astar(problema, heuristica) -> Tuple[Optional[No], dict]:
    """Algoritmo A*."""
    return busca_generica(problema, "astar", heuristica)


# ─── Problema do Quebra-Cabeça de 8 Peças ────────────────────────────────────

class Oito_Pecas:
    """
    O clássico quebra-cabeça de 8 peças (3×3).
    Estado: tupla de 9 elementos (0 = espaço vazio)
    """
    
    OBJETIVO = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    
    def __init__(self, estado_inicial: Tuple):
        self.inicio = estado_inicial
    
    def estado_inicial(self) -> Tuple:
        return self.inicio
    
    def eh_objetivo(self, estado: Tuple) -> bool:
        return estado == self.OBJETIVO
    
    def acoes(self, estado: Tuple) -> List[str]:
        pos_vazio = estado.index(0)
        r, c = divmod(pos_vazio, 3)
        acoes_validas = []
        if r > 0: acoes_validas.append("cima")
        if r < 2: acoes_validas.append("baixo")
        if c > 0: acoes_validas.append("esquerda")
        if c < 2: acoes_validas.append("direita")
        return acoes_validas
    
    def resultado(self, estado: Tuple, acao: str) -> Tuple:
        lista = list(estado)
        pos_vazio = lista.index(0)
        r, c = divmod(pos_vazio, 3)
        
        movimentos = {"cima": (-1, 0), "baixo": (1, 0),
                      "esquerda": (0, -1), "direita": (0, 1)}
        dr, dc = movimentos[acao]
        nr, nc = r + dr, c + dc
        nova_pos = nr * 3 + nc
        
        lista[pos_vazio], lista[nova_pos] = lista[nova_pos], lista[pos_vazio]
        return tuple(lista)
    
    def custo_acao(self, estado, acao, resultado) -> float:
        return 1.0
    
    def h_pecas_fora(self, estado: Tuple) -> int:
        """Heurística h1: número de peças fora do lugar."""
        return sum(1 for i, v in enumerate(estado) if v != 0 and v != self.OBJETIVO[i])
    
    def h_manhattan(self, estado: Tuple) -> int:
        """
        Heurística h2: soma das distâncias de Manhattan.
        É admissível e mais informada que h1.
        """
        total = 0
        for i, valor in enumerate(estado):
            if valor != 0:
                pos_atual_r, pos_atual_c = divmod(i, 3)
                pos_obj_r, pos_obj_c = divmod(self.OBJETIVO.index(valor), 3)
                total += abs(pos_atual_r - pos_obj_r) + abs(pos_atual_c - pos_obj_c)
        return total
    
    def exibir_estado(self, estado: Tuple, titulo: str = ""):
        """Exibe o estado do quebra-cabeça."""
        if titulo:
            print(titulo)
        for i in range(0, 9, 3):
            linha = " ".join(str(v) if v != 0 else "_" for v in estado[i:i+3])
            print(f"  {linha}")


# ─── Comparação de Algoritmos ─────────────────────────────────────────────────

def comparar_algoritmos_labirinto():
    """
    Compara todos os algoritmos no problema do labirinto.
    """
    # Labirinto 7×7 com algumas paredes
    grade = [
        ["S", 0, 1, 0, 0, 0, 0],
        [0,   0, 1, 0, 1, 0, 0],
        [0,   0, 0, 0, 1, 0, 0],
        [1,   1, 0, 1, 0, 0, 1],
        [0,   0, 0, 0, 0, 1, 0],
        [0,   1, 1, 1, 0, 0, 0],
        [0,   0, 0, 0, 0, 0, "G"],
    ]
    
    print("=" * 65)
    print("COMPARAÇÃO DE ALGORITMOS — Labirinto 7×7")
    print("=" * 65)
    
    problema = ProblemaLabirinto(grade)
    h = problema.heuristica_manhattan
    
    algoritmos = [
        ("BFS",           lambda p: bfs(p),               False),
        ("DFS",           lambda p: dfs(p),               False),
        ("UCS",           lambda p: ucs(p),               False),
        ("A* (Manhattan)", lambda p: astar(p, h),          True),
        ("Gulosa",        lambda p: busca_gulosa(p, h),   True),
    ]
    
    print(f"\n{'Algoritmo':<20} {'Nós Exp.':>10} {'Nós Ger.':>10} "
          f"{'Prof.':>6} {'Custo':>7} {'Tempo(ms)':>10}")
    print("-" * 65)
    
    for nome, func, usa_h in algoritmos:
        solucao, stats = func(problema)
        status = "✓" if solucao else "✗"
        print(f"{nome:<20} {stats['nos_expandidos']:>10} {stats['nos_gerados']:>10} "
              f"{stats['profundidade']:>6} {stats['custo']:>7.1f} "
              f"{stats['tempo_ms']:>10.3f}  {status}")
    
    # Exibir solução A*
    solucao, _ = astar(problema, h)
    if solucao:
        caminho = [no.estado for no in solucao.caminho()]
        problema.exibir_solucao(caminho)
        print(f"\nSolução A*: {' → '.join(solucao.acoes_caminho())}")


def comparar_heuristicas_8_pecas():
    """
    Compara as duas heurísticas no quebra-cabeça de 8 peças.
    """
    # Estado com solução de aproximadamente 20 movimentos
    estado_inicial = (7, 2, 4, 5, 0, 6, 8, 3, 1)
    
    print("\n" + "=" * 65)
    print("COMPARAÇÃO DE HEURÍSTICAS — Quebra-Cabeça de 8 Peças")
    print("=" * 65)
    
    problema = Oito_Pecas(estado_inicial)
    problema.exibir_estado(estado_inicial, "\nEstado inicial:")
    problema.exibir_estado(Oito_Pecas.OBJETIVO, "\nEstado objetivo:")
    
    configs = [
        ("A* + h1 (peças fora)", problema.h_pecas_fora),
        ("A* + h2 (Manhattan)",  problema.h_manhattan),
        ("UCS (sem heurística)", None),
    ]
    
    print(f"\n{'Configuração':<30} {'Nós Exp.':>10} {'Custo':>6} {'Tempo(ms)':>10}")
    print("-" * 58)
    
    for nome, h in configs:
        if h is None:
            solucao, stats = ucs(problema)
        else:
            solucao, stats = astar(problema, h)
        
        print(f"{nome:<30} {stats['nos_expandidos']:>10} "
              f"{stats['custo']:>6.0f} {stats['tempo_ms']:>10.3f}")
    
    # Mostrar valores das heurísticas para o estado inicial
    print(f"\nValores de heurística para o estado inicial:")
    print(f"  h1 (peças fora do lugar) = {problema.h_pecas_fora(estado_inicial)}")
    print(f"  h2 (Manhattan total)     = {problema.h_manhattan(estado_inicial)}")
    print(f"\n  h2 > h1 → h2 é MAIS INFORMADA → A* com h2 expande MENOS nós")


def verificar_admissibilidade():
    """
    Demonstra verificação de admissibilidade de heurísticas.
    """
    print("\n" + "=" * 60)
    print("VERIFICAÇÃO DE ADMISSIBILIDADE DE HEURÍSTICAS")
    print("=" * 60)
    
    problema = Oito_Pecas((1, 2, 3, 4, 5, 6, 0, 7, 8))
    
    # Estado simples onde sabemos o custo real
    # (1,2,3,4,5,6,0,7,8) → objetivo em 2 movimentos
    # Movimentos: direita → baixo → objetivo
    estado_teste = (1, 2, 3, 4, 5, 6, 0, 7, 8)
    custo_real = 2  # calculado manualmente
    
    h1 = problema.h_pecas_fora(estado_teste)
    h2 = problema.h_manhattan(estado_teste)
    
    print(f"\nEstado de teste:")
    problema.exibir_estado(estado_teste)
    print(f"\nCusto real até objetivo: {custo_real}")
    print(f"h1 (peças fora do lugar): {h1} {'≤' if h1 <= custo_real else '>'} {custo_real} "
          f"→ {'ADMISSÍVEL ✓' if h1 <= custo_real else 'NÃO ADMISSÍVEL ✗'}")
    print(f"h2 (Manhattan):           {h2} {'≤' if h2 <= custo_real else '>'} {custo_real} "
          f"→ {'ADMISSÍVEL ✓' if h2 <= custo_real else 'NÃO ADMISSÍVEL ✗'}")
    
    h_ruim = h2 * 3  # heurística deliberadamente inadmissível
    print(f"h_ruim (3 × Manhattan):   {h_ruim} {'≤' if h_ruim <= custo_real else '>'} {custo_real} "
          f"→ {'ADMISSÍVEL ✓' if h_ruim <= custo_real else 'NÃO ADMISSÍVEL ✗'}")
    print("\n⚠️  h_ruim superestima o custo — A* com h_ruim pode não encontrar solução ótima!")


# ─── Ponto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    comparar_algoritmos_labirinto()
    comparar_heuristicas_8_pecas()
    verificar_admissibilidade()
```

---

## 6. O Problema das N-Rainhas (Busca Local)

O problema das N-Rainhas é um exemplo clássico de **problema de satisfação de restrições** que também pode ser resolvido com busca:

**Objetivo**: Posicionar N rainhas em um tabuleiro N×N de forma que nenhuma ataque outra.

```python
def resolver_n_rainhas_backtracking(n: int) -> Optional[List[int]]:
    """
    Resolve N-Rainhas com backtracking (DFS com poda).
    
    Retorna lista onde posicoes[i] = coluna da rainha na linha i.
    """
    posicoes = []
    
    def eh_seguro(linha: int, coluna: int) -> bool:
        for r, c in enumerate(posicoes):
            if c == coluna:           # mesma coluna
                return False
            if abs(r - linha) == abs(c - coluna):  # mesma diagonal
                return False
        return True
    
    def backtrack(linha: int) -> bool:
        if linha == n:
            return True  # todas as rainhas foram colocadas
        for col in range(n):
            if eh_seguro(linha, col):
                posicoes.append(col)
                if backtrack(linha + 1):
                    return True
                posicoes.pop()  # backtrack
        return False
    
    if backtrack(0):
        return posicoes
    return None

# Teste rápido
for n in [4, 8, 12]:
    inicio = time.time()
    solucao = resolver_n_rainhas_backtracking(n)
    elapsed = (time.time() - inicio) * 1000
    if solucao:
        print(f"N={n:2d}: solução em {elapsed:6.2f}ms → colunas: {solucao}")
```

---

## 7. Questões para Reflexão

1. **Formulação**: O problema do Sudoku pode ser formulado como busca em espaço de estados. Descreva formalmente os 5 componentes (S₀, AÇÕES, RESULTADO, OBJETIVO?, CUSTO). Qual algoritmo você usaria? Por quê?

2. **Completude vs. Otimalidade**: Em que situações você sacrificaria otimalidade por velocidade? Cite um exemplo real de aplicação de IA onde a solução "boa o suficiente" é preferível à solução ótima.

3. **Heurísticas**: Para o problema do caixeiro viajante (encontrar a menor rota que visita N cidades e volta ao início), proponha uma heurística admissível. Prove (ou argumente) que ela é admissível.

4. **A* na prática**: Em jogos de videogame, A* é amplamente usado para pathfinding de NPCs. Quais desafios surgem quando o mapa tem 1000×1000 tiles e há 100 NPCs precisando calcular rotas simultaneamente? Que otimizações você usaria?

5. **Busca local**: Algoritmos como hill climbing e simulated annealing são usados para problemas onde o espaço de estados é muito grande para busca exaustiva. Pesquise esses algoritmos e explique como eles diferem dos algoritmos estudados nesta aula.

6. **Complexidade**: BFS tem complexidade O(b^d). Para um labirinto de 1000×1000 com fator de ramificação b=4 e solução à profundidade d=500, estime o número de nós que BFS precisaria explorar. Por que isso torna BFS impraticável? O A* resolveria isso?

---

## Referências

**[1]** RUSSELL, S.; NORVIG, P. **Inteligência Artificial: uma abordagem moderna**. 4. ed. Rio de Janeiro: GEN LTC, 2022. Cap. 3 (Resolução de Problemas por Busca) e Cap. 4 (Busca em Ambientes Complexos).

**[2]** FACELI, K. et al. **Inteligência Artificial: uma abordagem de aprendizado de máquina**. 2. ed. LTC, 2021.

**[3]** CORMEN, T. H. et al. **Algoritmos: Teoria e Prática**. 3. ed. Elsevier, 2012. Cap. 22 (Busca em Grafos), Cap. 24 (Dijkstra).

**[4]** HART, P. E.; NILSSON, N. J.; RAPHAEL, B. A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics*, v. 4, n. 2, p. 100–107, 1968.
> *O artigo original do A*.*

**[5]** PEARL, J. **Heuristics: Intelligent Search Strategies for Computer Problem Solving**. Addison-Wesley, 1984.

**[6]** KORF, R. E. Depth-First Iterative Deepening: An Optimal Admissible Tree Search. *Artificial Intelligence*, v. 27, p. 97–109, 1985.
> *O artigo que propôs o IDA*.*

---

*Aula anterior: [Aula 02 — Agentes Inteligentes](./aula-02-agentes-inteligentes.md)*  
*Próxima aula: [Aula 04 — Representação do Conhecimento](./aula-04-representacao-conhecimento.md)*
