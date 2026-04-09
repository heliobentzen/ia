# Aula 04 — Representação do Conhecimento

> **Módulo:** 01 — Fundamentos de Inteligência Artificial  
> **Duração:** 45 minutos  
> **Pré-requisitos:** Aulas 01–03

---

## Objetivos de Aprendizagem

Ao final desta aula, o estudante será capaz de:

1. **Distinguir** os principais formalismos de representação do conhecimento.
2. **Aplicar** lógica proposicional e de primeira ordem para modelar domínios.
3. **Explicar** a diferença entre encadeamento para frente e encadeamento para trás.
4. **Descrever** redes bayesianas e sua aplicação ao raciocínio incerto.
5. **Implementar** um sistema especialista simples baseado em regras em Python.
6. **Reconhecer** quando cada formalismo é mais adequado para um problema.

---

## 1. Por que Representação do Conhecimento Importa?

Um sistema inteligente precisa de **conhecimento** para raciocinar. Mas como armazenar e manipular conhecimento em um computador?

**Requisitos de um bom formalismo de representação:**
1. **Expressividade**: Capaz de representar tudo que precisamos.
2. **Eficiência inferencial**: Raciocinar rapidamente a partir do conhecimento.
3. **Aquisição eficiente**: Fácil de adicionar novo conhecimento.
4. **Clareza semântica**: Significado não ambíguo.

Diferentes formalismos fazem trade-offs diferentes entre esses requisitos.

---

## 2. Lógica Proposicional

### 2.1 Sintaxe

A lógica proposicional é a mais simples das lógicas formais:

- **Proposições atômicas**: `P`, `Q`, `Chuva`, `TemFebra` (verdadeiro ou falso)
- **Conectivos lógicos**:
  - `¬P` — negação ("não P")
  - `P ∧ Q` — conjunção ("P e Q")
  - `P ∨ Q` — disjunção ("P ou Q")
  - `P → Q` — implicação ("se P então Q")
  - `P ↔ Q` — bicondicional ("P se e somente se Q")

### 2.2 Semântica e Tabelas-Verdade

| P | Q | ¬P | P ∧ Q | P ∨ Q | P → Q | P ↔ Q |
|---|---|----|--------|--------|--------|--------|
| V | V | F  | V      | V      | V      | V      |
| V | F | F  | F      | V      | F      | F      |
| F | V | V  | F      | V      | V      | F      |
| F | F | V  | F      | F      | V      | V      |

**Observação**: `P → Q` é falso **apenas** quando P é verdadeiro e Q é falso.

### 2.3 Inferência em Lógica Proposicional

**Modus Ponens** (a regra de inferência mais básica):
```
P → Q
P
------
Q
```

**Exemplo**:
```
Premissa 1: Se chove, então a rua fica molhada.
Premissa 2: Está chovendo.
Conclusão:  A rua está molhada.

Formalização:
P = Chove
Q = RuaMolhada
Dado: Chove → RuaMolhada, Chove ⊢ RuaMolhada
```

**Resolução** (para provas por refutação):
```
Cláusula 1: ¬P ∨ Q    (equivale a: P → Q)
Cláusula 2: P
Resolução: Q
```

### 2.4 Limitações da Lógica Proposicional

- Não tem variáveis: precisa de proposição para **cada** objeto.
- Para representar "Todo humano é mortal", precisaria de uma proposição por humano:
  `SocratesMortal`, `PlataoMortal`, `AristotelesMortal`, ...
- Não escala para domínios com muitos objetos.

---

## 3. Lógica de Primeira Ordem (FOL)

### 3.1 Sintaxe Expandida

A Lógica de Primeira Ordem (FOL — First Order Logic) adiciona:

- **Termos**: Constantes (`Sócrates`, `Atenas`), Variáveis (`x`, `y`), Funções (`pai(Sócrates)`)
- **Predicados**: `Humano(x)`, `Mortal(x)`, `AmaEm(x, y)`
- **Quantificadores**:
  - `∀x` — "para todo x"
  - `∃x` — "existe um x tal que"

### 3.2 Exemplos

```prolog
% Em Prolog (linguagem baseada em FOL)

% Fatos (proposições atômicas com objetos específicos)
humano(socrates).
humano(platao).
humano(aristoteles).
pai(cronos, zeus).

% Regras (com variáveis e quantificadores implícitos)
mortal(X) :- humano(X).           % Todo humano é mortal
ancestral(X, Y) :- pai(X, Y).     % Pai implica ancestral direto
ancestral(X, Z) :- pai(X, Y), ancestral(Y, Z).  % Ancestral transitivo

% Consultas (queries)
?- mortal(socrates).              % Resultado: true
?- mortal(zeus).                  % Resultado: false (zeus não é humano)
?- mortal(X).                     % Resultado: X = socrates; X = platao; X = aristoteles
```

### 3.3 Exemplos em FOL — Domínio Médico

```
% Sintomas e diagnósticos
∀p: Paciente(p) ∧ TemFebra(p) ∧ TemTosse(p) → PossivelGripe(p)
∀p: Paciente(p) ∧ TemFebra(p) ∧ DorNaGarganta(p) → PossivelAngina(p)
∀p: Paciente(p) ∧ PossivelGripe(p) ∧ DorMuscular(p) → GripeConfirmada(p)

% Tratamentos
∀p: GripeConfirmada(p) → Tratamento(p, repouso_e_hidratação)
∀p: PossivelAngina(p) → EncaminharPara(p, otorrino)
```

### 3.4 Limitações da FOL

- A FOL pura é **incompleta** para certos domínios (Teorema da Incompletude de Gödel).
- Inferência em FOL é semi-decidível (pode não terminar).
- Não lida nativamente com incerteza.
- Para conhecimento incerto, precisamos de extensões probabilísticas.

---

## 4. Redes Semânticas e Frames

### 4.1 Redes Semânticas

Uma rede semântica é um **grafo dirigido** onde:
- **Nós** representam conceitos ou instâncias.
- **Arestas** representam relações semânticas entre eles.

```
Relações comuns:
- É-UM (is-a): Cachorro É-UM Animal
- INSTÂNCIA-DE: Rex INSTÂNCIA-DE Cachorro
- PARTE-DE: Roda PARTE-DE Carro
- ATRIBUTO: Cachorro ATRIBUTO Pelagem

Exemplo:
        É-UM                É-UM
Animal ←────── Mamífero ←──────── Cachorro ←─── Rex (instância)
  │                                   │
  │ respira                           │ latidos
  ▼                                   ▼
  ar                               som grave
```

**Herança**: Rex herda todas as propriedades de Cachorro, Mamífero e Animal.

### 4.2 Frames (Minsky, 1974)

Frames organizam conhecimento em **estruturas de dados com slots**:

```python
# Exemplo conceitual de Frame
Frame_Animal = {
    "nome": "Animal",
    "superclasse": None,
    "slots": {
        "respiracao": {"valor": "ar", "herdavel": True},
        "locomocao": {"valor": None, "herdavel": True},  # default: None
        "alimentacao": {"valor": None, "herdavel": True},
    }
}

Frame_Cachorro = {
    "nome": "Cachorro",
    "superclasse": "Mamífero",
    "slots": {
        "locomocao": {"valor": "4 patas", "herdavel": True},
        "som": {"valor": "latido", "herdavel": True},
        "pelagem": {"valor": "pelos", "herdavel": True},
    }
}

Frame_Rex = {
    "nome": "Rex",
    "superclasse": "Cachorro",
    "slots": {
        "nome_proprio": {"valor": "Rex"},
        "dono": {"valor": "João"},
        "cor": {"valor": "marrom"},
    }
}
```

Frames são os precursores das classes em OOP!

---

## 5. Ontologias

### 5.1 O que é uma Ontologia?

> *"Uma ontologia é uma especificação explícita e formal de uma conceitualização."*  
> — Gruber, 1993

Em IA, ontologias fornecem:
- Vocabulário compartilhado para um domínio.
- Hierarquias de conceitos.
- Relações entre conceitos.
- Axiomas e restrições.

### 5.2 OWL e RDF (Web Semântica)

**RDF (Resource Description Framework)**: Representa conhecimento em triplas `(sujeito, predicado, objeto)`:
```turtle
:Rex rdf:type :Cachorro .
:Rex :temDono :João .
:Cachorro rdfs:subClassOf :Mamífero .
:Mamífero rdfs:subClassOf :Animal .
```

**OWL (Web Ontology Language)**: Linguagem mais expressiva baseada em Lógica de Descrição:
```turtle
:Cachorro owl:subClassOf :Mamífero ;
    owl:subClassOf [
        owl:onProperty :temPatas ;
        owl:hasValue "4"^^xsd:integer
    ] .
```

**Ontologias conhecidas:**
- **WordNet**: Léxico semântico do inglês (e português via OpenWordNet-PT).
- **DBpedia**: Versão estruturada da Wikipedia.
- **SNOMED CT**: Ontologia médica com 350.000+ conceitos.
- **Gene Ontology**: Anotação de funções gênicas.

---

## 6. Sistemas Baseados em Regras

### 6.1 Estrutura de um Sistema Especialista

```
┌──────────────────────────────────────────────────────────┐
│                  SISTEMA ESPECIALISTA                    │
│                                                          │
│  ┌──────────────┐    ┌───────────────┐                  │
│  │   BASE DE    │    │   BASE DE     │                  │
│  │  FATOS       │←──→│   REGRAS      │                  │
│  │  (memória    │    │   (SE-ENTÃO)  │                  │
│  │  de trabalho)│    │               │                  │
│  └──────────────┘    └───────────────┘                  │
│          ↕                   ↕                          │
│    ┌─────────────────────────────────┐                  │
│    │        MOTOR DE INFERÊNCIA      │                  │
│    │   (encadeamento frente/trás)    │                  │
│    └─────────────────────────────────┘                  │
│                      ↕                                  │
│    ┌─────────────────────────────────┐                  │
│    │    INTERFACE COM O USUÁRIO      │                  │
│    └─────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Encadeamento para Frente (Forward Chaining)

**Direção**: Dos fatos → para conclusões.

**Algoritmo**: Aplica regras cujas condições são satisfeitas pelos fatos conhecidos, acrescentando novas conclusões à base de fatos. Repete até não haver mais regras aplicáveis.

**Quando usar**: Quando temos fatos e queremos descobrir todas as conclusões possíveis.

```
Fatos iniciais: {A, B}
Regras:
  R1: A ∧ B → C
  R2: C → D
  R3: A ∧ D → E

Passo 1: R1 dispara (A e B são fatos) → acrescenta C
         Fatos: {A, B, C}
Passo 2: R2 dispara (C é fato) → acrescenta D
         Fatos: {A, B, C, D}
Passo 3: R3 dispara (A e D são fatos) → acrescenta E
         Fatos: {A, B, C, D, E}
Passo 4: Nenhuma regra nova dispara → fim
```

**Aplicações**: Monitoramento em tempo real, detecção de fraude, sistemas de alerta.

### 6.3 Encadeamento para Trás (Backward Chaining)

**Direção**: Do objetivo → para trás, buscando provar o objetivo.

**Algoritmo**: Começa com o objetivo a provar. Para provar o objetivo, encontra regras cuja conclusão é o objetivo, e tenta provar as premissas recursivamente.

**Quando usar**: Quando temos uma hipótese específica e queremos confirmá-la.

```
Objetivo: Provar E

Busca regra com conclusão E:
  R3: A ∧ D → E
  → Sub-objetivo 1: provar A (está nos fatos? SIM)
  → Sub-objetivo 2: provar D
     Busca regra com conclusão D:
       R2: C → D
       → Sub-objetivo: provar C
          Busca regra com conclusão C:
            R1: A ∧ B → C
            → Provar A: SIM (fato)
            → Provar B: SIM (fato)
          → C provado!
       → D provado!
  → E provado!
```

**Aplicações**: Diagnóstico médico, sistemas de consulta, Prolog.

---

## 7. Redes Bayesianas

### 7.1 Raciocínio sob Incerteza

O mundo real é **incerto**. A lógica clássica é binária (verdadeiro/falso), mas a realidade é gradual:
- "O paciente tem febre" pode ser 0.7 (70% de probabilidade).
- "Dado que tem febre E tosse, a probabilidade de gripe é 0.85."

**Teorema de Bayes**:
```
P(hipótese | evidência) = P(evidência | hipótese) × P(hipótese) / P(evidência)

Ou: P(H|E) = P(E|H) × P(H) / P(E)

Onde:
- P(H|E): probabilidade posterior (hipótese dado evidência)
- P(E|H): verossimilhança (probabilidade de ver a evidência se a hipótese for verdade)
- P(H): probabilidade a priori (antes de ver a evidência)
- P(E): probabilidade marginal da evidência (normalizador)
```

### 7.2 Estrutura de uma Rede Bayesiana

Uma rede bayesiana é um **DAG** (grafo acíclico dirigido) onde:
- **Nós**: variáveis aleatórias.
- **Arestas**: dependências causais (A → B significa "A causa B").
- **CPT (Tabela de Probabilidade Condicional)**: para cada nó, `P(nó | pais)`.

```
Exemplo: Diagnóstico de Gripe/Alergia

                  Estação do Ano
                  (primavera/outono/inverno)
                       │
              ┌────────┴────────┐
              ▼                 ▼
           Gripe             Alergia
          (V/F)               (V/F)
              │         │       │
              │         ▼       │
              └──►   Coriza  ◄──┘
                     (V/F)
                       │
                       ▼
                     Febre
                     (V/F)
```

**Tabela de Probabilidade Condicional (CPT) para Febre:**

| Gripe | Febre = Verdadeiro | Febre = Falso |
|-------|-------------------|---------------|
| V     | 0.90              | 0.10          |
| F     | 0.05              | 0.95          |

### 7.3 Independência Condicional

A **independência condicional** é a chave que torna redes bayesianas computacionalmente tratáveis.

`P(X | Y, Z) = P(X | Z)` significa: dado Z, X é independente de Y.

**Por que importa?** Em vez de armazenar `P(X₁, X₂, ..., Xₙ)` com 2ⁿ entradas, a rede bayesiana armazena apenas as CPTs locais, que têm tamanho muito menor.

---

## 8. Implementação: Sistema Especialista Médico

```python
"""
Sistema Especialista Médico Simplificado
Demonstra:
1. Base de regras SE-ENTÃO
2. Encadeamento para frente (Forward Chaining)
3. Encadeamento para trás (Backward Chaining)
4. Rede Bayesiana simplificada para raciocínio incerto
"""

from typing import Set, Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import json


# ─── Motor de Inferência com Encadeamento para Frente ────────────────────────

@dataclass
class Regra:
    """
    Representa uma regra SE-ENTÃO.
    
    Exemplo:
        nome = "Gripe_R1"
        condicoes = {"febre", "tosse", "dor_muscular"}
        conclusao = "gripe_possivel"
        confianca = 0.85
    """
    nome: str
    condicoes: Set[str]
    conclusao: str
    confianca: float = 1.0
    explicacao: str = ""

    def __repr__(self):
        conds = " E ".join(sorted(self.condicoes))
        return f"SE {conds} → {self.conclusao} (CF={self.confianca:.2f})"


class MotorEncadeamentoFrente:
    """
    Motor de inferência com encadeamento para frente (Forward Chaining).
    Aplica regras cujas condições são satisfeitas pelos fatos conhecidos.
    """
    
    def __init__(self, regras: List[Regra]):
        self.regras = regras
        self.log_inferencias: List[str] = []
    
    def inferir(self, fatos_iniciais: Set[str],
                verbose: bool = True) -> Dict[str, float]:
        """
        Executa o encadeamento para frente.
        
        Args:
            fatos_iniciais: Conjunto de fatos conhecidos (sintomas, etc.)
            verbose: Se True, exibe o processo de inferência
        
        Returns:
            Dicionário {conclusão: confiança} para todos os fatos derivados
        """
        # Fatos de trabalho: {fato: confiança}
        fatos = {f: 1.0 for f in fatos_iniciais}
        self.log_inferencias = []
        
        mudou = True
        iteracao = 0
        
        while mudou:
            mudou = False
            iteracao += 1
            
            for regra in self.regras:
                # Verifica se todas as condições são satisfeitas
                if (regra.conclusao not in fatos and
                    all(c in fatos for c in regra.condicoes)):
                    
                    # Calcula confiança combinada
                    cf_condicoes = min(fatos[c] for c in regra.condicoes)
                    cf_conclusao = cf_condicoes * regra.confianca
                    
                    fatos[regra.conclusao] = cf_conclusao
                    mudou = True
                    
                    log = (f"[Iter {iteracao}] {regra.nome}: "
                           f"{' ∧ '.join(sorted(regra.condicoes))} → "
                           f"{regra.conclusao} (CF={cf_conclusao:.2f})")
                    self.log_inferencias.append(log)
                    
                    if verbose:
                        print(f"  ✓ {log}")
        
        # Retorna apenas os fatos derivados (não os iniciais)
        return {k: v for k, v in fatos.items()
                if k not in fatos_iniciais}


class MotorEncadeamentoTras:
    """
    Motor de inferência com encadeamento para trás (Backward Chaining).
    Tenta provar um objetivo específico a partir dos fatos.
    """
    
    def __init__(self, regras: List[Regra], fatos: Set[str]):
        self.regras = regras
        self.fatos = fatos
        self.rastro_prova: List[str] = []
        self._visitados: Set[str] = set()  # Evita loops
    
    def provar(self, objetivo: str, profundidade: int = 0) -> Tuple[bool, float]:
        """
        Tenta provar um objetivo via encadeamento para trás.
        
        Returns:
            (sucesso, confiança)
        """
        indent = "  " * profundidade
        
        # Caso base: objetivo já é um fato conhecido
        if objetivo in self.fatos:
            self.rastro_prova.append(f"{indent}✓ '{objetivo}' é fato direto")
            return True, 1.0
        
        # Evitar loops
        if objetivo in self._visitados:
            return False, 0.0
        self._visitados.add(objetivo)
        
        self.rastro_prova.append(f"{indent}? Tentando provar: '{objetivo}'")
        
        # Busca regras que concluem o objetivo
        regras_candidatas = [r for r in self.regras if r.conclusao == objetivo]
        
        for regra in regras_candidatas:
            self.rastro_prova.append(
                f"{indent}  Usando regra: {regra.nome}"
            )
            
            todas_provadas = True
            cf_min = regra.confianca
            
            for condicao in regra.condicoes:
                sucesso, cf = self.provar(condicao, profundidade + 1)
                if not sucesso:
                    todas_provadas = False
                    self.rastro_prova.append(
                        f"{indent}  ✗ Falhou ao provar: '{condicao}'"
                    )
                    break
                cf_min = min(cf_min, cf)
            
            if todas_provadas:
                self.rastro_prova.append(
                    f"{indent}✓ '{objetivo}' PROVADO (CF={cf_min:.2f})"
                )
                return True, cf_min
        
        self.rastro_prova.append(f"{indent}✗ '{objetivo}' NÃO PODE SER PROVADO")
        return False, 0.0


# ─── Rede Bayesiana Simplificada ─────────────────────────────────────────────

class RedeBayesianaSimples:
    """
    Rede Bayesiana simplificada para diagnóstico médico.
    Suporta apenas consultas de variáveis com priors independentes.
    
    Para uma implementação completa, use a biblioteca pgmpy.
    """
    
    def __init__(self):
        # Probabilidades a priori P(doença)
        self.priors = {
            "gripe": 0.05,       # 5% da população tem gripe em média
            "resfriado": 0.10,   # 10% tem resfriado
            "alergia": 0.15,     # 15% tem alergia
            "covid19": 0.03,     # 3% (varia com surtos)
        }
        
        # P(sintoma | doença) — verossimilhanças
        # Estrutura: {sintoma: {doença: P(sintoma | doença)}}
        self.verossimilhancas = {
            "febre": {
                "gripe": 0.90, "resfriado": 0.30,
                "alergia": 0.05, "covid19": 0.85
            },
            "tosse": {
                "gripe": 0.80, "resfriado": 0.75,
                "alergia": 0.60, "covid19": 0.70
            },
            "espirros": {
                "gripe": 0.60, "resfriado": 0.85,
                "alergia": 0.90, "covid19": 0.20
            },
            "dor_muscular": {
                "gripe": 0.85, "resfriado": 0.20,
                "alergia": 0.05, "covid19": 0.80
            },
            "perda_olfato": {
                "gripe": 0.10, "resfriado": 0.15,
                "alergia": 0.10, "covid19": 0.65
            },
            "coriza": {
                "gripe": 0.70, "resfriado": 0.90,
                "alergia": 0.85, "covid19": 0.40
            },
        }
    
    def diagnosticar(self, sintomas: List[str],
                     verbose: bool = True) -> Dict[str, float]:
        """
        Calcula P(doença | sintomas) usando Bayes ingênuo (Naive Bayes).
        
        Assume independência condicional entre sintomas dado a doença.
        (Simplificação — Naive Bayes não é uma Rede Bayesiana completa)
        
        P(D|S₁,...,Sₙ) ∝ P(D) × ∏ P(Sᵢ|D)
        
        Returns:
            Dicionário {doença: probabilidade_posterior} normalizado
        """
        posteriors = {}
        
        for doenca, prior in self.priors.items():
            # P(doença | sintomas) ∝ P(doença) × ∏ P(sintoma | doença)
            likelihood = 1.0
            for sintoma in sintomas:
                if sintoma in self.verossimilhancas:
                    p_sint_dado_doenca = self.verossimilhancas[sintoma].get(doenca, 0.01)
                    likelihood *= p_sint_dado_doenca
                    
            posteriors[doenca] = prior * likelihood
        
        # Normalizar para que somem 1
        total = sum(posteriors.values())
        if total > 0:
            posteriors = {d: p/total for d, p in posteriors.items()}
        
        if verbose:
            print(f"\n🔬 Diagnóstico Bayesiano")
            print(f"   Sintomas observados: {', '.join(sintomas)}")
            print(f"\n   {'Doença':<15} {'Prior':>8} {'Posterior':>10} {'Variação':>10}")
            print(f"   {'-'*45}")
            for doenca, posterior in sorted(posteriors.items(),
                                            key=lambda x: -x[1]):
                prior = self.priors[doenca]
                variacao = (posterior - prior) / prior * 100
                barra = "█" * int(posterior * 30)
                print(f"   {doenca:<15} {prior:>8.1%} {posterior:>10.1%} "
                      f"{variacao:>+9.0f}%  {barra}")
        
        return posteriors
    
    def explicar_bayes(self, sintoma: str, doenca: str):
        """Demonstra o cálculo de Bayes para um sintoma e doença específicos."""
        prior = self.priors.get(doenca, 0)
        p_sint_dado_doenca = self.verossimilhancas.get(sintoma, {}).get(doenca, 0)
        
        # P(sintoma) = Σ P(sintoma|doença) × P(doença) para todas as doenças
        p_sintoma = sum(
            self.verossimilhancas.get(sintoma, {}).get(d, 0.01) * p
            for d, p in self.priors.items()
        )
        
        posterior = (p_sint_dado_doenca * prior) / max(p_sintoma, 1e-10)
        
        print(f"\n📐 Cálculo de Bayes:")
        print(f"   P({doenca}) = {prior:.2f}  [prior]")
        print(f"   P({sintoma}|{doenca}) = {p_sint_dado_doenca:.2f}  [verossimilhança]")
        print(f"   P({sintoma}) = {p_sintoma:.2f}  [evidência]")
        print(f"   P({doenca}|{sintoma}) = {p_sint_dado_doenca:.2f} × {prior:.2f} / {p_sintoma:.2f}")
        print(f"                        = {posterior:.2f}  [posterior]")


# ─── Base de Conhecimento Médico ─────────────────────────────────────────────

def criar_base_regras_medica() -> List[Regra]:
    """
    Cria uma base de regras para diagnóstico de doenças respiratórias comuns.
    Baseada em critérios clínicos simplificados (fins educativos apenas).
    """
    return [
        # ── Gripe ──────────────────────────────────────────────────────────
        Regra(
            nome="GRIPE_BASICA",
            condicoes={"febre_alta", "dor_muscular", "tosse"},
            conclusao="gripe_possivel",
            confianca=0.80,
            explicacao="Tríade clássica da gripe: febre alta + mialgias + tosse"
        ),
        Regra(
            nome="GRIPE_CONFIRMADA",
            condicoes={"gripe_possivel", "inicio_abrupto"},
            conclusao="gripe_confirmada",
            confianca=0.90,
            explicacao="Início abrupto é característico da gripe vs. resfriado"
        ),
        Regra(
            nome="GRIPE_TRATAMENTO",
            condicoes={"gripe_confirmada"},
            conclusao="indicar_antiviral_oseltamivir",
            confianca=0.85,
            explicacao="Oseltamivir em até 48h do início dos sintomas"
        ),
        
        # ── Resfriado ──────────────────────────────────────────────────────
        Regra(
            nome="RESFRIADO_BASICO",
            condicoes={"coriza", "espirros", "tosse_leve"},
            conclusao="resfriado_possivel",
            confianca=0.75,
            explicacao="Sintomas típicos de resfriado (rhinovirus)"
        ),
        Regra(
            nome="RESFRIADO_CONFIRMA",
            condicoes={"resfriado_possivel", "sem_febre_alta"},
            conclusao="resfriado_confirmado",
            confianca=0.85,
            explicacao="Resfriado raramente causa febre alta em adultos"
        ),
        Regra(
            nome="RESFRIADO_TRATAMENTO",
            condicoes={"resfriado_confirmado"},
            conclusao="indicar_tratamento_sintomatico",
            confianca=1.0,
            explicacao="Não há tratamento específico; sintomáticos"
        ),
        
        # ── Alergia Respiratória ────────────────────────────────────────────
        Regra(
            nome="ALERGIA_BASICA",
            condicoes={"espirros", "olhos_lacrimejando", "sem_febre"},
            conclusao="alergia_possivel",
            confianca=0.85,
            explicacao="Tríade alérgica: espirros + lacrimejamento + ausência de febre"
        ),
        Regra(
            nome="ALERGIA_CRONICA",
            condicoes={"alergia_possivel", "historico_alergico"},
            conclusao="rinite_alergica",
            confianca=0.90,
            explicacao="Histórico familiar/pessoal aumenta probabilidade"
        ),
        Regra(
            nome="ALERGIA_TRATAMENTO",
            condicoes={"rinite_alergica"},
            conclusao="indicar_anti_histaminico",
            confianca=0.95,
            explicacao="Anti-histamínicos de 2ª geração são primeira linha"
        ),
        
        # ── Urgência ────────────────────────────────────────────────────────
        Regra(
            nome="URGENCIA_RESPIRATORIA",
            condicoes={"dificuldade_respirar", "cianose"},
            conclusao="urgencia_medica",
            confianca=0.99,
            explicacao="Sinal de alarme: risco de vida imediato"
        ),
        Regra(
            nome="ENCAMINHAR_URGENCIA",
            condicoes={"urgencia_medica"},
            conclusao="encaminhar_emergencia",
            confianca=1.0,
            explicacao="Encaminhar imediatamente para emergência"
        ),
    ]


# ─── Demonstração Completa ────────────────────────────────────────────────────

def demonstrar_sistema_especialista():
    """Demonstra o sistema especialista completo."""
    print("=" * 65)
    print("SISTEMA ESPECIALISTA MÉDICO — Demonstração Educacional")
    print("ATENÇÃO: Este é um sistema simplificado para fins didáticos.")
    print("Não substitui consulta médica profissional!")
    print("=" * 65)
    
    regras = criar_base_regras_medica()
    
    # ── Caso 1: Gripe ──────────────────────────────────────────────────────
    print("\n📋 CASO 1: Paciente com gripe")
    print("   Sintomas: febre alta, dor muscular, tosse, início abrupto")
    
    sintomas_caso1 = {"febre_alta", "dor_muscular", "tosse", "inicio_abrupto"}
    
    print("\n🔄 Encadeamento para FRENTE:")
    motor_frente = MotorEncadeamentoFrente(regras)
    conclusoes = motor_frente.inferir(sintomas_caso1, verbose=True)
    
    print(f"\n   📊 Conclusões derivadas:")
    for conclusao, cf in sorted(conclusoes.items(), key=lambda x: -x[1]):
        print(f"      • {conclusao} (CF={cf:.2f})")
    
    # ── Caso 2: Backward chaining ──────────────────────────────────────────
    print("\n\n📋 CASO 2: Verificando hipótese de gripe confirmada")
    print("   Fatos disponíveis:", sintomas_caso1)
    
    print("\n🔄 Encadeamento para TRÁS:")
    print("   Objetivo: provar 'gripe_confirmada'")
    
    motor_tras = MotorEncadeamentoTras(regras, sintomas_caso1)
    sucesso, confianca = motor_tras.provar("gripe_confirmada")
    
    for linha in motor_tras.rastro_prova:
        print(f"   {linha}")
    
    print(f"\n   Resultado: {'PROVADO ✓' if sucesso else 'NÃO PROVADO ✗'} "
          f"(CF={confianca:.2f})")
    
    # ── Diagnóstico Bayesiano ──────────────────────────────────────────────
    print("\n\n📊 DIAGNÓSTICO BAYESIANO")
    rede = RedeBayesianaSimples()
    
    # Caso com febre, dor muscular e tosse
    sintomas_bayes = ["febre", "dor_muscular", "tosse"]
    rede.diagnosticar(sintomas_bayes, verbose=True)
    
    # Demonstração do cálculo de Bayes
    rede.explicar_bayes("febre", "gripe")
    
    # ── Caso com sintomas de COVID ─────────────────────────────────────────
    print("\n\n📊 DIAGNÓSTICO BAYESIANO — Caso suspeito de COVID")
    sintomas_covid = ["febre", "perda_olfato", "tosse", "dor_muscular"]
    rede.diagnosticar(sintomas_covid, verbose=True)


def comparar_formalismos():
    """Compara os diferentes formalismos de representação."""
    print("\n" + "=" * 65)
    print("COMPARAÇÃO DE FORMALISMOS DE REPRESENTAÇÃO DO CONHECIMENTO")
    print("=" * 65)
    
    comparacao = {
        "Lógica Proposicional": {
            "expressividade": "Baixa (sem objetos, sem variáveis)",
            "eficiencia": "Alta (decidível, NP-completo)",
            "incerteza": "Não suporta",
            "aplicacoes": "Circuitos booleanos, verificação formal simples",
            "pros": ["Simples", "Decidível", "Ferramentas maduras"],
            "contras": ["Não escala", "Sem variáveis", "Sem incerteza"],
        },
        "Lógica de 1ª Ordem": {
            "expressividade": "Alta (quantificadores, funções, relações)",
            "eficiencia": "Semi-decidível (pode não terminar)",
            "incerteza": "Não suporta nativamente",
            "aplicacoes": "Prolog, bancos de dados dedutivos, ontologias",
            "pros": ["Muito expressiva", "Fundamentos sólidos"],
            "contras": ["Semi-decidível", "Sem incerteza", "Lento na prática"],
        },
        "Sistemas de Regras": {
            "expressividade": "Média (domínio específico)",
            "eficiencia": "Alta para domínios pequenos",
            "incerteza": "Suporte parcial (fator de certeza)",
            "aplicacoes": "Sistemas especialistas, diagnóstico, triagem",
            "pros": ["Interpretável", "Aquisição fácil", "Explicativo"],
            "contras": ["Não escala", "Gargalo do especialista", "Frágil"],
        },
        "Redes Bayesianas": {
            "expressividade": "Alta (probabilística)",
            "eficiencia": "NP-difícil em geral; tratável em estruturas esparsas",
            "incerteza": "Suporte completo e rigoroso",
            "aplicacoes": "Diagnóstico, reconhecimento fala, spam, genômica",
            "pros": ["Raciocínio incerto rigoroso", "Interpretável", "Aprendível"],
            "contras": ["Difícil de construir", "Assume independência", "Lento"],
        },
    }
    
    for formalismo, props in comparacao.items():
        print(f"\n🔷 {formalismo}:")
        print(f"   Expressividade: {props['expressividade']}")
        print(f"   Eficiência:     {props['eficiencia']}")
        print(f"   Incerteza:      {props['incerteza']}")
        print(f"   ✓ Prós: {', '.join(props['pros'])}")
        print(f"   ✗ Contras: {', '.join(props['contras'])}")


# ─── Ponto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    demonstrar_sistema_especialista()
    comparar_formalismos()
```

---

## 9. Questões para Reflexão

1. **Lógica vs. Probabilidade**: Um sistema especialista médico baseado em lógica pura (SE-ENTÃO) e outro baseado em redes bayesianas. Em quais situações clínicas cada um seria mais adequado? O que acontece quando um paciente tem sintomas atípicos?

2. **Gargalo do especialista**: Sistemas especialistas clássicos falharam em parte pelo "knowledge acquisition bottleneck" — dificuldade de extrair conhecimento de especialistas humanos. Como o aprendizado de máquina resolve esse problema? Qual é o custo dessa solução?

3. **Independência de Naive Bayes**: O classificador Naive Bayes assume independência condicional entre features dado a classe. Febre e dor muscular são independentes dado que o paciente tem gripe? Essa suposição é válida? Por que Naive Bayes ainda funciona bem na prática apesar de violá-la?

4. **Encadeamento**: Quando preferir encadeamento para frente vs. para trás? Pense em um sistema de detecção de fraude bancária e em um sistema de diagnóstico médico. Qual abordagem é mais natural para cada?

5. **Ontologias na IA moderna**: Como ontologias e grafos de conhecimento (como o Google Knowledge Graph) complementam LLMs modernos? Por que não basta treinar um LLM em todo o texto da Wikipedia?

6. **Raciocínio causal**: Redes bayesianas capturam correlações, mas não necessariamente causalidade. Quais são as implicações disso para um sistema de diagnóstico médico? O que seria necessário para raciocinar sobre causas (não apenas correlações)?

---

## Referências

**[1]** RUSSELL, S.; NORVIG, P. **Inteligência Artificial: uma abordagem moderna**. 4. ed. GEN LTC, 2022. Cap. 7 (Agentes Lógicos), Cap. 8 (FOL), Cap. 12 (Probabilidade), Cap. 13 (Redes Bayesianas).

**[2]** FACELI, K. et al. **Inteligência Artificial: uma abordagem de aprendizado de máquina**. 2. ed. LTC, 2021.

**[3]** PEARL, J. **Probabilistic Reasoning in Intelligent Systems**. Morgan Kaufmann, 1988.
> *O livro fundamental sobre redes bayesianas.*

**[4]** MINSKY, M. A Framework for Representing Knowledge. *MIT AI Memo 306*, 1974.
> *O artigo original sobre frames.*

**[5]** SHORTLIFFE, E. H. **Computer-Based Medical Consultations: MYCIN**. Elsevier, 1976.
> *O sistema especialista médico pioneiro.*

**[6]** PEARL, J.; MACKENZIE, D. **The Book of Why: The New Science of Cause and Effect**. Basic Books, 2018.
> *Leitura acessível sobre raciocínio causal vs. correlacional.*

**[7]** GRUBER, T. R. A Translation Approach to Portable Ontology Specifications. *Knowledge Acquisition*, v. 5, n. 2, p. 199-220, 1993.

---

*Aula anterior: [Aula 03 — Busca e Resolução de Problemas](./aula-03-busca-e-resolucao-problemas.md)*  
*Próxima aula: [Aula 05 — Python para IA e Ciência de Dados](./aula-05-python-para-ia.md)*
