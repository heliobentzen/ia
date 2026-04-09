# Aula 60 — Uso Responsável de IA

> **Módulo 12 · Ética, Interpretabilidade e Uso Responsável de IA** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Conhecer os principais frameworks de IA responsável
- Compreender os riscos de IA em domínios críticos
- Refletir sobre o futuro da IA e o papel dos profissionais de tecnologia

---

## 1. Princípios de IA Responsável

Principais frameworks internacionais convergem em princípios comuns:

| Princípio | Descrição |
|-----------|-----------|
| **Transparência** | Sistemas compreensíveis por humanos |
| **Justiça (Fairness)** | Ausência de discriminação injusta |
| **Responsabilização** | Atribuição clara de responsabilidade |
| **Privacidade** | Proteção de dados pessoais |
| **Beneficência** | Sistemas que beneficiam a humanidade |
| **Não-maleficência** | Evitar danos intencionais e não intencionais |
| **Autonomia** | Respeitar agência e escolha dos usuários |
| **Supervisão humana** | Humano no loop para decisões críticas |

---

## 2. Frameworks Principais

- **NIST AI RMF** (EUA): Risk Management Framework para IA
- **EU AI Act** (Europa): regulação por nível de risco (proibido, alto, médio, baixo)
- **Responsible AI Standard** (Microsoft): 6 princípios + ferramentas (Fairlearn, InterpretML)
- **Trustworthy AI** (Google): 7 princípios + Model Card, Datasheet
- **ANPD** (Brasil): diretrizes vinculadas à LGPD

---

## 3. EU AI Act — Classificação por Risco

```
RISCO INACEITÁVEL (Proibido):
├── Pontuação social por governos
├── Exploração de vulnerabilidades
└── Vigilância biométrica em massa

RISCO ALTO (Regulado):
├── Infraestrutura crítica
├── Educação e seleção profissional
├── Crédito, seguros
├── Aplicação da lei
└── Saúde e medicina

RISCO MÉDIO (Transparência obrigatória):
├── Chatbots (devem se identificar como IA)
└── Deepfakes (devem ser rotulados)

RISCO BAIXO:
└── Filtros de spam, recomendações
```

---

## 4. Cartão de Modelo (Model Card)

Documentação padronizada para modelos de ML. Proposto por Google em 2019.

```markdown
# Model Card — [Nome do Modelo]

## Detalhes do Modelo
- Desenvolvido por: ...
- Data: ...
- Versão: ...
- Tipo: [classificação / regressão / geração]
- Framework: [sklearn / TensorFlow / PyTorch]

## Uso Pretendido
- Casos de uso primários: ...
- Casos de uso fora do escopo: ...

## Dados de Treinamento
- Fonte dos dados: ...
- Período: ...
- Pré-processamento: ...

## Métricas de Performance
| Métrica | Geral | Grupo A | Grupo B |
|---------|-------|---------|---------|
| Acurácia | 0.92 | 0.93 | 0.88 |
| F1 | 0.91 | 0.92 | 0.86 |

## Considerações Éticas
- Viés potencial: ...
- Limitações conhecidas: ...
- Populações afetadas: ...

## Avaliação de Risco
- Risco de dano: [baixo/médio/alto]
- Supervisão humana recomendada: [sim/não]
```

---

## 5. Reflexões sobre o Futuro

**Riscos de curto prazo:**
- Desinformação e deepfakes
- Deslocamento de empregos
- Concentração de poder tecnológico
- Dependência de sistemas opacos

**Riscos de longo prazo:**
- AGI (Artificial General Intelligence) e alinhamento
- Corrida armamentista de IA

**Oportunidades:**
- Acelerar ciência (AlphaFold, descoberta de medicamentos)
- Educação personalizada
- Acessibilidade
- Sustentabilidade e clima

---

## 6. O Papel do Profissional de IA

Como desenvolvedor de sistemas de IA, você tem responsabilidades:

1. **Questionar** os dados, os objetivos e os impactos
2. **Documentar** decisões e limitações
3. **Testar** para populações diversas
4. **Incluir** perspectivas multidisciplinares (direito, psicologia, serviço social)
5. **Recusar** projetos claramente prejudiciais
6. **Manter-se atualizado** sobre regulações e melhores práticas

> "Com grande poder vem grande responsabilidade." — não é só do Spider-Man.

---

## Questões para Reflexão
1. Como o EU AI Act poderia afetar o desenvolvimento de sistemas de IA no Brasil?
2. Quais sistemas de IA do seu dia a dia são de alto risco segundo o EU AI Act?
3. Que tipo de IA você se recusaria a construir? Por quê?

## Referências
- Russell & Norvig, cap. 27
- Faceli et al., cap. 13
- euaiact.eu, nist.gov/ai

---
*🎓 Curso concluído! Revise as práticas em [praticas/](../../praticas/README.md)*
