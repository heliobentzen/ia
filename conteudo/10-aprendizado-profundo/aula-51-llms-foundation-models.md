# Aula 51 — LLMs e Foundation Models

> **Módulo 10 · Introdução ao Aprendizado Profundo** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender o que são LLMs e Foundation Models
- Aplicar prompting eficiente (zero-shot, few-shot, chain-of-thought)
- Implementar RAG (Retrieval-Augmented Generation) básico

---

## 1. O que são LLMs?

Large Language Models (LLMs) são modelos Transformer treinados em enormes quantidades de texto com o objetivo de prever o próximo token. Emergem capacidades surpreendentes de raciocínio, tradução e código.

| Modelo | Empresa | Contexto | Destaques |
|--------|---------|---------|---------|
| GPT-4o | OpenAI | 128k tokens | Multimodal, código |
| Claude 3.5 Sonnet | Anthropic | 200k tokens | Raciocínio longo |
| Gemini 1.5 Pro | Google | 1M tokens | Multimodal |
| Llama 3.1 | Meta | 128k tokens | Open source |
| Mistral | Mistral AI | 32k tokens | Eficiente, open |

---

## 2. Prompting Eficiente

```python
# Zero-shot: apenas a instrução
zero_shot = """
Classifique o sentimento (positivo/negativo/neutro) do texto abaixo.
Responda apenas com a classificação.

Texto: "O produto chegou no prazo e a qualidade superou minhas expectativas!"
Resposta:"""

# Few-shot: exemplos no prompt
few_shot = """
Classifique o sentimento:

Texto: "Produto horrível, não comprem!" → Negativo
Texto: "Entrega rápida e produto impecável." → Positivo
Texto: "O produto é ok, nada demais." → Neutro

Texto: "A embalagem estava danificada mas o produto funcionou." →"""

# Chain-of-Thought: pensar passo a passo
cot = """
Problema: Uma empresa tem 120 funcionários. 30% trabalham remotamente.
Quantos trabalham presencialmente?

Vamos pensar passo a passo:
1. ..."""

# Usando a API da OpenAI
from openai import OpenAI

client = OpenAI()  # usa OPENAI_API_KEY do ambiente

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Você é um assistente especialista em análise de sentimentos."},
        {"role": "user", "content": few_shot}
    ],
    temperature=0.0,  # determinístico para classificação
    max_tokens=50
)
print(response.choices[0].message.content)
```

---

## 3. RAG — Retrieval-Augmented Generation

Permite que o LLM responda sobre documentos que não estão no seu treinamento:

```python
# Simplified RAG pipeline
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import OpenAI

# 1. Indexar documentos (offline)
model_emb = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

documentos = [
    "A LGPD (Lei Geral de Proteção de Dados) foi sancionada em 2018 e vigora desde 2020.",
    "O titular dos dados tem direito de acesso, correção e exclusão de seus dados pessoais.",
    "O encarregado (DPO) é o responsável pela proteção de dados na organização.",
    "A ANPD é a autoridade nacional que fiscaliza o cumprimento da LGPD.",
]
embeddings = model_emb.encode(documentos)

def buscar_contexto(pergunta, top_k=2):
    q_emb = model_emb.encode([pergunta])
    sims = (embeddings @ q_emb.T).squeeze()
    idx = sims.argsort()[::-1][:top_k]
    return [documentos[i] for i in idx]

def responder(pergunta):
    contexto = buscar_contexto(pergunta)
    prompt = f"""Responda à pergunta com base apenas no contexto fornecido.

Contexto:
{chr(10).join(contexto)}

Pergunta: {pergunta}
Resposta:"""
    
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    return resp.choices[0].message.content

# Teste
print(responder("Quem fiscaliza a LGPD no Brasil?"))
```

---

## 4. Técnicas Avançadas de Fine-Tuning

| Técnica | Descrição | Parâmetros treinados |
|---------|-----------|---------------------|
| Full fine-tuning | Treina todo o modelo | 100% |
| LoRA | Low-Rank Adaptation: matrizes de baixo rank | ~0.1–1% |
| QLoRA | LoRA com quantização 4-bit | ~0.1% |
| Prompt tuning | Apenas prompts "soft" | <<1% |

---

## Questões para Reflexão
1. O que é "hallucination" em LLMs e como o RAG pode mitigá-la?
2. Por que LLMs com temperatura=0 são determinísticos?
3. Qual a diferença entre fine-tuning e RAG para adaptar um LLM a um domínio específico?

## Referências
- Tunstall et al., cap. 10–12
- Géron, cap. 16

---
*Módulo 10 concluído! Próximo → [Módulo 11: Aplicações de ML em Problemas Reais](../11-aplicacoes-reais/README.md)*
