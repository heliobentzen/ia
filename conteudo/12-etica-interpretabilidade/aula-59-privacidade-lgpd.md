# Aula 59 — Privacidade e LGPD

> **Módulo 12 · Ética, Interpretabilidade e Uso Responsável de IA** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender os princípios da LGPD e GDPR aplicados ao ML
- Identificar dados pessoais e sensíveis em projetos de ML
- Conhecer técnicas de privacy-preserving ML

---

## 1. LGPD — Lei Geral de Proteção de Dados (Lei 13.709/2018)

**Dados Pessoais:** qualquer dado que identifique ou possa identificar uma pessoa natural.
**Dados Sensíveis:** origem racial, saúde, biometria, religião, orientação sexual.

**Bases legais para tratamento:**
1. Consentimento explícito
2. Cumprimento de obrigação legal
3. Execução de políticas públicas
4. Legítimo interesse
5. Proteção da vida
6. Tutela da saúde
7. Execução de contrato
8. Exercício de direitos

---

## 2. Direitos dos Titulares (impacto em ML)

| Direito | Implicação em sistemas de ML |
|---------|------------------------------|
| Confirmação e acesso | Saber se seus dados são usados |
| Correção | Atualizar dados no treino |
| Anonimização/exclusão | Direito ao esquecimento — Machine Unlearning |
| Portabilidade | Exportar dados em formato estruturado |
| Oposição | Recusar tratamento, inclusive para ML |
| **Decisões automatizadas** | **Direito à revisão humana** |

---

## 3. Decisões Automatizadas (Art. 20 da LGPD)

> "O titular dos dados tem direito a solicitar a revisão de decisões tomadas unicamente com base em tratamento automatizado de dados pessoais que afetem seus interesses."

**Impacto em ML:**
- Sistemas de crédito, seleção de candidatos, diagnóstico médico
- Exige explicabilidade (SHAP, LIME)
- Deve haver processo de revisão humana

---

## 4. Técnicas de Privacy-Preserving ML

### 4.1 Anonimização e Pseudonimização

```python
import pandas as pd
import hashlib
import re

def pseudonimizar(df, colunas_pii):
    df_anon = df.copy()
    for col in colunas_pii:
        # Hash consistente (mesma entrada → mesmo hash)
        df_anon[col] = df[col].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
        )
    return df_anon

def anonimizar_email(email):
    user, domain = email.split('@')
    return f"{user[0]}***@{domain}"

# Dados sintéticos
df = pd.DataFrame({
    'nome': ['João Silva', 'Maria Santos'],
    'email': ['joao@email.com', 'maria@email.com'],
    'cpf': ['123.456.789-00', '987.654.321-00'],
    'renda': [5000, 8000]
})

df_anon = pseudonimizar(df, ['nome', 'cpf'])
df_anon['email'] = df['email'].apply(anonimizar_email)
print(df_anon)
```

### 4.2 k-Anonimidade

```python
def verificar_k_anonimidade(df, quasi_identificadores, k=5):
    groups = df.groupby(quasi_identificadores).size()
    violacoes = groups[groups < k]
    if len(violacoes) == 0:
        print(f"✅ Dataset é {k}-anônimo")
    else:
        print(f"❌ {len(violacoes)} grupos violam {k}-anonimidade:")
        print(violacoes)
    return violacoes
```

### 4.3 Privacidade Diferencial (conceito)

Adiciona ruído calibrado às consultas para que presença/ausência de um indivíduo não afete o resultado de forma detectável.

$$P[M(D) \in S] \leq e^{\epsilon} \cdot P[M(D') \in S] + \delta$$

---

## Questões para Reflexão
1. Qual a diferença entre anonimização e pseudonimização segundo a LGPD?
2. Um modelo treinado em dados pessoais e depois exportado ainda contém dados pessoais?
3. O que é "machine unlearning" e por que é tecnicamente difícil?

## Referências
- Lei 13.709/2018 (LGPD) — planalto.gov.br
- Russell & Norvig, cap. 27
- Faceli et al., cap. 13

---
*Próxima aula → [Aula 60: Uso Responsável de IA](aula-60-uso-responsavel.md)*
