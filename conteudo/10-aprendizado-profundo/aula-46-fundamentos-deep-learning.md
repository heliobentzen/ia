# Aula 46 — Fundamentos do Aprendizado Profundo

> **Módulo 10 · Introdução ao Aprendizado Profundo** | ⏱ 45 minutos

## Objetivos de Aprendizagem
- Compreender o que diferencia Deep Learning de ML clássico
- Conhecer os principais frameworks e hardware
- Configurar o ambiente de desenvolvimento (GPU/Colab)

---

## 1. O que é Deep Learning?

Deep Learning é um subcampo do ML que usa redes neurais com **muitas camadas** para aprender representações hierárquicas dos dados.

```
ML Clássico:            Deep Learning:
Features manuais        Features aprendidas automaticamente
   ↓                         ↓
Classificador           Camadas empilhadas
                         ↓
                        Representações brutas → features baixo nível
                        → features médio nível → features alto nível
                        → predição
```

**Por que agora?**
- Volume de dados: ImageNet (1M imagens), Common Crawl, etc.
- Hardware: GPUs NVIDIA, TPUs do Google
- Avanços algorítmicos: ReLU, Dropout, BatchNorm, Transformers

---

## 2. Principais Frameworks

| Framework | Empresa | Características |
|-----------|---------|----------------|
| TensorFlow/Keras | Google | Produção, deployment, TFLite |
| PyTorch | Meta | Pesquisa, flexibilidade, debug fácil |
| JAX | Google | Pesquisa, auto-diferenciação funcional |
| HuggingFace | HF | NLP, modelos pré-treinados |

---

## 3. Configuração do Ambiente

```python
# Verificar GPU disponível
import tensorflow as tf
import torch

print("TensorFlow:", tf.__version__)
print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))

print("PyTorch:", torch.__version__)
print("CUDA disponível:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Google Colab: verificar GPU
# !nvidia-smi
```

---

## 4. Quando Usar Deep Learning?

| Situação | Recomendação |
|---------|-------------|
| Dados tabulares pequenos (<100k) | ML clássico (GBM, RF) |
| Dados tabulares grandes | DL ou GBM (comparar) |
| Imagens | CNN / Vision Transformers |
| Texto, linguagem | Transformers, BERT, GPT |
| Áudio / Fala | CNN 1D, Transformers |
| Vídeo | CNN + RNN / ViT |
| Dados escassos | Transfer learning |

---

## Questões para Reflexão
1. Por que redes neurais profundas só se tornaram práticas nos últimos 15 anos?
2. Para um problema de classificação com 1000 exemplos e 20 features, você usaria DL ou ML clássico?
3. Qual a diferença entre GPU e TPU para treinamento de modelos?

## Referências
- Géron, cap. 10
- Faceli et al., cap. 7

---
*Próxima aula → [Aula 47: CNNs](aula-47-cnn.md)*
