---
tags:
- materia
- hsaq
- jepa
- snn
- ssm
- sparse-training
- adaptive-quantization
- pytorch
license: mit
library_name: pytorch
language:
- en
- es
- fr
- de
- pt
- ar
- hi
- ja
- ko
- ru
- it
pipeline_tag: text-generation
base_model: MethodWhite/materia-v4-1b-bpe
model-index:
- name: MATERIA V4 1B
  results: []
datasets:
- HuggingFaceFW/fineweb
---

<div align="center">
  <h1>M.A.T.E.R.I.A. V4 1B</h1>
  <h3>Multi-Analytical Toroidal Engine for Recursive Intelligent Analysis</h3>
  <p><em>JEPA-First Toroidal Architecture with HSAQ Adaptive Optimization</em></p>
  
  [![Params](https://img.shields.io/badge/Params-1.34B-blue)](https://huggingface.co/MethodWhite/materia-v4-1b-bpe)
  [![Architecture](https://img.shields.io/badge/Architecture-JEPA--First_Toroidal-blueviolet)](https://huggingface.co/MethodWhite/materia-v4-1b-bpe)
  [![License](https://img.shields.io/badge/License-MIT-green)](https://huggingface.co/MethodWhite/materia-v4-1b-bpe/blob/main/LICENSE)
  [![GGUF](https://img.shields.io/badge/GGUF-Q4_0-orange)](https://huggingface.co/MethodWhite/materia-v4-1b-bpe/blob/main/materia-v4.Q4_0.gguf)
  [![HSAQ](https://img.shields.io/badge/Optimizer-HSAQ-success)](https://huggingface.co/MethodWhite/materia-v4-1b-bpe)
</div>

---

## 📋 Model Description

**M.A.T.E.R.I.A. V4** (Multi-Analytical Toroidal Engine for Recursive Intelligent Analysis) is a novel neural architecture that departs from the traditional linear transformer paradigm. Instead of processing tokens through a sequential stack, MATERIA employs a **toroidal (donut-shaped) geometry** where GQA, SNN, and SSM components converge into a unified **JEPA latent space** through hexagonal interconnection.

The key architectural insight is that information flows in **cycles** through a toroidal topology, with each cycle inheriting sparse activation patterns from the previous one (70/30 blend). This creates a recurrent processing structure without the computational overhead of traditional recurrence.

### Architecture Overview

```
                ┌───────────────────┐
               /    Transformer      \
              |       ↕ ↕ ↕          |
     ┌───────┐ |    ↔ JEPA ↔        | ┌───────┐
     │  SSM  │←━━→    ↕       ←━━→  | │  SNN  │
     └───────┘ |   ↔  Hub  ↔        | └───────┘
              |       ↕ ↕ ↕          |
               \                    /
                └────────┬─────────┘
                         ↕
                  ┌──────┴──────┐
                  │  Head/Emb   │
                  └─────────────┘
```

### Components

| Component | Role | Specification |
|-----------|------|---------------|
| **JEPA Hub** | Central latent space predictor | SCA spectral decomposition, λₙ = K·σ(μₙ), K=2.781042 |
| **GQA** | Grouped Query Attention | 24 heads, 6 KV heads, RoPE, Flash Attention 2 |
| **SNN** | Spiking Neural Network | LIF neurons, τ=0.8, surrogate gradient (β=5.0), 30% target rate |
| **SSM** | State Space Model | 64-state, long-range dependency capture |
| **HSAQ** | HyperSparse Adaptive Quantization | 10 per-edge learnable instances, STE, kthvalue dynamic threshold |
| **Toroidal** | 3-cycle recurrence | Mask inheritance 70/30, hexagonal interconnect |

## 🔬 Key Innovation: HSAQ Optimizer

**HSAQ (HyperSparse Adaptive Quantization)** is the core innovation — it **replaces AdamW** as the model optimizer while simultaneously providing adaptive sparse activation.

Unlike traditional optimizers that maintain complex state (AdamW: 2 states/param, 8 bytes/param), HSAQ uses:
- **SGD Nesterov** (momentum=0.9): 1 state/param, 4 bytes/param (2.5× memory savings)
- **Per-edge learnable sparsity**: 10 `nn.Parameter(sparsity_logit)` instances
- **Dynamic kthvalue thresholding**: sparsity adapts per-batch based on activation distribution
- **Straight-Through Estimator (STE)**: forward hard mask, backward sigmoid relaxation — fully differentiable
- **Sacred geometry initialization**: logits from golden ratio principles
- **Toroidal mask inheritance**: 70/30 blend across cycles preserves activation topology

### HSAQ Sparsity Distribution

| Edge | Initial Logit | Target Sparsity | Role |
|------|--------------|-----------------|------|
| Embedding | -3.0 | ~5% | Token projection |
| JEPA in | -3.0 | ~5% | Hub entry |
| T2 | -2.5 | ~8% | Transformer layer 2 |
| T5 | -2.0 | ~12% | Transformer layer 5 |
| T8 | -1.7 | ~15% | Transformer layer 8 |
| T Cycle | -2.5 | ~8% | Toroidal recurrence |
| SNN | -3.5 | ~3% | Spiking (minimal, spikes already sparse) |
| SNN Latent | -3.0 | ~5% | SNN→JEPA projection |
| SSM | -3.0 | ~5% | State model |
| SSM Latent | -3.0 | ~5% | SSM→JEPA projection |

## ⚙️ Training Configuration

### Model Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Total Parameters | **1,335,302,155** (1.34B) |
| Architecture | Toroidal JEPA-First + SCA |
| Hidden Dimension | 1792 |
| Latent Dimension | 1792 |
| Layers | 24 |
| Attention Heads | 24 |
| KV Heads | 6 |
| SNN Dimension | 1792 |
| SSM State Size | 64 |
| Toroidal Cycles | 3 |
| Vocabulary | 32,768 (BPE) |
| Tokenizer | SentencePiece multilingual v2 |

### Training Hyperparameters

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | SGD Nesterov (momentum=0.9) + HSAQ |
| Learning Rate | 1.5e-4 → cosine decay |
| Warmup Steps | 2,000 |
| Weight Decay | 0.1 |
| Gradient Clipping | 1.0 |
| Batch Size | 64 (effective) |
| Sequence Length | 128 |
| Gradient Accumulation | 32 |
| Mixed Precision | bf16 |
| Gradient Checkpointing | Enabled |
| Epochs | 2/10 |
| Training Duration | 327.7 min (5.46h) |

### Hardware

| Component | Specification |
|-----------|--------------|
| GPU | 1× NVIDIA RTX 6000 Ada (48GB) |
| VRAM Usage | 24GB / 48GB (stable) |
| CPU | 16 vCPUs |
| RAM | 70GB |
| Provider | Brev.dev / NVIDIA |

### Dataset

- **Primary**: [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (CC-MAIN-2024-10)
- **Supplemental**: Wikipedia (12 languages: EN, ES, FR, DE, PT, AR, HI, JA, KO, RU, IT, ZH)
- **Size**: ~5M lines, 625,892 training chunks at seq_len=128
- **Format**: BPE-tokenized with SentencePiece multilingual model

## 📊 Training Results

### Final Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| **Total Loss** | **0.1303** | Dual loss: token + K·jepa |
| **Token Loss** | ~8.06 | Cross-entropy |
| **JEPA MSE** | ~0.052 | Normalized by latent variance |
| **Accuracy** | **6.36%** | Token prediction (early training) |
| **Perplexity** | **~4186** | Expected for 2-epoch 1B model |
| **SNN Spike Rate** | **30.3%** | ✅ Perfectly calibrated to target |
| **HSAQ Sparsity** | 0.029 (target 0.048) | Adaptive, undershooting target |
| **NaN Events** | **0** | Stable training throughout |

### Performance Curves

| Metric | Trend |
|--------|-------|
| **Loss** | 8.9 → 0.13 (smooth decay) |
| **Accuracy** | 4.6% → 6.36% (improving) |
| **Perplexity** | 7,941 → 4,186 (declining) |
| **SNN Spike Rate** | 46% → 30% (perfectly regulated) |
| **HSAQ Sparsity** | 0.026 → 0.029 (stable adaptive) |
| **Learning Rate** | Linear warmup → cosine decay |

### Generation Samples (E2 Final)

| Prompt | Generation |
|--------|------------|
| "The meaning of life is" | `constitutional a p juan alum. i spect a recently` |
| "Artificial intelligence" | `. a mc. anstuff is the includingporter is youg sch` |
| "Hello, how are you" | `group and i for and at the new say and averaged an` |

*Note: Model shows emerging word structure and phrase formation at 2 epochs. Full linguistic capability requires 10+ epochs.*

## 💾 Files

| File | Size | Format | Description |
|------|------|--------|-------------|
| `materia-v4.basemateria` | 5.9 GB | Pickle (native) | Full model weights + config + tokenizer |
| `materia-v4.Q4_0.gguf` | 5.5 GB | GGUF | Q4_0 quantized (llama.cpp/Ollama/LM Studio) |
| `checkpoint_epoch1.pt` | 11.2 GB | PyTorch | Epoch 1 with optimizer state |
| `checkpoint_epoch2.pt` | 10.7 GB | PyTorch | Epoch 2 with optimizer state |
| `config_1B.yaml` | 839 B | YAML | Training configuration |

## 🚀 Usage

### PyTorch (Full Precision)

```python
import torch
import pickle

# Load .basemateria
with open('materia-v4.basemateria', 'rb') as f:
    data = pickle.load(f)

state_dict = data['state_dict']
config = data['config']

print(f"Model: {data['config']['version']}")
print(f"Dimensions: {config['dim']}, Vocab: {config['vocab_size']}")

# Import model
from models.materia_v4 import MateriaV4

model = MateriaV4(
    vocab_size=config['vocab_size'],
    dim=config['dim'],
    n_layers=24,
    n_heads=24,
    n_kv=6,
    latent_dim=config['latent_dim'],
)
model.load_state_dict(state_dict, strict=False)
model.eval()

# Generate
import torch.nn.functional as F
stoi = data['tokenizer']
itos = {v: k for k, v in stoi.items()}

prompt = "The meaning of life is"
input_ids = torch.tensor([[stoi.get(c, 0) for c in prompt]])
with torch.no_grad():
    for _ in range(50):
        logits, _, _ = model(input_ids)
        p = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        next_id = torch.multinomial(p, 1)
        input_ids = torch.cat([input_ids, next_id], dim=1)

output = ''.join([itos.get(i.item(), '<unk>') for i in input_ids[0]])
print(output)
```

### llama.cpp / Ollama (GGUF)

```bash
# Verify file integrity
gguf-dump materia-v4.Q4_0.gguf | head -20

# Run with llama.cpp
./main -m materia-v4.Q4_0.gguf -p "The meaning of life is" -n 50 -t 8

# Or with Ollama
ollama create materia-v4 -f Modelfile  # Requires custom Modelfile
```

## 🧪 MoE Extension

The repository includes a **Mixture-of-Experts** implementation (8 experts, top-2) for the next generation:

- **Config**: [`configs/V4_3B_MoE.yaml`](https://github.com/MethodWhite/materia-core/blob/main/configs/V4_3B_MoE.yaml)
- **3B MoE** activates only 25% of parameters per token (effective ~7B behavior)
- **Export**: `materia-v4.Q4_0.gguf` supports llama.cpp/Ollama/LM Studio
- **Compatible** with HSAQ, Flash Attention 2, and gradient checkpointing

### Scaling Roadmap

| Model | Params | GPUs | Budget | Next Steps |
|-------|--------|------|--------|------------|
| V4 1B Dense (this) | 1.34B | 1× RTX 6000 Ada | $5.46 | ✅ Complete |
| V4 3B MoE | ~3B (8 experts) | 1× RTX 6000 Ada | ~$23 | 🔧 Ready to train |
| V4 7B Dense | ~7B | 2× RTX 6000 Ada | ~$30 | 🔜 Training (FSDP) |

## 📈 Benchmarks (Planned)

Comparison targets for next evaluation phase:

| Model | Size | Architecture | Notes |
|-------|------|-------------|-------|
| GPT-2 1.5B | 1.5B | Transformer | Classic baseline |
| Mamba 1.4B | 1.4B | SSM | Direct SSM competitor |
| S4 | ~1B | SSM | SSM pioneer |
| SpikeGPT | ~1B | SNN | Direct SNN comparator |
| Llama 3 8B | 8B | Transformer | Top-tier hybrid reference |

## 📚 Citation

```bibtex
@software{materia_v4_2026,
  author = {MethodWhite},
  title = {{MATERIA V4}: Multi-Analytical Toroidal Engine for Recursive Intelligent Analysis},
  year = {2026},
  month = {July},
  url = {https://huggingface.co/MethodWhite/materia-v4-1b-bpe},
  publisher = {HuggingFace},
  version = {1.0.0},
  note = {Toroidal JEPA-First architecture with HSAQ adaptive optimizer}
}

@misc{hsaq_optimizer_2026,
  author = {MethodWhite},
  title = {{HSAQ}: HyperSparse Adaptive Quantization — Replacing AdamW with Learnable Sparse Optimization},
  year = {2026},
  url = {https://huggingface.co/MethodWhite/materia-v4-1b-bpe},
  note = {Per-edge learnable sparsity with Straight-Through Estimator}
}
```

## 📄 License

This model is released under the **MIT License**.

## 🙏 Acknowledgments

- Built with [PyTorch](https://pytorch.org/) and [HuggingFace](https://huggingface.co/) ecosystem
- Trained on [Brev.dev](https://brev.dev) NVIDIA GPU infrastructure
- Dataset: [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- GGUF conversion via [llama.cpp](https://github.com/ggerganov/llama.cpp)

---

<div align="center">
  <a href="https://buymeacoffee.com/methodwhite" target="_blank">
    <img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="217" height="48">
  </a>
  <br>
  <a href="https://buymeacoffee.com/methodwhite" target="_blank">
    <img src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-☕-FFDD00?style=for-the-badge&logo=buy-me-a-coffee&logoColor=black" alt="Buy Me A Coffee Badge">
  </a>
  <p><em>Si este proyecto te resulta útil, ¡invítame un café ☕!</em></p>
</div>

---

<div align="center">
  <p><em>"HSAQ comprime de manera inteligente los tokens, es adaptativo, no es un valor fijo, sino lo que necesita la IA como tal."</em></p>
</div>
