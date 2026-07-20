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
---

# MATERIA V4 1B — Multi-Analytical Toroidal Engine for Recursive Intelligent Analysis

![MATERIA Architecture](https://img.shields.io/badge/Architecture-JEPA--First_Toroidal-blueviolet)
![Params](https://img.shields.io/badge/Params-1.34B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## Model Description

**M.A.T.E.R.I.A. V4** is a **toroidal JEPA-First architecture** that converges GQA (Grouped Query Attention), SNN (Spiking Neural Network), and SSM (State Space Model) into a unified JEPA latent space. Unlike traditional linear transformers, MATERIA processes information in **toroidal cycles** with hexagonal interconnection, inspired by sacred geometry principles.

| Component | Description |
|-----------|-------------|
| **JEPA Hub** | Central latent space predictor with SCA (Spectral Coupling Analysis) |
| **GQA** | Grouped Query Attention with 24 heads, 6 KV heads, RoPE |
| **SNN** | Spiking Neural Network with LIF neurons, surrogate gradient, 30% target spike rate |
| **SSM** | State Space Model for long-range dependencies |
| **HSAQ** | HyperSparse Adaptive Quantization — replaces AdamW as optimizer |

## Key Innovation: HSAQ Optimizer

**HSAQ** (HyperSparse Adaptive Quantization) is a **learnable sparse activation optimization** method that:

- Replaces traditional optimizers (AdamW) with adaptive per-layer dynamic sparsity
- Uses **kthvalue** dynamic thresholding per batch — no fixed prune ratio
- Features **Straight-Through Estimator (STE)** for fully differentiable sparse masks
- 10 per-edge learnable `sparsity_logit` parameters with sacred-geometry initialization
- Toroidal mask inheritance (70/30 blend) across cycles
- Reduces optimizer memory by **2.5×** vs AdamW (SGD Nesterov momentum only)

> *"HSAQ comprime de manera inteligente los tokens, es adaptativo, no es un valor fijo, sino lo que necesita la IA como tal."*

## Architecture

```
                ┌─────────────┐
               ╱  Transformer  ╲
              │     ↕ ↕ ↕      │
     ┌───────┐│   ↔ JEPA ↔    │┌───────┐
     │  SSM  │←━━→   ↕   ←━━→││  SNN  │
     └───────┘│  ↔  Hub  ↔   │└───────┘
              │     ↕ ↕ ↕      │
               ╲              ╱
                └─────┬───────┘
                      ↕
                ┌─────┬───────┐
                │ Head/Emb    │
                └─────────────┘
```

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Parameters | **1,335,302,155** (1.34B) |
| Architecture | Toroidal JEPA-First + SCA |
| Dimensions | dim=1792, latent_dim=1792, n_layers=24 |
| Attention | n_heads=24, n_kv=6, RoPE |
| SNN | LIF threshold=0.001, tau=0.8, target rate=30% |
| SSM | State size=64 |
| HSAQ | 10 instances, per-edge learnable, init from sacred geometry |
| Tokenizer | BPE 32K vocab (SentencePiece multilingual) |
| Optimizer | SGD Nesterov (momentum=0.9) — HSAQ replaces AdamW |
| Learning Rate | 1.5e-4 → cosine decay, warmup 2000 steps |
| Weight Decay | 0.1 |
| Training Data | FineWeb (HuggingFaceFW/fineweb) + Wikipedia (multilingual) |
| GPU | 1× NVIDIA RTX 6000 Ada (48GB) |
| Batch Size | 64 (seq_len=128), grad_accum=32 |
| Epochs | 2/10 (E2 complete: 327.7 min) |

## Training Results

| Metric | Value |
|--------|-------|
| Final Loss | 0.1303 |
| Token Accuracy | 6.36% |
| Perplexity | ~4186 |
| SNN Spike Rate | **30.3%** (perfectly calibrated) |
| HSAQ Sparsity | 0.029 (target 0.048, adaptive) |
| Optimization | NaN-free, no gradient issues |

## Files

| File | Size | Description |
|------|------|-------------|
| `materia-v4.basemateria` | 5.9 GB | Full model weights (native MATERIA format, pickle) |
| `checkpoint_epoch1.pt` | 11.2 GB | Epoch 1 checkpoint (PyTorch, full optimizer state) |
| `checkpoint_epoch2.pt` | 11.2 GB | Epoch 2 checkpoint (PyTorch, full optimizer state) |
| `materia-v4.Q4_0.gguf` | 1.16 GB | GGUF format (Q4_0 quantization, compatible with llama.cpp) |
| `config_1B.yaml` | 839 B | Training configuration |

## Usage

### Loading with PyTorch

```python
import torch
import pickle

# Load .basemateria
with open('materia-v4.basemateria', 'rb') as f:
    data = pickle.load(f)

# Access weights
state_dict = data['state_dict']
config = data['config']
print(f"Model: {config['version']}, dim={config['dim']}")

# Import model architecture
from models.materia_v4 import MateriaV4

model = MateriaV4(
    vocab_size=config['vocab_size'],
    dim=config['dim'],
    n_layers=24, n_heads=24, n_kv=6,
    latent_dim=config['latent_dim'],
)
model.load_state_dict(state_dict, strict=False)
model.eval()
```

### Running with llama.cpp (GGUF)

```bash
# Build llama.cpp with MATERIA support
# Then run:
./main -m materia-v4.Q4_0.gguf -p "The meaning of life is" -n 50
```

## Training Your Own

```bash
# Clone the repo
git clone https://github.com/MethodWhite/materia-core.git
cd materia-core

# Install dependencies
pip install torch numpy matplotlib pyyaml

# Train
python scripts/train_v4_enhanced.py --config configs/V4_3B_MoE.yaml
```

## MoE Extension

The repository includes a Mixture-of-Experts implementation (8 experts, top-2) for the next generation:

- **Config**: `configs/V4_3B_MoE.yaml`
- **3B MoE** behaves like ~7B dense with only 25% parameter activation per token
- Designed for RTX 6000 Ada 48GB (fits with BS=16, seq_len=256)

## Citation

```bibtex
@software{materia_v4_2026,
  author = {MethodWhite},
  title = {MATERIA V4: Multi-Analytical Toroidal Engine for Recursive Intelligent Analysis},
  year = {2026},
  url = {https://huggingface.co/MethodWhite/materia-v4-1b-bpe}
}
```

## License

MIT
