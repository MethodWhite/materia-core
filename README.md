# M.A.T.E.R.I.A. V4

Multi-Analytical Toroidal Engine for Recursive Intelligent Analysis

Arquitectura JEPA-First + SCA: JEPA + GQA + RoPE (NTK) + SwiGLU + LIF-SNN + SSM + Flash Attention 2 + Synapsis + HSAQ (INT8/INT4)

## Estado Actual (Julio 2026)

| Componente | Detalle |
|-----------|---------|
| Modelo base | `materia-v4` (140.9M params con BPE 32K) |
| Arquitectura | JEPA-First + SCA (K=2.781042) |
| Entrenamiento | Epoch 8/10, val perplexity=1.06, val acc=99.8% |
| Tokenizer | BPE 32K tokens (SentencePiece multilingüe) |
| Contexto | 256 tokens (RoPE NTK scaling 2x) |
| Hidden dim | 768 (config 142M) |
| GPU | RTX 3050 4GB VRAM |

## Estructura

```
MATERIA/
├── models/
│   ├── materia_v4.py     ← Modelo V4 (JEPA-First + SCA)
│   └── core/
│       ├── blocks.py     ← GQA, RoPE NTK, FlashGQA, SwiGLU
│       ├── hsaq.py       ← HSAQ con weight quant INT8/INT4
│       ├── synapsis.py   ← Memoria persistente
│       ├── neuro.py      ← LIF-SNN, SSM
│       └── jepa.py       ← JEPA + SCA Predictor
├── configs/              ← Configs YAML (4.7M a 142M)
│   ├── V4_20M_BPE.yaml   ← 35.6M params BPE
│   └── V4_142M_BPE.yaml  ← 140.9M params BPE (recomendado)
├── scripts/
│   ├── train_v4_enhanced.py ← Entrenamiento mejorado
│   ├── inference_v4.py      ← Inferencia desde checkpoint
│   ├── export_hsaq.py       ← Export ONNX/TorchScript
│   ├── calibrate_hsaq.py    ← AWQ calibration
│   └── eval_benchmarks.py   ← WikiText-2 / HellaSwag
├── docs/
│   ├── PAPER_CIENTIFICO_MATERIA_V4.md  ← Paper V4
│   └── HSAQ_DOCUMENTACION_DETALLADA.md
├── data/                 ← Datasets (gitignored)
├── logs/                 ← Logs (gitignored)
└── outputs/              ← Checkpoints (gitignored)
```

## Entrenar

```bash
# 142M BPE (recomendado)
python scripts/train_v4_enhanced.py --config configs/V4_142M_BPE.yaml --no-synapsis

# 20M BPE (rápido)
python scripts/train_v4_enhanced.py --config configs/V4_20M_BPE.yaml --no-synapsis
```

## Requisitos

`pip install torch sentencepiece pyyaml numpy matplotlib`
