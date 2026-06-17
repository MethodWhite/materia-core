# Plan de Escalado M.A.T.E.R.I.A. V3 — 250M → 1B Parámetros

## Hardware Disponible
- **GPU**: RTX 3050 Laptop (4GB VRAM, CUDA 12.4)
- **RAM**: ~32GB sistema
- **CPU**: Multi-core

## Fase 1: 250M parámetros (RTX 3050 4GB)

### Técnicas de Optimización
| Técnica | Ahorro VRAM | Implementación |
|---------|-------------|----------------|
| FP16 (AMP) | 50% | `torch.cuda.amp` |
| Gradient Checkpointing | 60%+ | `torch.utils.checkpoint` |
| CPU Offloading | Variable | Optimizer states en CPU |
| Activation Compression | ~30% | `checkpoint` con `keep_rng_state=False` |

### Arquitectura 250M
| Componente | Configuración |
|------------|---------------|
| Vocab size | 32,768 (BPE tokenizer) |
| Hidden dim | 1,024 |
| Layers | 24 |
| Attention heads | 16 |
| KV heads | 8 |
| FFN intermediate | 4,096 |
| JEPA dim | 512 |
| Synapsis slots | 2,048 |
| Max sequence | 2,048 |
| **Total params** | **~252M** |

### Requerimientos VRAM (estimados)
| Modo | VRAM | Técnicas |
|------|------|----------|
| FP32 sin optimización | ~3.5GB + 3.5GB grad + 7GB opt = 14GB | ❌ No cabe |
| FP16 + checkpointing | ~1.8GB + ~0.5GB temp = 2.5GB | ✅ Cabe |
| FP16 + checkpointing + CPU offload | ~1.5GB | ✅ Sobra |

### Datasets
| Dataset | Tamaño | Propósito |
|---------|--------|-----------|
| The Stack (Python) | ~50GB | Código |
| CodeAlpaca 20K | ~5MB | Instrucciones código |
| OpenWebMath | ~15GB | Razonamiento matemático |
| C4 EN | ~300GB (subset ~10GB) | Lenguaje general |
| GSM8K | ~7MB | Problemas matemáticos |

### Evaluación
| Benchmark | Métrica | Objetivo |
|-----------|---------|----------|
| HumanEval | pass@1 | >30% |
| MBPP | pass@1 | >40% |
| GSM8K | accuracy | >50% |
| MMLU | accuracy | >40% |

### Timeline
- Setup y preparación de datos: 2 días
- Entrenamiento en RTX 3050 (~2 semanas)
- Evaluación y fine-tuning: 3 días
- **Total Fase 1: ~3 semanas**

## Fase 2: 1B parámetros (Cloud GPU)

### Hardware Requerido
- **GPU**: 1x A100 80GB o 4x RTX 4090 24GB
- **Costo cloud**: ~$2-4/hora (A100), ~$1-2/hora (RTX 4090)
- **Tiempo estimado**: ~2-4 semanas

### Opciones Cloud
| Proveedor | GPU | Costo/hora | Disponibilidad |
|-----------|-----|------------|----------------|
| Lambda Labs | 1x A100 80GB | $1.10 | ✅ |
| RunPod | 1x RTX 4090 | $0.44 | ✅ |
| Vast.ai | 1x A100 | $0.80-1.50 | ✅ |
| Google Colab Pro+ | 1x A100 | $50/mes | Limitado |

### Arquitectura 1B
| Componente | Configuración |
|------------|---------------|
| Vocab size | 32,768 (BPE) |
| Hidden dim | 2,048 |
| Layers | 24 |
| Attention heads | 32 |
| KV heads | 8 (GQA) |
| FFN intermediate | 8,192 |
| JEPA dim | 1,024 |
| Synapsis slots | 4,096 |
| Max sequence | 4,096 |
| **Total params** | **~1.02B** |

### Entrenamiento Distribuido
- **Framework**: PyTorch FSDP + DeepSpeed ZeRO-3
- **Paralelismo**: Data Parallel (4x GPU) + Tensor Parallel (opcional)
- **Mixed Precision**: BF16 (A100) o FP16 (RTX 4090)
- **Batch size global**: 256-512
- **Learning rate**: 3e-4 (cosine schedule)

### Métricas Objetivo 1B
| Benchmark | Meta Mínima | Meta Ideal |
|-----------|-------------|------------|
| HumanEval pass@1 | >35% | >50% |
| GSM8K | >55% | >70% |
| MMLU | >45% | >60% |
| Habilidad código | Sí | Generación funciones complejas |
| Razonamiento multi-step | Sí | Cadenas de 5+ pasos |

## Resumen Ejecutivo

```
Fase 1 (ahora):   250M params → RTX 3050 4GB → 3 semanas → HumanEval >30%
                         ↓
Fase 2 (siguiente): 1B params → Cloud GPU A100 → 4 semanas → HumanEval >50%
                         ↓
Fase 3 (futuro):   7B+ params → Multi-GPU cluster → Competitivo con opensource
```

## Costos Estimados
| Fase | Hardware | Costo |
|------|----------|-------|
| Fase 1 | RTX 3050 (ya tenemos) | $0 |
| Fase 2 | Cloud 1x A100 80GB | ~$300-700 |
| Fase 3 | Cloud 4x A100 | ~$2000-5000 |
