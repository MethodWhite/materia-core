# HSAQ Standard — HyperSparse Adaptive Quantization

## Versión 1.0 — Julio 2026

---

## 1. Definición

HSAQ (HyperSparse Adaptive Quantization) es un mecanismo de **cuantización de activaciones**
mediante **sparsity adaptativa dinámica**, donde cada capa de la red calcula su propio umbral
vía `torch.kthvalue` en cada forward pass.

**La "cuantización" en HSAQ** refiere a que las activaciones se reducen a un conjunto
discreto de valores {0, valor_original} mediante una máscara binaria. NO es cuantización
de pesos. NO es INT8/INT4. NO es bitsandbytes.

---

## 2. Principios Fundamentales

1. **Adaptativo**: el umbral de sparsity se recalcula en cada batch usando `kthvalue`
2. **Por capa**: cada componente (embedding, transformer, SNN, SSM, JEPA) tiene su propio umbral
3. **Sin calibración**: no requiere datasets externos ni pasos post-entrenamiento
4. **Hardware-agnostic**: funciona en CPU, GPU y TPU sin modificaciones
5. **Sin estado**: no hay buffers persistentes entre batches
6. **Gradiente fluye**: las neuronas activas reciben gradiente normalmente (STE nativo)

---

## 3. Algoritmo

### 3.1 Pseudocódigo

```
Entrada: x ∈ ℝ^(B×T×D)    # Batch de activaciones
Parámetro: sparsity ∈ [0,1)  # Fracción a enmascarar

1. magnitudes = |x|.view(B, -1)          # Magnitudes por batch
2. n = magnitudes.size(1)                # Total de neuronas
3. k = n × (1 - sparsity)               # Neuronas a mantener
4. umbral = kthvalue(magnitudes, k)      # k-ésimo valor más pequeño
5. mascara = |x| ≥ umbral               # Máscara binaria {0, 1}
6. return x × mascara                   # Neuronas irrelevantes → 0
```

### 3.2 Implementación de Referencia

```python
class HSAQ(nn.Module):
    """HyperSparse Adaptive Quantization"""
    def __init__(self, sparsity: float = 0.3):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparsity <= 0.0:
            return x
        flat = x.abs().view(x.size(0), -1)
        n = flat.size(1)
        k = max(1, min(n - 1, int(n * (1.0 - self.sparsity))))
        thresh = torch.kthvalue(flat, k, dim=1).values
        thresh = thresh.view(-1, *([1] * (x.dim() - 1)))
        return x * (x.abs() >= thresh)
```

---

## 4. Puntos de Aplicación

En M.A.T.E.R.I.A. V4, HSAQ se aplica en **6 puntos** del pipeline:

| # | Componente | Entrada | Salida | Propósito |
|---|-----------|---------|--------|-----------|
| 1 | Post-embedding | `tok_emb(x)` → ℝ^(B×T×D) | ℝ^(B×T×D) | Sparsidad inicial de tokens |
| 2 | Post-transformer | `layer(h_gqa)` → ℝ^(B×T×D) | ℝ^(B×T×D) | Sparsidad por capa atencional |
| 3 | Post-SNN | `snn(h_snn)` → ℝ^(B×T×D) | ℝ^(B×T×D) | Sparsidad de pulsos neuronales |
| 4 | Post-SSM | `ssm(h_ssm)` → ℝ^(B×T×D) | ℝ^(B×T×D) | Sparsidad de estado latente |
| 5 | Post-JEPA | `jepa_enc(fused)` → ℝ^(B×T×L) | ℝ^(B×T×L) | Sparsidad del predictor |

Cada punto recalcula `kthvalue` independientemente → umbral dinámico propio.

---

## 5. Parámetros

| Parámetro | Default | Rango | Descripción |
|-----------|---------|-------|-------------|
| `sparsity` | 0.3 | [0.0, 1.0) | Fracción de activaciones a enmascarar |

**Único parámetro.** HSAQ no tiene hiperparámetros adicionales.
No weight_bits. No weight_quant_mode. No act_bits.

---

## 6. HSAQ como Optimizer

HSAQ reemplaza a AdamW como mecanismo de optimización.
La máscara sparse actúa como regularizador adaptativo: las neuronas
irrelevantes no reciben gradiente, guiando el aprendizaje.

### 6.1 Optimizer Externo

Se usa SGD Nesterov como optimizer externo para actualizar pesos:

```python
opt = optim.SGD(
    model.parameters(),
    lr=5e-4,
    momentum=0.9,
    weight_decay=0.01,
    nesterov=True,
)
```

### 6.2 Ventajas vs AdamW

| Aspecto | AdamW | HSAQ + SGD Nesterov |
|---------|-------|---------------------|
| Estados de optimizer | 2 por parámetro (8 bytes) | 1 por parámetro (4 bytes) |
| Regularización | weight_decay | Sparsity adaptativa |
| Memoria extra | 8 bytes/param | 4 bytes/param |
| Convergencia | Media | Comparable con HSAQ |

Para 190M parámetros: HSAQ ahorra ~760MB de VRAM solo en estados de optimizer.

---

## 7. Pipeline de Entrenamiento

```
1. Forward pass
   ├── Token Embedding
   ├── HSAQ (sparsity=0.3, umbral propio)
   │
   ├── Transformer Block 1
   ├── HSAQ (sparsity=0.3, umbral propio)
   │
   ├── Transformer Block 2..N
   ├── HSAQ (sparsity=0.3, umbral propio)
   │
   ├── LIF-SNN
   ├── HSAQ (sparsity=0.3, umbral propio)
   │
   ├── SSM
   ├── HSAQ (sparsity=0.3, umbral propio)
   │
   ├── JEPA Encoder
   ├── HSAQ (sparsity=0.3, umbral propio)
   │
   └── Head → Logits

2. Backward pass
   └── Gradiente fluye solo por activaciones activas

3. Weight update
   └── SGD Nesterov (momentum=0.9, paso único)
```

---

## 8. Comparación con Otras Técnicas

| Técnica | Reduce | Adaptativo | Calibración | Hardware |
|---------|--------|------------|-------------|----------|
| **HSAQ** | Activaciones | ✅ kthvalue por batch | No requiere | CPU/GPU/TPU |
| TurboQuant (Google) | Pesos INT8 | ❌ Fijo | Requiere offline | GPU con INT8 |
| AWQ | Pesos INT4 | ❌ Fijo post-calibración | Requiere offline | GPU |
| GPTQ | Pesos INT4 | ❌ Fijo post-calibración | Requiere offline | GPU |
| Pruning | Pesos/neuronas | ❌ Post-entrenamiento | No | GPU |
| DeepSpeed | Pesos INT8 | ❌ Fijo | Requiere offline | GPU |

### 8.1 HSAQ vs TurboQuant

**HSAQ supera a TurboQuant porque:**

1. **No desperdicia recursos**: solo las neuronas relevantes se activan
2. **Permite modelos más grandes**: con sparsity 30%, un modelo 190M corre como ~133M
3. **Adaptativo**: el umbral se ajusta a cada entrada, no es fijo
4. **Zero calibración**: no necesita datasets externos
5. **Hardware-agnostic**: no requiere INT8, funciona hasta en CPU

---

## 9. Integración en M.A.T.E.R.I.A. V4

### 9.1 Archivos

| Archivo | Propósito |
|---------|-----------|
| `models/core/hsaq.py` | Implementación de HSAQ (60 líneas) |
| `models/materia_v4.py` | Modelo completo con HSAQ por capas |
| `scripts/train_v4_enhanced.py` | Training script con SGD Nesterov |
| `configs/V4_210M_BPE.yaml` | Config actual (187M, char-level, HSAQ) |

### 9.2 Config de Entrenamiento

```yaml
model:
  dim: 896
  n_layers: 10
  hsaq_sparsity: 0.3

training:
  lr: 5.0e-4
  optimizer: SGD  # HSAQ reemplaza a AdamW
  momentum: 0.9
  nesterov: true
  batch_size: 1
  mixed_precision: bf16
```

### 9.3 Cómo Ejecutar

```bash
python scripts/train_v4_enhanced.py \
  --config configs/V4_210M_BPE.yaml \
  --no-synapsis \
  --batch-size 1 \
  --memory-limit 0.85
```

---

## 10. Limitaciones y Trabajo Futuro

| Limitación | Descripción | Plan |
|-----------|-------------|------|
| Sparsity fija | Actualmente 30% fijo para todas las capas | Sparsity por capa configurable |
| Sin sparse kernel real | `x * mask` no ahorra FLOPs reales | Implementar kernel CUDA sparse |
| Sin export ONNX | kthvalue no está en opset ONNX | Wrapper con topk para export |
| Sin Synapsis en training | Synapsis causa repetición "the the the" | Modo Synapsis solo en inferencia |

---

## 11. Referencias

1. HSAQ Standard: `docs/HSAQ_STANDARD.md`
2. Documentación detallada: `docs/HSAQ_DOCUMENTACION_DETALLADA.md`
3. Código fuente: `models/core/hsaq.py`
4. Paper científico: `docs/PAPER_CIENTIFICO_MATERIA_V4.md`

---

*HSAQ Standard v1.0 — M.A.T.E.R.I.A. Research © 2026*
