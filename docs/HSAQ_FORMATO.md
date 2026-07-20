# HSAQ — HyperSparse Adaptive Quantization

## Definición Formal

HSAQ es un mecanismo de **cuantización de activaciones vía sparsity adaptativa**.
La "cuantización" refiere a que las activaciones se reducen a {0, valor} mediante
una máscara binaria dinámica calculada por batch.

**No es cuantización de pesos.** No es INT8/INT4. No es bitsandbytes.
HSAQ opera exclusivamente sobre **activaciones**, no sobre pesos.

---

## 1. Algoritmo

```
Entrada: x ∈ ℝ^(B×T×D)  (batch de activaciones)
Parámetro: sparsity ∈ [0,1)

  flat = |x|.reshape(B, -1)       # Magnitudes por batch
  n = flat.size(1)                 # Total de neuronas
  k = n * (1 - sparsity)           # Neuronas a mantener (top-K%)
  thresh = kthvalue(flat, k)       # Umbral dinámico por batch
  mask = |x| >= thresh             # Máscara binaria {0, 1}
  return x * mask                  # ~30% de activaciones → 0
```

### 1.1 Propiedades clave

| Propiedad | Descripción |
|-----------|-------------|
| **Adaptativo** | El umbral `kthvalue` se recalcula en cada batch |
| **Por elemento** | Cada neurona se evalúa individualmente contra el umbral |
| **Hardware-agnostic** | Solo usa torch.abs, reshape, kthvalue, multiplicación |
| **Gradiente fluye** | STE implícito: el gradiente pasa por las neuronas activas |
| **Sin estado** | No hay buffers persistentes entre batches |

### 1.2 Parámetros

| Parámetro | Default | Función |
|-----------|---------|---------|
| `sparsity` | 0.3 | Fracción de activaciones a enmascarar (0.3 = 30%) |

Éste es el **único parámetro** de HSAQ. Todo lo demás (weight_bits, AWQ, etc.)
son externos y no forman parte del mecanismo central.

---

## 2. HSAQ como Optimizer

HSAQ **reemplaza a AdamW** como mecanismo de optimización.

### 2.1 Por qué funciona

1. La máscara sparse (kthvalue) selecciona las neuronas más activas por batch
2. El gradiente solo fluye por las neuronas no enmascaradas
3. Esto crea un **regularización adaptativa**: las neuronas irrelevantes no reciben gradiente
4. El umbral dinámico evita la necesidad de momentum/estados de optimizer

### 2.2 Optimizer externo

Se usa SGD Nesterov (momentum=0.9) para actualizar pesos:

```
HSAQ + SGD Nesterov = optimizer completo
  ├── HSAQ: máscara sparse adaptativa (regularización dinámica)
  └── SGD: actualización de pesos con momentum
```

**No se usa AdamW.** SGD con momentum tiene solo 1 estado de optimizer por
parámetro (vs 2 de AdamW), ahorrando 4 bytes por parámetro.

### 2.3 Hyperparámetros recomendados

| Parámetro | Valor | Razón |
|-----------|-------|-------|
| `sparsity` | 0.3 | Balance cómputo/precisión |
| `lr` | 5e-4 | Tasa de aprendizaje |
| `momentum` | 0.9 | Nesterov momentum |
| `weight_decay` | 0.01 | Regularización L2 |
| `clip_grad_norm` | 1.0 | Estabilidad |

---

## 3. Pipeline de Entrenamiento (con HSAQ por capas)

```
1. Embedding → HSAQ (sparsity 30%)
2. Transformer Block 1 → HSAQ (sparsity 30%)   ← umbral propio
3. Transformer Block 2 → HSAQ (sparsity 30%)   ← umbral propio
4. Transformer Block N → HSAQ (sparsity 30%)   ← umbral propio
5. SNN + SSM → JEPA → Head → logits
```

Cada capa tiene su propio umbral dinámico calculado via kthvalue.
Esto permite que:

- Capas tempranas (bajo nivel) tengan patrones de activación distintos
- Capas tardías (alto nivel) se especialicen en representaciones más abstractas
- El modelo aprenda qué información preservar en cada nivel
- Diferentes distribuciones de activación por capa no afecten el umbral global

### 3.1 Forward con HSAQ por capas

```
h = Embedding(x)         # [B, T, dim]
h = HSAQ(h)               # Sparsity post-embedding

for layer in transformer:
    h = layer(h)           # Forward del transformer block
    h = HSAQ(h)            # Sparsity por capa (umbral propio)

h = SNN(h)                 # Neuronas de pulsos
h = SSM(h)                 # State Space Model
h = JEPA(h)                # Espacio latente
h = Head(h)                # Logits finales
```

```
1. Forward pass
   ├── Token Embedding → ℝ^(B×T×D)
   ├── HSAQ sparsity → 30% de activaciones → 0
   ├── Transformer Blocks (GQA + RoPE + SwiGLU)
   ├── LIF-SNN (neuronas de pulsos)
   ├── SSM (State Space Model)
   ├── JEPA Encoder → espacio latente
   └── Head → logits

2. Backward pass
   └── Gradiente fluye solo por activaciones activas (STE nativo)

3. Weight update
   └── SGD Nesterov (momentum 0.9)
```

---

## 4. No es HSAQ (cosas que NO pertenecen)

| Componente | Motivo de exclusión |
|-----------|---------------------|
| INT8/INT4 weight quantization | HSAQ cuantiza activaciones, no pesos |
| bitsandbytes 8-bit Adam | HSAQ reemplaza a AdamW |
| AWQ calibration | Es post-training, no parte de HSAQ |
| GPTQ | Es compresión de pesos, ortogonal a HSAQ |
| BPE tokenizer | HSAQ funciona con char-level |
| Weight tying | Es optimización de arquitectura, no de HSAQ |

---

## 5. Código Mínimo

```python
class HSAQ(nn.Module):
    """HyperSparse Adaptive Quantization — sparsity adaptativa"""
    def __init__(self, sparsity=0.3):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, x):
        flat = x.abs().view(x.size(0), -1)    # Magnitudes
        k = int(flat.size(1) * (1 - self.sparsity))  # Top-K
        thresh = torch.kthvalue(flat, k, dim=1).values  # Umbral dinámico
        thresh = thresh.view(-1, *([1] * (x.dim() - 1)))
        return x * (x.abs() >= thresh)         # Máscara binaria

# Modo de uso en modelo:
# h = self.tok_emb(x)
# h = HSAQ(sparsity=0.3)(h)    ← 30% de activaciones → 0
# h = transformer(h)           ← gradiente solo fluye por neuronas activas
```

---

## 6. HSAQ vs TurboQuant (Google)

| Aspecto | TurboQuant (Google) | HSAQ |
|---------|--------------------|------|
| **Enfoque** | Cuantización fija post-entrenamiento | Sparsity adaptativa dinámica |
| **Granularidad** | Por tensor (pesos) | Por elemento (activaciones) |
| **Umbral** | Fijo (calibrado offline) | Dinámico (kthvalue por batch) |
| **Hardware** | Requiere soporte INT8 | CPU/GPU/TPU (solo kthvalue) |
| **Calibración** | Dataset de calibración offline | Zero overhead (inline) |
| **Adaptabilidad** | Ninguna (mismo esquema siempre) | Por batch (cambia con cada input) |
| **Permite modelos más grandes** | No (solo comprime) | Sí (sparsity = menos recursos) |

### Por qué HSAQ supera a TurboQuant

1. **No malgasta recursos**: solo las neuronas relevantes se activan por batch
2. **Modelos más grandes en hardware limitado**: con sparsity=0.3, un modelo 190M
   corre como si fuera ~133M, permitiendo ejecutar modelos que no cabrían de otra forma
3. **Adaptativo**: el umbral se ajusta a la entrada, no hay configuración fija
4. **Sin calibración**: no necesita datasets externos ni pasos post-entrenamiento
5. **Más eficiente energéticamente**: menos FLOPs = menos consumo

---

## 7. Referencia rápida

| Concepto | Respuesta |
|----------|-----------|
| ¿Qué cuantiza? | **Activaciones** (no pesos) |
| ¿Cómo? | Máscara binaria vía kthvalue |
| ¿Cada cuánto se recalcula? | **Cada batch** (umbral dinámico) |
| ¿Qué reemplaza? | **AdamW** como optimizer |
| ¿Qué optimizer usa? | SGD Nesterov (momentum=0.9) |
| Parámetros | Solo `sparsity` (default 0.3) |
| ¿INT8? | NO |
| ¿bitsandbytes? | NO |
| ¿BPE? | NO |
