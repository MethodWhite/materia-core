# M.A.T.E.R.I.A. V3 — Documentación Técnica de Arquitectura

## 1. Visión General

M.A.T.E.R.I.A. V3 es una implementación desde cero de un transformer con arquitectura híbrida que combina:

- **Grouped Query Attention (GQA)**: Atención eficiente con 4 KV heads y 8 query heads
- **Rotary Position Embeddings (RoPE)**: Codificación posicional rotatoria
- **SwiGLU Activation**: Función de activación Swish-Gated Linear Unit
- **LIF-SNN**: Neuronas Leaky Integrate-and-Fire con dinámica de membrana real
- **SSM (State Space Model)**: Modelo de espacio de estados para secuencias largas
- **JEPA Predictive Embeddings**: Embeddings predictivos en espacio latente
- **Synapsis Memory**: Memoria persistente con 128 slots y top-3 retrieval
- **HSAQ Sparse Execution**: Ejecución dispersa adaptativa con umbral dinámico

### 1.1 Parámetros del Modelo Base (`materia-v3.basemateria`)

| Parámetro | Valor |
|-----------|-------|
| Vocab size | 208 tokens (char-level) |
| Hidden size | 256 dims |
| Layers | 3 |
| Attention heads | 8 query / 4 KV |
| FFN size | 1,024 |
| Max context | 64 tokens |
| JEPA latent dim | 128 |
| Synapsis slots | 128 |
| SSM state dim | 32 |
| SNN dim | 128 |
| HSAQ sparsity | 0.3 |
| Parámetros totales | 3,533,568 |

## 2. Grouped Query Attention (GQA)

La GQA es una variante eficiente de la atención multi-cabeza donde las cabezas clave (K) y valor (V) se comparten entre grupos de cabezas de consulta (Q).

### 2.1 Arquitectura

```
Entrada (256 dims)
    |
    v
Q projection (256 -> 256) = 8 heads x 32 dims cada una
K projection (256 -> 128) = 4 heads x 32 dims cada una  (compartidas)
V projection (256 -> 128) = 4 heads x 32 dims cada una  (compartidas)
    |
    v
RoPE aplicado a Q y K
    |
    v
K y V repetidos (repeat_interleave) para igualar 8 heads
    |
    v
Score = Q * K^T / sqrt(32)
    |
    v
Softmax + Weighted sum
    |
    v
Output projection (256 -> 256)
```

### 2.2 Ventajas

- **Menor memoria en inferencia**: Almacenar KV cache para 4 heads en lugar de 8
- **Rendimiento similar**: La repetición de K/V mantiene la calidad de atención
- **Factor de compresión**: 2x en memoria de KV cache

## 3. Rotary Position Embeddings (RoPE)

RoPE codifica la posición relativa entre tokens mediante una rotación en el espacio de embedding.

### 3.1 Funcionamiento

```
Para cada par de dimensiones (2i, 2i+1):

    pos_embedding[2i]   = sin(pos / 10000^(2i/dim))
    pos_embedding[2i+1] = cos(pos / 10000^(2i/dim))

    x_rotated[2i]   = x[2i] * cos(pos_emb[2i]) - x[2i+1] * sin(pos_emb[2i])
    x_rotated[2i+1] = x[2i] * sin(pos_emb[2i]) + x[2i+1] * cos(pos_emb[2i])
```

### 3.2 Propiedades

- **Traducción relativa**: El producto punto entre dos embeddings rotados depende solo de su diferencia de posición
- **Decaimiento natural**: Las posiciones lejanas tienen menor influencia
- **Sin parámetros adicionales**: No requiere pesos entrenados

## 4. SwiGLU Activation

SwiGLU combina Swish (SiLU) con una puerta lineal:

```
SwiGLU(x) = Swish(gate(x)) * up(x)
```

Donde `gate` y `up` son proyecciones lineales separadas, y `ffn_dim = dim * 4 = 1024`.

### 4.1 Ventajas sobre ReLU

- **No satura**: Swish permite gradientes fluidos incluso para valores negativos grandes
- **Puerta adaptativa**: La componente GLU permite que la red decida qué información pasar
- **Mejor convergencia**: Reportado en la literatura como superior a ReLU y GELU

## 5. LIF-SNN (Leaky Integrate-and-Fire)

El componente SNN utiliza neuronas LIF reales con dinámica de membrana:

```
dV/dt = (-V + I_in) / tau    (leaky integration)
V(t) = V(t-1) * tau + I_in * (1 - tau)
spike = 1 if V >= threshold
V = V - spike * threshold     (soft reset)
```

- `threshold = 0.05`, `tau = 0.8`
- SNN dim = 128 (proyección desde 256)
- Spikes binarios {0,1} con surrogate gradient para backprop

### 5.1 Nota sobre la implementación original

La implementación original usaba `torch.sigmoid(currents * 5)` que NO constituye un SNN real:
- No modela el potencial de membrana V(t)
- No tiene umbral de disparo (threshold)
- No implementa reset post-spike
- No tiene dinámica temporal (constante de tiempo tau)
- Produce valores continuos [0,1], no spikes binarios {0,1}

Esto fue corregido e implementado con LIF real en `core/neuro.py`.

## 6. JEPA Predictive Embeddings

JEPA (Joint Embedding Predictive Architecture) es un componente que predice representaciones en espacio latente.

### 6.1 Flujo JEPA

```
Input embeddings (256 dims)
    |
    v
JEPA encoder: 256 -> 128 (espacio latente)
    |
    v
JEPA predictor: 128 -> 128
    |
    v
JEPA decoder: 128 -> 256 (reconstrucción residual)
```

### 6.2 Propósito

- **Aprendizaje auto-supervisado**: El predictor JEPA aprende a anticipar representaciones futuras
- **Señal de entrenamiento adicional**: Proporciona gradientes complementarios al loss principal
- **Espacio latente compacto**: 128 dimensiones vs 256 del embedding principal

## 7. Synapsis Memory

La memoria Synapsis permite persistencia de contexto entre sesiones de inferencia, con 128 slots y retrieval por similitud de coseno.

### 7.1 Arquitectura

```
Clave: key_proj(embedding actual)
Valor: val_proj(embedding actual)
    |
    v
Escritura: slot = step % 128 (circular)
    |
    v
Lectura: top-3 por producto punto (similitud de coseno)
    |
    v
Inyección: contribución residual (factor 0.3) al embedding actual
```

### 7.2 Características

- **Persistente**: Los recuerdos sobreviven entre sesiones
- **Acotado**: Máximo 128 entradas (política circular FIFO)
- **Ligero**: Retrieval por producto punto, no requiere vector database externa

## 8. HSAQ Sparse Execution

HSAQ (HyperSparse Adaptive Quantization) activa solo las neuronas más relevantes durante el forward pass, usando un umbral dinámico por batch.

### 8.1 Funcionamiento

```python
flat = x.abs().view(B, -1)
k = int(n * (1 - sparsity))  # sparsity=0.3 → top-70%
thresh = torch.kthvalue(flat, k, dim=1).values
mask = x.abs() >= thresh
return x * mask  # enmascara neuronas irrelevantes
```

A diferencia de un threshold fijo (ej. 0.01), `kthvalue` calcula el umbral óptimo dinámicamente, reteniendo siempre el top-70% de las activaciones por magnitud.

### 8.2 Beneficios

- **Menor costo computacional**: ~30% de las neuronas no se evalúan
- **Sin pérdida de precisión**: Solo se descartan activaciones cercanas a cero
- **Adaptativo**: El umbral se ajusta por batch según la distribución de activaciones

## 9. Estado del Modelo Base

| Componente | Estado |
|-----------|--------|
| Arquitectura PyTorch (GQA+RoPE+SwiGLU) | Implementada |
| LIF-SNN (LIF real, no sigmoid) | Implementado y verificado |
| SSM (State Space Model) | Implementado |
| JEPA embeddings (latent=128) | Implementado |
| Synapsis memory (128 slots) | Implementado |
| HSAQ sparse execution (sparsity=0.3) | Implementado |
| Entrenamiento completo (4 épocas) | ✅ REALIZADO |
| Loss final | 0.0363 |
| Accuracy final | 98.83% |

## 10. Próximos Pasos

1. **Mejorar tokenizador**: Implementar BPE con 8K-32K tokens para mejor representación
2. **Aumentar parámetros**: Escalar de 3.5M a 50M+ con GPU
3. **Ampliar dataset**: Usar datasets más grandes (C4 completo, The Stack)
4. **Evaluación en benchmarks**: MMLU, GSM8K, HumanEval
5. **Fine-tuning especializado**: Código, instrucciones, razonamiento matemático

---

*Documento actualizado: Julio 2026*
*Modelo: materia-v3.basemateria (3,533,568 parámetros)*
*Datos de entrenamiento: logs/materia-v3_basemateria_log.csv*
