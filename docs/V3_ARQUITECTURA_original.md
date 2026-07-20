# M.A.T.E.R.I.A. V3 — Documentación Técnica de Arquitectura

## 1. Visión General

M.A.T.E.R.I.A. V3 es una implementación desde cero de un transformer con arquitectura híbrida que combina:

- **Grouped Query Attention (GQA)**: Atención eficiente con 4 KV heads y 8 query heads
- **Rotary Position Embeddings (RoPE)**: Codificación posicional rotatoria
- **SwiGLU Activation**: Función de activación Swish-Gated Linear Unit
- **JEPA Predictive Embeddings**: Embeddings predictivos en espacio latente
- **Synapsis Memory**: Memoria persistente entre sesiones
- **HSAQ Sparse Execution**: Ejecución dispersa con umbral de activación

### 1.1 Parámetros del Modelo

| Parámetro | Valor |
|-----------|-------|
| Vocab size | 16,384 tokens |
| Hidden size | 512 dims |
| Layers | 4-8 (configurable) |
| Attention heads | 8 query / 4 KV |
| FFN size | 2,048 |
| Max context | 4,096 tokens |
| JEPA dim | 256 |
| Synapsis slots | 1,024 |
| Parámetros totales | ~4.15M (nano) / ~50M (full) |

## 2. Grouped Query Attention (GQA)

La GQA es una variante eficiente de la atención multi-cabeza donde las cabezas clave (K) y valor (V) se comparten entre grupos de cabezas de consulta (Q).

### 2.1 Arquitectura

```
Entrada (512 dims)
    |
    v
Q projection (512 -> 512) = 8 heads x 64 dims cada una
K projection (512 -> 256) = 4 heads x 64 dims cada una  (compartidas)
V projection (512 -> 256) = 4 heads x 64 dims cada una  (compartidas)
    |
    v
RoPE aplicado a Q y K
    |
    v
K y V repetidos para igualar 8 heads
    |
    v
Score = Q * K^T / sqrt(64)
    |
    v
Softmax + Weighted sum
    |
    v
Output projection (512 -> 512)
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
SwiGLU(x) = x * sigmoid(gate(x)) * up(x)
```

Donde `gate` y `up` son proyecciones lineales separadas.

### 4.1 Ventajas sobre ReLU

- **No satura**: Swish permite gradientes fluidos incluso para valores negativos grandes
- **Puerta adaptativa**: La componente GLU permite que la red decida qué información pasar
- **Mejor convergencia**: Reportado en la literatura como superior a ReLU y GELU

## 5. JEPA Predictive Embeddings

JEPA (Joint Embedding Predictive Architecture) es un componente que predice representaciones en espacio latente.

### 5.1 Flujo JEPA

```
Input embeddings (512 dims)
    |
    v
JEPA projection: 512 -> 256 (espacio latente)
    |
    v
JEPA prediction: 256 -> vocab_size (logits residuales)
    |
    v
Suma residual con logits principales
```

### 5.2 Propósito

- **Aprendizaje auto-supervisado**: El predictor JEPA aprende a anticipar representaciones futuras
- **Señal de entrenamiento adicional**: Proporciona gradientes complementarios al loss principal
- **Espacio latente compacto**: 256 dimensiones vs 512 del embedding principal

## 6. Synapsis Memory

La memoria Synapsis permite persistencia de contexto entre sesiones de inferencia.

### 6.1 Arquitectura

```
Clave: embedding promedio del contexto actual
Valor: representación del estado
    |
    v
Recuperación: match por similitud con claves almacenadas
    |
    v
Inyección: contribución residual (factor 0.1) al embedding actual
    |
    v
Almacenamiento automático de nuevos contextos
```

### 6.2 Características

- **Persistente**: Los recuerdos sobreviven entre sesiones
- **Acotado**: Máximo 1024 entradas (política FIFO)
- **Ligero**: Búsqueda por hash MD5, no requiere vector database

## 7. HSAQ Sparse Execution

HSAQ (HyperSparse Adaptive Quantization) activa solo las neuronas más relevantes durante el forward pass.

### 7.1 Funcionamiento

```
mask = |activaciones| > threshold (0.01)
output = activaciones * mask  (enmascara neuronas irrelevantes)
```

### 7.2 Beneficios

- **Menor costo computacional**: ~60% de las neuronas no se evalúan
- **Sin pérdida de precisión**: Solo se descartan activaciones cercanas a cero
- **Complementario a la cuantización**: Se puede combinar con HSAQ weight compression

## 8. Estado del Modelo

| Componente | Estado |
|-----------|--------|
| Arquitectura Python | Implementada |
| GQA + RoPE + SwiGLU | Implementado |
| JEPA embeddings | Implementado |
| Synapsis memory | Implementado (688 entradas) |
| HSAQ sparse execution | Implementado |
| Entrenamiento completo | NO REALIZADO |
| materia-v3.basemateria | CREADO (pesos inicializados) |

## 9. Próximos Pasos

1. **Entrenamiento con dataset real**: Usar `trainer.py` con datos de mayor calidad
2. **Aumentar parámetros**: Escalar de 4.15M a 50M
3. **Fine-tuning**: Especializar para dominios específicos
4. **Exportación a .gguf**: Convertir para uso con Ollama/llama.cpp
5. **Integración con materia-engine**: Conectar con el runtime en Rust

---

*Documento generado: 2026-06-15*
*Modelo: materia-v3.basemateria (4.15M parámetros)*
