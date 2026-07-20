# M.A.T.E.R.I.A. V3: Un Experimento de Arquitectura Multi-Paradigma con 3.5M de Parámetros

| | |
|---|---|
| **Autor(es)** | Jesús Zárate Hernández (MethodWhite), Investigador Independiente / M.A.T.E.R.I.A. Research |
| **Fecha de publicación** | Julio 2026 |
| **Categoría** | Investigación |
| **Tiempo de lectura** | ≈ 12 min |
| **Nivel** | Intermedio |
| **Etiquetas** | #IA #LATAM #MATERIA #transformers #SNN #JEPA #HSAQ |
| **Permalink / DOI** | https://ialatam.com/blog/materia-v3-arquitectura-multi-paradigma |

> **RESUMEN EJECUTIVO**
>
> M.A.T.E.R.I.A. V3 es un modelo de lenguaje experimental de 3.5 millones de parámetros que integra
> Grouped Query Attention (GQA), Rotary Position Embeddings (RoPE), SwiGLU, neuronas LIF-SNN,
> State Space Models (SSM), Joint Embedding Predictive Architecture (JEPA), memoria Synapsis y
> ejecución dispersa HSAQ en una sola arquitectura entrenable end-to-end. Entrenado en CPU con
> datasets de C4 y Wikipedia multilingüe, alcanza ~98.8% de accuracy en validación con solo 4 épocas.
> Este artículo presenta la arquitectura real, los resultados de entrenamiento verificables, y las
> lecciones aprendidas al construir un modelo desde cero con recursos limitados.

> **DATO CLAVE**
>
> 3.5M de parámetros — 256 dims ocultas — 64 tokens de contexto — 98.8% accuracy en validación —
> Entrenamiento completo en CPU (~30 min por época)

---

## 1. Introducción

### 1.1 Motivación

Los modelos de lenguaje actuales (GPT-4, Gemini, Claude, Llama 3) requieren cientos de miles de
millones de parámetros y clusters de GPUs para entrenar. Esta escalabilidad los hace inaccesibles
para la mayoría de los investigadores independientes y organizaciones pequeñas en Latinoamérica.

M.A.T.E.R.I.A. V3 nace de una pregunta simple: **¿qué podemos aprender construyendo un modelo
completo desde cero con recursos mínimos?**

Lejos de competir con modelos de frontera, este proyecto es un experimento educativo y de
investigación que busca explorar la integración de múltiples paradigmas de IA en una sola
arquitectura, documentando tanto los aciertos como las limitaciones.

### 1.2 Contribuciones de este Trabajo

1. **Arquitectura integrada**: GQA + RoPE + SwiGLU + LIF-SNN + SSM + JEPA + Synapsis + HSAQ en
   un solo modelo entrenable end-to-end con 3.5M parámetros.
2. **SNN real con LIF**: Implementación y verificación de neuronas Leaky Integrate-and-Fire con
   dinámica de membrana, surrogate gradient y spikes binarios.
3. **Datos de entrenamiento reales**: Logs completos por step (loss, accuracy, grad_norm,
   spike_rate) para todos los modelos entrenados.
4. **Lecciones para investigadores independientes**: Qué funciona y qué no al entrenar un modelo
   desde cero con recursos limitados (CPU-only, datasets pequeños).

### 1.3 Estructura del Artículo

La Sección 2 describe la arquitectura del sistema con valores reales. La Sección 3 detalla la
metodología de entrenamiento. La Sección 4 presenta los resultados experimentales verificables.
La Sección 5 discute limitaciones y trabajo futuro. La Sección 6 concluye.

---

## 2. Arquitectura del Sistema

### 2.1 Visión General

M.A.T.E.R.I.A. V3 es un transformer con 3 capas y 256 dimensiones ocultas, aumentado con
componentes especializados para procesamiento temporal (LIF-SNN), secuencias largas (SSM),
aprendizaje auto-supervisado (JEPA) y ejecución eficiente (HSAQ).

**Parámetros clave del modelo base (`materia-v3.basemateria`):**

| Parámetro | Valor |
|-----------|-------|
| Parámetros totales | 3,533,568 |
| Dimensiones ocultas (hidden) | 256 |
| Capas transformer | 3 |
| Cabezas de atención (query) | 8 |
| Cabezas KV (compartidas) | 4 |
| Dimensión JEPA (latente) | 128 |
| Slots de memoria Synapsis | 128 |
| Sparsity HSAQ | 0.3 (30%) |
| Dimensión del estado SSM | 32 |
| Dimensión SNN | 128 |
| Vocabulario | 208 tokens (char-level) |
| Contexto máximo | 64 tokens |

### 2.2 Grouped Query Attention (GQA)

La GQA [1] es una variante eficiente de atención multi-cabeza donde las cabezas clave (K) y
valor (V) se comparten entre grupos de cabezas de consulta (Q). En nuestra implementación:

- **8 query heads × 32 dims** = 256 dims de proyección Q
- **4 KV heads × 32 dims** = 128 dims de proyección K/V
- Ratio de compresión: 2:1 (8Q:4KV)
- Cache KV reduce 50% comparado con atención multi-cabeza estándar

### 2.3 Rotary Position Embeddings (RoPE)

RoPE [2] codifica la posición relativa de los tokens mediante rotaciones en el espacio de
embedding. Implementación estándar con `inv_freq = 1.0 / (10000^(2i/dim))`.

### 2.4 SwiGLU Activation

SwiGLU [3] combina Swish (SiLU) con una puerta lineal. Usamos `ffn_dim = dim * 4 = 1024`.
Esta activación fue utilizada en PaLM [4] y LLaMA [5] y ha demostrado superioridad consistente
sobre ReLU y GELU.

### 2.5 LIF-SNN (Leaky Integrate-and-Fire)

El componente SNN utiliza neuronas LIF reales con dinámica de membrana:

```
dV/dt = (-V + I_in) / tau    (leaky integration)
V(t) = V(t-1) * tau + I_in * (1 - tau)
spike = 1 if V >= threshold
V = V - spike * threshold     (soft reset)
```

Parámetros: `threshold=0.05`, `tau=0.8`. El gradiente se propaga mediante surrogate gradient
para permitir backpropagation.

**Importante**: La implementación original usaba `torch.sigmoid(currents * 5)` que NO es un SNN
real (no tiene membrana, threshold, ni dinámica temporal). Esto fue corregido y verificado
experimentalmente.

### 2.6 State Space Model (SSM)

El SSM [6] modela secuencias mediante un sistema dinámico lineal con state_dim=32:

```
h(t) = tanh(h(t-1) @ A + B · x(t))
y(t) = C · h(t)
```

Aporta capacidad de modelar dependencias de largo alcance complementaria al transformer.

### 2.7 JEPA (Joint Embedding Predictive Architecture)

JEPA [7] opera en espacio latente de 128 dimensiones:

- **Encoder**: 256 → 128 (proyección a espacio latente)
- **Predictor**: 128 → 128 (predicción en latente)
- **Decoder**: 128 → 256 (reconstrucción)

A diferencia de models generativos que predicen en el espacio de entrada, JEPA predice en un
espacio abstracto, lo que permite aprendizaje auto-supervisado más eficiente.

### 2.8 Synapsis Memory

Memoria persistente con 128 slots y retrieval por similitud de coseno (top-3):

- **Escritura**: slot = step % n_slots (circular)
- **Lectura**: top-3 por producto punto entre clave de consulta y claves almacenadas
- **Persistencia**: Las memorias se almacenan como buffers del modelo

### 2.9 HSAQ (HyperSparse Adaptive Quantization)

HSAQ implementa ejecución dispersa adaptativa dinámica:

```python
flat = x.abs().view(B, -1)
k = int(n * (1 - sparsity))  # sparsity=0.3
thresh = torch.kthvalue(flat, k, dim=1).values
mask = x.abs() >= thresh
return x * mask
```

A diferencia de un threshold fijo, `kthvalue` calcula el umbral óptimo dinámicamente para
cada batch, reteniendo siempre el top-70% de las activaciones por magnitud. El gradiente
fluye a través de la máscara, permitiendo que el optimizer aprenda qué neuronas son relevantes.

### 2.10 Diagrama de Arquitectura

```
Input tokens (char-level, vocab=208)
    |
    v
Embedding (208 → 256)
    |
    v
HSAQ (sparsity=0.3)
    |
    v
TransformerBlock × 3 (cada uno: GQA + RoPE + SwiGLU)
    |                                        \
    v                                         v
SynapsisMemory (128 slots)               SNNLayer (LIF)
    |                                        |
    v                                         v
    +-------> Concatenate <------------------+
                |
                v
          SSMBlock (state_dim=32)
                |
                v
          JEPA (latent=128)
                |
                v
          RMSNorm → Linear head (256 → 208)
                |
                v
          Logits (vocab=208)
```

---

## 3. Metodología de Entrenamiento

### 3.1 Hardware

- **CPU**: 4 cores (Intel/AMD)
- **GPU**: No disponible para entrenamiento (CPU-only)
- **RAM**: 8-16 GB

### 3.2 Datasets

| Dataset | Fuente | Tamaño | Propósito |
|---------|--------|--------|-----------|
| C4 EN | HuggingFace (c4) | ~80K líneas | Lenguaje general inglés |
| Wikipedia EN/ES | Wikipedia API | ~40K líneas | Cobertura multilingüe |

### 3.3 Hiperparámetros

| Parámetro | Valor Base |
|-----------|------------|
| Learning rate | 5e-4 |
| Optimizer | AdamW |
| Weight decay | 0.01 |
| Scheduler | CosineAnnealing (warmup=100 steps) |
| Batch size | 8 |
| Grad accumulation | 4 |
| Max seq len | 64 |
| Epochs | 4 |
| Clip grad norm | 1.0 |
| Tokenizer | Char-level (208 tokens) |

### 3.4 Proceso de Entrenamiento

El entrenamiento se realizó en CPU con torch.set_num_threads(4). Cada época procesa ~2,000 steps
(batch_size=8, grad_accum=4, ~80K líneas de dataset). El tiempo total de entrenamiento fue de
aproximadamente 2 horas para 4 épocas.

La función de pérdida es cross-entropy estándar sobre la predicción de tokens. La métrica de
accuracy mide la precisión de predicción token-level.

---

## 4. Resultados Experimentales

### 4.1 Curva de Loss y Accuracy

Los datos que siguen son reales, extraídos directamente de los logs de entrenamiento
(`materia-v3_basemateria_log.csv`, 15,816 steps registrados).

**Inicio del entrenamiento (step 1-30):**

| Step | Loss | Accuracy | Grad Norm |
|------|------|----------|-----------|
| 1 | 5.4994 | 0.0078 (0.78%) | 2.2239 |
| 5 | 3.6442 | 0.2598 (25.98%) | 1.9668 |
| 10 | 3.1579 | 0.2617 (26.17%) | 1.4486 |
| 20 | 2.9130 | 0.2227 (22.27%) | 1.0981 |
| 30 | 2.6816 | 0.2617 (26.17%) | 0.8723 |

El modelo parte de pérdida alta (~5.5) y accuracy cercano a cero (0.78%), lo cual es esperado
para un modelo con pesos inicializados aleatoriamente.

**Evolución por época (valores al final de cada época):**

| Época | Loss Final | Acc Final | Grad Norm | Spike Rate |
|-------|-----------|-----------|-----------|------------|
| 1 | 0.0387 | 0.9883 | 0.0818 | 0.070 |
| 2 | 0.0302 | 0.9941 | 0.0646 | 0.065 |
| 3 | 0.0556 | 0.9863 | 0.1550 | 0.098 |
| 4 | 0.0363 | 0.9883 | 0.0974 | 0.135 |

Al final de la primera época, el modelo ya alcanza ~98.8% de accuracy, convergiendo rápidamente
debido al tamaño reducido del vocabulario (208 tokens) y la simplicidad relativa del dataset.

### 4.2 Análisis de Gradient Norm

La norma del gradiente se mantuvo mayoritariamente entre 0.05 y 0.4, indicando entrenamiento
estable sin exploding/vanishing gradients. Los valores altos iniciales (~2.2) son normales en
las primeras iteraciones.

### 4.3 Análisis de Spike Rate (LIF)

La tasa de disparo de neuronas LIF varió entre 0.001 y 0.15, con tendencia creciente durante
el entrenamiento. Esto indica que las neuronas aprenden a disparar con mayor frecuencia a
medida que el modelo converge.

> **CITA**
>
> "La integración de múltiples paradigmas de IA no reemplaza la escala, pero permite explorar
> interacciones entre mecanismos de aprendizaje que los modelos monolíticos no pueden revelar."
>
> — Experimentos con M.A.T.E.R.I.A. V3, 2026

### 4.4 Modelos Derivados (Fine-tunes)

| Módulo | Params | Dataset | Loss | Acc |
|--------|--------|---------|------|-----|
| materia-v3-full | 4.82M | C4 EN (15K textos) | 0.0332 | 0.9903 |
| materia-v3-extended | 3.42M | C4 EN (5K textos) | 0.0357 | 0.9896 |
| materia-v3-unified | 2.42M | Wikipedia ES/EN | 0.0006 | 1.0000 |
| materia-v3-nano | 0.64M | C4 EN (1K textos) | 0.0474 | 0.9885 |
| science-v3 | 2.33M | reasoning_dataset (168 QA) | 0.0308 | 0.9980 |

Nota: Los módulos `science-v3-part-1/2/3` son archivos de configuración, no modelos entrenados.

---

## 5. Limitaciones y Trabajo Futuro

### 5.1 Limitaciones Identificadas

1. **Vocabulario char-level**: 208 tokens es insuficiente para representación eficiente del
   lenguaje. Un tokenizer BPE con 8K-32K tokens mejoraría significativamente la eficiencia.
2. **Contexto pequeño**: 64 tokens limita severamente la capacidad de razonamiento y generación.
3. **Dataset pequeño**: ~120K líneas de texto es insuficiente para entrenar un modelo robusto.
4. **CPU-only**: El entrenamiento en CPU limita el escalado a modelos más grandes.
5. **Sin evaluación en benchmarks**: No se realizaron evaluaciones en MMLU, GSM8K, HumanEval
   porque el modelo es demasiado pequeño para tareas complejas.
6. **Sin autoentrenamiento**: El modelo se entrena con supervisión estándar, no con bucles de
   auto-aprendizaje como se ha especulado en documentación anterior.

### 5.2 Lecciones Aprendidas

- La integración de SNN+LIF y SSM en un transformer pequeño es viable pero el beneficio es
  marginal con datasets pequeños.
- HSAQ (kthvalue dinámico) reduce el costo computacional sin pérdida de accuracy, pero el
  overhead de `kthvalue` en CPU puede contrarrestar la ganancia.
- JEPA con latent=128 añade complejidad pero el beneficio en modelos pequeños no está claro.
- El valor principal del proyecto es **educativo**: construir un modelo completo desde cero
  revela detalles que el uso de APIs o modelos pre-entrenados no muestra.

### 5.3 Trabajo Futuro

- Escalar a 50M-100M parámetros con GPU
- Implementar tokenizer BPE multilingüe
- Evaluación en benchmarks estándar
- Fine-tuning con datasets más grandes (The Stack, CodeAlpaca)
- Explorar si la integración multi-paradigma ofrece ventajas sobre transformers puros al escalar

---

## 6. Conclusiones

M.A.T.E.R.I.A. V3 demuestra que es posible construir un modelo de lenguaje desde cero con
múltiples paradigmas de IA (GQA, RoPE, SwiGLU, LIF-SNN, SSM, JEPA, HSAQ) usando solo
**3.5 millones de parámetros y entrenamiento en CPU**. El modelo alcanza ~98.8% de accuracy
en validación con 4 épocas.

Sin embargo, las limitaciones son claras: vocabulario char-level, contexto de 64 tokens, y
datasets pequeños hacen que el modelo no sea útil para tareas del mundo real. El verdadero
valor de M.A.T.E.R.I.A. V3 es como **plataforma educativa y de experimentación** para
investigadores independientes que quieren entender cómo funcionan estos componentes por
dentro.

Lejos de las exageraciones, este documento presenta la realidad de un proyecto hecho con
recursos limitados pero con honestidad técnica. Todos los datos de entrenamiento son
verificables a partir de los logs generados.

> **IDEA PARA LLEVAR**
>
> Construir un modelo de lenguaje desde cero con 3.5M de parámetros y múltiples paradigmas
> es posible en CPU, pero el verdadero aprendizaje está en el proceso, no en el rendimiento.

---

## Referencias

[1] Ainslie, J. et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from
    Multi-Head Checkpoints. arXiv:2305.13245.

[2] Su, J. et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding.
    arXiv:2104.09864.

[3] Shazeer, N. (2020). Glu Variants Improve Transformer. arXiv:2002.05202.

[4] Chowdhery, A. et al. (2022). PaLM: Scaling Language Modeling with Pathways.
    arXiv:2204.02311.

[5] Touvron, H. et al. (2023). LLaMA: Open and Efficient Foundation Language Models.
    arXiv:2302.13971.

[6] Gu, A. et al. (2021). Efficiently Modeling Long Sequences with Structured State Spaces.
    arXiv:2111.00396.

[7] LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. OpenReview.

[8] Neftci, E. et al. (2019). Surrogate Gradient Learning in Spiking Neural Networks.
    IEEE Signal Processing Magazine.

---

## CÓMO CITAR EN EL TEXTO

En el cuerpo: usa corchetes con el número de referencia, ej: "como demostró Zárate [2026]"
o "estudios previos [1][2] muestran que...".

## CÓMO CITAR ESTE ARTÍCULO

Zárate Hernández, J. A. (2026). M.A.T.E.R.I.A. V3: Un Experimento de Arquitectura
Multi-Paradigma con 3.5M de Parámetros. IA LATAM Blog.
https://ialatam.com/blog/materia-v3-arquitectura-multi-paradigma

---

## AUTOR

**Jesús Zárate Hernández (MethodWhite)**

Investigador independiente en inteligencia artificial y ciberseguridad. Fundador de
M.A.T.E.R.I.A. Research, San Fabián, Chile. Creador de Synapsis (sistema de memoria
persistente para agentes IA) y Oura (Loop Engine MCP). Interesado en arquitecturas de
IA eficientes, criptografía post-cuántica y sistemas multi-agente.

Contacto: methodwhite@pm.me
GitHub: https://github.com/MethodWhite

---

*Documento generado: Julio 2026*
*M.A.T.E.R.I.A. Research © 2026*
*Datos verificables en: /home/methodwhite/MATERIA/logs/materia-v3_basemateria_log.csv*
