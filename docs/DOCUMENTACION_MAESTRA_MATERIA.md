# M.A.T.E.R.I.A. — Documentación Maestra V4

**Multi-Analytical Toroidal Engine for Recursive Intelligent Analysis**

---

## 1. Qué es M.A.T.E.R.I.A. V4?

M.A.T.E.R.I.A. V4 es la evolución del sistema, reestructurado bajo un paradigma **JEPA-First**
con **SCA (Spectral Coupling Analysis)** donde la Joint Embedding Predictive Architecture actúa
como eje central del modelado:

- **JEPA-First + SCA** — Predicción en espacio latente con K = √π·e·γ = 2.781042
- **GQA + RoPE (NTK)** — Atención eficiente con escalado posicional rotatorio
- **Flash Attention 2** — Atención O(n) con soporte para contexto largo
- **SwiGLU** — Activación con puerta adaptativa
- **LIF-SNN** — Neuronas de pulsos reales con dinámica de membrana
- **SSM** — Procesamiento de secuencias largas
- **HSAQ v2** — Ejecución dispersa + cuantización de pesos INT8/INT4
- **Synapsis** — Memoria persistente (solo inferencia)

## 2. Estado Actual (Julio 2026)

| Componente | Detalle |
|-----------|---------|
| Modelo | `materia-v4` (140.9M params con BPE 32K) |
| Arquitectura | JEPA-First + SCA + Flash Attention 2 + RoPE NTK |
| Entrenamiento | Epoch 8/10 en GPU RTX 3050 |
| Val perplexity | 1.06 |
| Val accuracy | 99.8% |
| Tokenizer | BPE 32K tokens (SentencePiece v2) |
| Contexto | 256 tokens (NTK scaling 2x) |
| Hidden dim | 768 (config 142M) |
| Checkpoints | epoch 1-7 disponibles |

## 3. Archivos del Proyecto

### Documentación

| Archivo | Propósito |
|---------|-----------|
| `M.A.T.E.R.I.A._V3_CORREGIDO.md` | Documento principal corregido (formato blog IA Latam) |
| `V3_ARQUITECTURA.md` | Especificación técnica actualizada |
| `PAPER_CIENTIFICO_MATERIA_V3_original.md` | Paper original (respaldado) |
| `CATALOGO_BASEMATERIA.md` | Catálogo de modelos |

### Modelos

| Archivo | Tipo | Params |
|---------|------|--------|
| `materia-v3.basemateria` | Base | 3,533,568 |
| `materia-v3-full.materia` | Fine-tune | 4,820,224 |
| `materia-v3-extended.materia` | Fine-tune | 3,418,880 |
| `materia-v3-unified.materia` | Fine-tune | 2,417,920 |
| `materia-v3-nano.materia` | Fine-tune | 639,104 |
| `science-v3.materia` | Fine-tune | 2,334,976 |

Nota: `science-v3-part-1/2/3` son archivos de configuración (584-588 bytes), no modelos entrenados.

---

*Última actualización: Julio 2026*
