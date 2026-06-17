# CATÁLOGO DE MODELOS .basemateria y MÓDULOS .materia

**M.A.T.E.R.I.A. — Registro completo de modelos base y módulos de expansión**

---

## Arquitectura del Sistema

```
.basemateria (BASE)          .materia (MÓDULOS EXPANSIÓN)
       │                              │
       │  materia-v3-full.materia  ◄──┤
       │  materia-v3-extended.materia ◄┤
       │  materia-v3-unified.materia  ◄┤
       │  materia-v3-nano.materia     ◄┤
       │  science-v3.materia         ◄──┤
       │  science-v3-part-1.materia   ◄┤
       │  science-v3-part-2.materia   ◄┤
       │  science-v3-part-3.materia   ◄┤
       v                              v
  materia-v3.basemateria     (múltiples .materia cargables)
```

- **.basemateria**: Modelo base del sistema (el "cerebro"). Contiene los pesos entrenados del núcleo.
- **.materia**: Módulos de expansión que cargan conocimiento adicional sobre el .basemateria.

---

## Modelo Base (único .basemateria activo)

| Archivo | Versión | Params | Estado | Último entrenamiento |
|---------|---------|--------|--------|---------------------|
| `materia-v3.basemateria` | V3 | 678,784 | ✅ **ENTRENADO** | 2026-06-16 |

### Detalles del BaseModel:
- **Arquitectura**: GQA + RoPE + SwiGLU + JEPA + Synapsis + HSAQ
- **Layers**: 2 | **Hidden**: 128 | **Heads**: 4 | **KV**: 2
- **Vocab**: 181 tokens (char-level)
- **Contexto max**: 64 tokens
- **Dataset**: C4 EN (3,000 textos, ~150K chars)
- **Epochs**: 5 | **Loss final**: 0.0344
- **Weight module**: materia-v3.materia
- **Backend**: Ollama (`materia-v3:latest`)

---

## Módulos .materia (Expansión del BaseModel)

| # | Archivo | Params | Propósito | Estado |
|---|---------|--------|-----------|--------|
| 1 | `materia-v3-full.materia` | 8.1M | Expansión completa con 4 capas | ✅ Entrenado |
| 2 | `materia-v3-extended.materia` | 3.2M | Extendido LLM+SNN+SSM+JEPA | ✅ Entrenado |
| 3 | `materia-v3-unified.materia` | 2.2M | Unificado multi-arquitectura | ✅ Entrenado |
| 4 | `materia-v3-nano.materia` | 4.1M | Ligero para inferencia rápida | ✅ Pre-entrenado |
| 5 | `science-v3.materia` | ~1M | Conocimiento científico general | 📦 Disponible |
| 6 | `science-v3-part-1.materia` | ~350K | Física, Química, Matemáticas | 📦 Disponible |
| 7 | `science-v3-part-2.materia` | ~350K | Biología, Medicina, Neurociencia | 📦 Disponible |
| 8 | `science-v3-part-3.materia` | ~350K | Ingeniería, Computación, Tecnología | 📦 Disponible |

### Carga de módulos:
```bash
# El .basemateria carga módulos .materia automáticamente
materia-v3.basemateria
  +-- materia-v3-full.materia      # capas extra
  +-- science-v3.materia           # conocimiento científico
  +-- science-v3-part-1.materia    # ciencias exactas
```

---

## Modelos Legacy (.basemateria antiguos, ahora .materia)

Los siguientes archivos fueron convertidos de `.basemateria` a `.materia`:

| Archivo antiguo | Archivo nuevo | Estado |
|-----------------|---------------|--------|
| materia-v3-full.basemateria | materia-v3-full.materia | ✅ Convertido |
| materia-v3-unified.basemateria | materia-v3-unified.materia | ✅ Convertido |
| materia-v3-extended.basemateria | materia-v3-extended.materia | ✅ Convertido |

Versiones legacy conservadas con prefijo `legacy-` por compatibilidad.

---

## Modelos Perdidos (Requieren Re-creación)

| Modelo | Versión | Estado |
|--------|---------|--------|
| materia-v1.basemateria | V1 | ❌ PERDIDO |
| materia-v2.basemateria | V2 | ❌ PERDIDO |

---

**Total respaldados: 1 .basemateria + 8 .materia**
**Total perdidos: 2 modelos**
