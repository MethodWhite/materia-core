# CATÁLOGO DE MODELOS .basemateria y MÓDULOS .materia

**M.A.T.E.R.I.A. — Registro completo de modelos base y módulos de expansión**

---

> ⚠️ **Nota**: Los valores han sido corregidos respecto a documentación anterior.
> Para detalles técnicos actualizados, ver `V3_ARQUITECTURA.md`.

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
       v                              v
  materia-v3.basemateria     (múltiples .materia cargables)
```

- **.basemateria**: Modelo base del sistema con pesos entrenados
- **.materia**: Módulos de fine-tuning que expanden capacidades sobre el base

---

## Modelo Base (único .basemateria activo)

| Archivo | Versión | Params | Estado | Último entrenamiento |
|---------|---------|--------|--------|---------------------|
| `materia-v3.basemateria` | V3 | **3,533,568** | ✅ ENTRENADO | 2026-06-16 |

### Detalles del BaseModel:
- **Arquitectura**: GQA + RoPE + SwiGLU + LIF-SNN + SSM + JEPA + Synapsis + HSAQ
- **Layers**: 3 | **Hidden**: 256 | **Heads**: 8 | **KV**: 4
- **Vocab**: 208 tokens (char-level)
- **Contexto max**: 64 tokens
- **Dataset**: C4 EN + Wikipedia multilingüe
- **Epochs**: 4 | **Loss final**: 0.0363 | **Accuracy**: 98.83%
- **SNN**: LIF real (threshold=0.05, tau=0.8)
- **JEPA latent dim**: 128
- **Synapsis slots**: 128
- **HSAQ sparsity**: 0.3

---

## Módulos .materia (Fine-tuning del BaseModel)

| # | Archivo | Params | Dataset | Loss | Acc | Estado |
|---|---------|--------|---------|------|-----|--------|
| 1 | `materia-v3-full.materia` | 4,820,224 | C4 EN (15K textos) | 0.0332 | 0.9903 | ✅ Entrenado |
| 2 | `materia-v3-extended.materia` | 3,418,880 | C4 EN (5K textos) | 0.0357 | 0.9896 | ✅ Entrenado |
| 3 | `materia-v3-unified.materia` | 2,417,920 | Wikipedia ES/EN | 0.0006 | 1.0000 | ✅ Entrenado |
| 4 | `materia-v3-nano.materia` | 639,104 | C4 EN (1K textos) | 0.0474 | 0.9885 | ✅ Pre-entrenado |
| 5 | `science-v3.materia` | 2,334,976 | reasoning_dataset (168 QA) | 0.0308 | 0.9980 | ✅ Entrenado |

### Notas:
- `science-v3-part-1.materia` (584B), `science-v3-part-2.materia` (585B) y `science-v3-part-3.materia` (588B) son archivos de configuración, no modelos con pesos.
- Los módulos .materia se cargan como pesos adicionales sobre el modelo base mediante pickle.

---

## Modelos V1 y V2 (Perdidos)

| Modelo | Versión | Estado |
|--------|---------|--------|
| materia-v1.basemateria | V1 | ❌ PERDIDO (incidente 31 marzo 2026) |
| materia-v2.basemateria | V2 | ❌ PERDIDO (incidente 31 marzo 2026) |

---

**Total archivos: 1 .basemateria + 5 .materia entrenados**
**Total perdidos: 2 modelos legacy**

*Última actualización: Julio 2026*
