# PLAN DE RECUPERACIÓN — M.A.T.E.R.I.A.

**Fecha:** 2026-06-15
**Objetivo:** Recuperar modelos perdidos y estandarizar la nomenclatura.

---

## 1. Estado Actual de Modelos

### 1.1 Respaldados (DRAGON_BACKUP)

| Modelo | Ruta | Formato |
|--------|------|---------|
| materia-base.basemateria | `.../models/materia-base.basemateria` | Texto (ref Ollama) |
| materia-core.basemateria | `.../models/materia-core.basemateria` | Texto (ref Ollama) |
| materia-security.basemateria | `.../models/materia-security.basemateria` | Texto (ref Ollama) |
| materia-consolidated.basemateria | `.../models/materia-consolidated.basemateria` | Texto |
| materia-kino.basemateria | `.../models/materia-kino.basemateria` | Texto |
| materia-kino-core.basemateria | `.../models/materia-kino-core.basemateria` | Texto |
| materia-recovery-authenticator.basemateria | `.../models/materia-recovery-authenticator.basemateria` | Texto |
| materia-recovery-exodus.basemateria | `.../models/materia-recovery-exodus.basemateria` | Texto |
| materia-science.basemateria | `.../models/materia-science.basemateria` | Texto |
| materia-science-general.basemateria | `.../models/materia-science-general.basemateria` | Texto |
| materia-science-arxiv.basemateria | `.../models/materia-science-arxiv.basemateria` | Texto |
| materia-science-qa.basemateria | `.../models/materia-science-qa.basemateria` | Texto |
| materia-science-summarize.basemateria | `.../models/materia-science-summarize.basemateria` | Texto |
| num-jepa-v6.basemateria | `.../modules/num_jepa/num-jepa-v6.basemateria` | Python class |
| deep-jepa.basemateria | `.../modules/num_jepa/deep-jepa.basemateria` | Python class |
| +14 modelos .pkl NUM-JEPA | `.../models/num_jepa_*.pkl` | Pickle (pesos) |
| materia_v3_nano_phase1.materia | `.../v3-from-scratch/checkpoints/` | Checkpoint ~154MB |
| materia_v3_nano_phase2.materia | `.../v3-from-scratch/checkpoints/` | Checkpoint ~38MB |

### 1.2 Perdidos (Requieren Re-creación)

| Modelo | Causa | Acción Requerida |
|--------|-------|-----------------|
| materia-v1.basemateria | Eliminación accidental 31/03/2026 | Re-crear desde documentación histórica |
| materia-v2.basemateria | Eliminación accidental 31/03/2026 | Re-crear desde documentación histórica |
| materia-v1-science.basemateria | Eliminación accidental 31/03/2026 | Re-entrenar desde arXiv |
| materia-v2-science.basemateria | Eliminación accidental 31/03/2026 | Re-entrenar desde arXiv |
| GGUF materia-science:latest | Pérdida asociada | Re-entrenar con Ollama |
| GGUF materia-base:latest (~4.7GB) | Pérdida asociada | Descargar o re-entrenar |
| GGUF materia-core:latest | Pérdida asociada | Descargar o re-entrenar |
| GGUF materia-security:latest | Pérdida asociada | Descargar o re-entrenar |

### 1.3 Nunca Entrenados

| Modelo | Estado |
|--------|--------|
| materia-v3.basemateria | Arquitectura implementada, pesos sin entrenar |

---

## 2. Estandarización de Nomenclatura Propuesta

### 2.1 Nuevo Esquema de Nombres

```
materia-{version}-{dominio}-{variante}.basemateria

Ejemplos:
materia-v3-base.basemateria
materia-v3-science.basemateria
materia-v3-science-arxiv.basemateria
materia-v3-science-qa.basemateria
materia-v3-security.basemateria
materia-v3-kino.basemateria
```

### 2.2 Renombrar Modelos Existentes

| Nombre Actual | Nuevo Nombre Propuesto |
|---------------|----------------------|
| materia-base.basemateria | materia-v3-base.basemateria |
| materia-core.basemateria | materia-v3-core.basemateria |
| materia-science.basemateria | materia-v3-science.basemateria |
| materia-science-general.basemateria | materia-v3-science-general.basemateria |
| materia-science-arxiv.basemateria | materia-v3-science-arxiv.basemateria |
| materia-science-qa.basemateria | materia-v3-science-qa.basemateria |
| materia-science-summarize.basemateria | materia-v3-science-summarize.basemateria |
| materia-kino.basemateria | materia-v3-kino.basemateria |
| materia-kino-core.basemateria | materia-v3-kino-core.basemateria |

---

## 3. Pasos para Re-entrenar Modelos Science Perdidos

### 3.1 Descargar Dataset de arXiv

```bash
cd /run/media/methodwhite/DRAGON_BACKUP/projects/materia-vault
python scripts/download_bulk_science.py
python scripts/build_science_dataset.py
```

### 3.2 Entrenar Modelo Science

```bash
python scripts/train_science_model.py

# Esto generará:
# - materia-v3-science.basemateria (texto)
# - materia-science:latest (GGUF en Ollama)
```

### 3.3 Verificar Correlaciones

```bash
python scripts/science_correlation_engine.py
```

---

## 4. Pasos para Entrenar materia-v3.basemateria

```bash
cd /run/media/methodwhite/DRAGON_BACKUP/projects/materia-vault/models/v3-from-scratch
python src/materia_v3_arch.py
python src/trainer.py
```

---

## 5. Verificación de Integridad

```bash
# Verificar que todos los .basemateria tengan contenido válido
find /run/media/methodwhite/DRAGON_BACKUP -name "*.basemateria" -exec file {} \;

# Verificar que los .pkl no estén corruptos
find /run/media/methodwhite/DRAGON_BACKUP -name "*.pkl" -exec python3 -c "
import pickle, sys
try:
    with open(sys.argv[1], 'rb') as f:
        pickle.load(f)
    print(f'OK: {sys.argv[1]}')
except:
    print(f'CORRUPT: {sys.argv[1]}')
" {} \;
```

---

## 6. Respaldo Automático Propuesto

Crear script de backup diario:

```bash
#!/bin/bash
# backup_materia.sh
BACKUP_SRC="/home/methodwhite/MATERIA"
BACKUP_DST="/run/media/methodwhite/DRAGON_BACKUP/backups/MATERIA_$(date +%Y%m%d)"
rsync -av --progress "$BACKUP_SRC/" "$BACKUP_DST/"
```

---

*Plan de Recuperación — M.A.T.E.R.I.A. 2026*
