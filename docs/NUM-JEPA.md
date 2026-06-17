# NUM-JEPA — Fine-Tune de M.A.T.E.R.I.A. para Kino Chileno

## Aclaración Importante

**NUM-JEPA NO es el modelo base original de M.A.T.E.R.I.A.** Es un **fine-tune específico** (tipo `materia-v1-NUM-JEPA.basemateria`) creado exclusivamente para la predicción del sorteo Kino Chileno. El modelo base real es materia-v3.basemateria (aún no entrenado desde cero).

## Descripción

NUM-JEPA aplica la arquitectura JEPA (Joint Embedding Predictive Architecture) al dominio numérico de la lotería Kino Chile (números 1-25, selección de 14). Utiliza una malla hexagonal toroidal para modelar las relaciones entre números.

## Arquitectura del Fine-Tune

- **Encoder**: Mapea números a posiciones en un toroide hexagonal
- **Predictor**: Predice embeddings latentes de sorteos futuros
- **Decoder**: Convierte embeddings latentes a probabilidades por número
- **Entrenamiento**: Minimiza MSE entre embedding predicho y real

## Precisión

- Métodos estadísticos: ~8-9/14 (57-64%) máximo alcanzable
- NUM-JEPA v2: 64.29% accuracy reportado
- Sobreajuste detectado en métodos que mostraban 14/14

## Modelos Disponibles

### En `~/MATERIA/models/` (local):
- `num_jepa_kino_chile_v3.pkl` (30 MB) — Fine-tune Kino Chile
- `num_jepa_kino_mega.pkl` (16 MB) — Fine-tune Kino Mega

### Respaldados en DRAGON_BACKUP:
- 17+ modelos .pkl adicionales (v2-v14, ensemble, toroidal, caótico)
- 7+ archivos .basemateria asociados (num-jepa-v*, deep-jepa, etc.)

## Relación con M.A.T.E.R.I.A.

```
JEPA (Arquitectura General)
  └── M.A.T.E.R.I.A. (Motor Principal)
        ├── materia-engine (Rust) — Core del sistema
        ├── materia-vault (datos) — Modelos y módulos
        └── NUM-JEPA (Fine-tune Kino) — Modelo específico numérico
```

## Notas

1. NUM-JEPA es UNO de los posibles modelos fine-tune para M.A.T.E.R.I.A.
2. NO es necesario para el funcionamiento del motor principal
3. Los modelos .pkl son los pesos entrenados; los .basemateria son las definiciones
4. Para más detalles, ver `DOCUMENTACION_MAESTRA_MATERIA.md`

---

*Fine-tune para Kino Chileno — 2026*
