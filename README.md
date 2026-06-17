# M.A.T.E.R.I.A. V3

Multi-Analytical Toroidal Engine for Recursive Intelligent Analysis

Arquitectura multi-paradigma: GQA + RoPE + SwiGLU + LIF-SNN + SSM + JEPA + Synapsis + HSAQ

## Estructura

```
MATERIA/
├── engine/           ← Web UI (materia-engine repo, Vite+React)
├── models/           ← Codigo del modelo
│   └── core/         ← Componentes modulares
├── configs/          ← Configuraciones YAML escalables (3.8M a 1B)
├── scripts/          ← Entrenamiento y utilidades
│   └── cloud/        ← Deploy en RunPod/Modal/Colab
├── docs/             ← Documentacion, paper, diagramas
├── data/             ← Datasets (gitignored, muy grandes)
├── logs/             ← Logs de entrenamiento (gitignored)
└── outputs/          ← Resultados generados (gitignored)
```

## Entrenar

```bash
python scripts/train.py --config configs/3.8M.yaml --memory-limit 0.75
```

## Requisitos

`pip install -r requirements.txt`
