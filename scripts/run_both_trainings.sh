#!/bin/bash
# Ejecuta ambos entrenamientos V4 en secuencia:
# 1. Sin Synapsis (evitar maldicion de repeticion)
# 2. Con Synapsis (comparar comportamiento)
# Ambos guardan checkpoints progresivos por epoch.

set -e

MATERIA_HOME="/home/methodwhite/MATERIA"
LOG_DIR="$MATERIA_HOME/outputs"

echo "=========================================="
echo "MATERIA V4 - Entrenamiento Doble"
echo "Sin Synapsis + Con Synapsis"
echo "=========================================="
echo ""

# ─── Entrenamiento 1: SIN Synapsis ───────────────────────────
echo "[1/2] Entrenamiento SIN Synapsis"
echo "Config: V4_3.8M_nosynapsis.yaml"
echo "Log: $LOG_DIR/training_v4_nosynapsis.log"
echo ""

cd "$MATERIA_HOME"
CUDA_VISIBLE_DEVICES="" python scripts/train_v4.py \
  --config configs/V4_3.8M_nosynapsis.yaml \
  --no-synapsis \
  --batch-size 8 \
  --memory-limit 0.75 \
  2>&1 | tee "$LOG_DIR/training_v4_nosynapsis_final.log"

echo ""
echo "✓ Entrenamiento SIN Synapsis completado"
echo ""

# ─── Entrenamiento 2: CON Synapsis ───────────────────────────
echo "[2/2] Entrenamiento CON Synapsis"
echo "Config: V4_3.8M_con_synapsis.yaml"
echo "Log: $LOG_DIR/training_v4_con_synapsis.log"
echo ""

cd "$MATERIA_HOME"
CUDA_VISIBLE_DEVICES="" python scripts/train_v4.py \
  --config configs/V4_3.8M_con_synapsis.yaml \
  --batch-size 8 \
  --memory-limit 0.75 \
  2>&1 | tee "$LOG_DIR/training_v4_con_synapsis_final.log"

echo ""
echo "✓ Entrenamiento CON Synapsis completado"
echo ""
echo "=========================================="
echo "Ambos entrenamientos completados"
echo "Checkpoints guardados en outputs/"
echo "=========================================="
