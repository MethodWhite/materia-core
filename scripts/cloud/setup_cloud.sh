#!/bin/bash
# MATERIA V3 - Setup para cloud (RunPod, Modal, Vast.ai, Colab)
# Uso: bash scripts/cloud/setup_cloud.sh
set -e

echo "=== MATERIA V3 - Cloud Setup ==="

# Detectar entorno
if [ -n "$RUNPOD_POD_ID" ]; then
    echo "Platform: RunPod"
elif [ -n "$KAGGLE_KERNEL_RUN_TYPE" ]; then
    echo "Platform: Kaggle"
elif [ -n "$COLAB_RELEASE_TAG" ]; then
    echo "Platform: Colab"
else
    echo "Platform: Unknown / Manual"
fi

# Crear enlace simbolico para MATERIA_HOME
export MATERIA_HOME="$(cd "$(dirname "$0")/../.." && pwd)"
echo "MATERIA_HOME=$MATERIA_HOME"

# Instalar dependencias
echo "Installing dependencies..."
pip install --quiet torch numpy matplotlib pyyaml 2>&1 | tail -1

# Verificar GPU
python3 -c "
import torch, os
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'GPU: {gpu} ({mem:.1f}GB)')
else:
    print('WARNING: No GPU detected')
"

echo ""
echo "Setup complete. Run training with:"
echo "  python scripts/train.py --config configs/1B.yaml"
