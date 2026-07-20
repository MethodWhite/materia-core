#!/bin/bash
# Espera a que termine el entrenamiento sin Synapsis y notifica
TRAINING_PID=$(pgrep -f "train_v4.py.*nosynapsis" | head -1)
LOG="/home/methodwhite/MATERIA/outputs/training_v4_nosynapsis_final.log"

echo "Monitoreando PID: $TRAINING_PID"
echo "Log: $LOG"

while kill -0 $TRAINING_PID 2>/dev/null; do
    sleep 60
    LAST=$(tail -1 $LOG 2>/dev/null)
    echo "[$(date +%H:%M:%S)] $LAST"
done

echo ""
echo "=========================================="
echo "ENTRENAMIENTO SIN SYNAPSIS COMPLETADO"
echo "=========================================="
echo "Fecha: $(date)"
echo ""
echo "Ultimas lineas del log:"
tail -10 $LOG

# Notificar por desktop si disponible
which notify-send >/dev/null 2>&1 && notify-send "MATERIA V4" "Entrenamiento sin Synapsis completado!"
