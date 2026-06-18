"""
MATERIA V3 - Colab Persistent Runner
Pega TODO esto en UNA sola celda de Colab.
El entrenamiento sigue aunque se cierre el navegador,
y puede reanudarse si se desconecta.

Manda el ngrok URL que aparece al final para que pueda
monitorear y controlar el entrenamiento remotamente.
"""
# ═══════════════════════════════════════════════════════════════════
# CELDA UNICA - Copia todo esto en Colab y ejecuta
# ═══════════════════════════════════════════════════════════════════

import os, sys, json, threading, time, subprocess, shutil, warnings
warnings.filterwarnings('ignore')

print("=== MATERIA V3 - Colab Persistent Runner ===")

# ── 1. Setup ──
NGROK_TOKEN = "TU_TOKEN_AQUI"  # Obtén token gratis: https://dashboard.ngrok.com/login

# ── 2. Clonar repo (si no existe) ──
if not os.path.exists('/content/materia-core'):
    !git clone https://github.com/MethodWhite/materia-core
    %cd materia-core
    !pip install -q torch numpy pyyaml matplotlib psutil sentencepiece datasets pyngrok fastapi uvicorn
else:
    %cd materia-core

# ── 3. Tokenizer (si no está entrenado) ──
tok_path = '/content/materia-core/data/multilingual/tokenizer/materia_multilingual_v4.model'
if not os.path.exists(tok_path):
    print("Entrenando tokenizer BPE...")
    !python scripts/prepare_data.py --train-tokenizer --vocab-size 32768

# ── 4. Verificar si hay checkpoint para resumir ──
OUTPUT_DIR = "/content/drive/MyDrive/materia-100m-v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('checkpoint_epoch') and f.endswith('.pt')]
RESUME_FLAG = ""
if checkpoints:
    last = sorted(checkpoints)[-1]
    RESUME_FLAG = f"--resume {OUTPUT_DIR}/{last}"
    print(f"Reanudando desde: {last}")

# ── 5. Iniciar entrenamiento ──
train_cmd = f"""python scripts/train.py \
  --config configs/100M.yaml \
  --dataset hf:allenai/c4:en \
  --max-lines 500000 \
  --memory-limit 0.85 \
  --output {OUTPUT_DIR} {RESUME_FLAG}"""

print(f"\nEjecutando: {train_cmd}\n")
proc = subprocess.Popen(train_cmd, shell=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)

def print_output():
    for line in proc.stdout:
        print(line, end='', flush=True)

threading.Thread(target=print_output, daemon=True).start()

# ── 6. API + ngrok ──
from fastapi import FastAPI, Query
import uvicorn
from pyngrok import ngrok, conf
conf.get_default().auth_token = NGROK_TOKEN

app = FastAPI(title="MATERIA V3 - Colab Remote")

@app.get("/status")
def status():
    running = proc.poll() is None
    return {
        'running': running,
        'returncode': proc.returncode,
        'output_dir': OUTPUT_DIR,
    }

@app.get("/resume")
def resume():
    """Forzar reanudación si se detuvo."""
    global proc
    if proc.poll() is None:
        return {'status': 'already running'}
    checkpoints = [f for f in os.listdir(OUTPUT_DIR) if f.startswith('checkpoint_epoch')]
    resume = f"--resume {OUTPUT_DIR}/{sorted(checkpoints)[-1]}" if checkpoints else ""
    cmd = f"python scripts/train.py --config configs/100M.yaml --dataset hf:allenai/c4:en --max-lines 500000 --memory-limit 0.85 --output {OUTPUT_DIR} {resume}"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
    threading.Thread(target=lambda: [print(l, end='') for l in proc.stdout], daemon=True).start()
    return {'status': 'resumed'}

@app.get("/files")
def files():
    import glob
    fs = []
    for f in glob.glob(f"{OUTPUT_DIR}/**/*.*", recursive=True):
        if os.path.isfile(f):
            fs.append(f.replace(OUTPUT_DIR, ''))
    return {'files': fs}

# Abrir tunel
try:
    public_url = ngrok.connect(8000).public_url
    print(f"\n{'='*60}")
    print(f"  MATERIA API activa:")
    print(f"  URL: {public_url}")
    print(f"{'='*60}")
except Exception as e:
    print(f"ngrok error: {e}")
    print("El entrenamiento igual está corriendo en background")

uvicorn.run(app, host='0.0.0.0', port=8000)
