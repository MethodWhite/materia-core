"""
MATERIA V3 - Colab API Server
Copia esto en una celda de Colab para que pueda conectarme remotamente.

Expone un servidor HTTP via ngrok al que puedo hacer requests para
monitorear entrenamiento, modificar parámetros, y descargar resultados.
"""
# ─── Celda 1: Setup ───────────────────────────────────────────────
colab_api_setup = """
import os, sys, json, threading, time, subprocess, shutil

# Instalar ngrok
!pip install -q pyngrok fastapi uvicorn pyyaml

# Setup ngrok (necesitas token gratis en https://dashboard.ngrok.com/get-started/your-authtoken)
NGROK_TOKEN = "TU_TOKEN_AQUI"  # <-- Cámbialo por tu token

from pyngrok import ngrok, conf
conf.get_default().auth_token = NGROK_TOKEN

# ─── Celda 2: Clonar repo ─────────────────────────────────────────

!git clone https://github.com/MethodWhite/materia-core
%cd materia-core
!pip install -q torch numpy pyyaml matplotlib psutil sentencepiece datasets

# ─── Celda 3: API Server ──────────────────────────────────────────

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
from pyngrok import ngrok

app = FastAPI(title="MATERIA V3 - Colab API")

# Estado global
state = {
    'training': False,
    'status': 'idle',
    'last_log': '',
    'output_dir': '/content/drive/MyDrive/materia-outputs',
}

@app.get("/status")
def get_status():
    return state

@app.get("/train")
def start_training(
    config: str = Query("configs/100M.yaml"),
    dataset: str = Query("hf:allenai/c4:en"),
    max_lines: int = Query(200000),
    output: str = Query(None),
):
    if state['training']:
        return {'error': 'Already training'}
    
    out = output or f"/content/drive/MyDrive/materia-{config.replace('configs/', '').replace('.yaml', '')}"
    os.makedirs(out, exist_ok=True)
    
    def run():
        state['training'] = True
        state['status'] = 'running'
        cmd = f"python scripts/train.py --config {config} --dataset {dataset} --max-lines {max_lines} --memory-limit 0.85 --output {out}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            state['last_log'] = line.strip()
            print(line, end='')
        proc.wait()
        state['training'] = False
        state['status'] = 'completed' if proc.returncode == 0 else 'failed'
    
    threading.Thread(target=run, daemon=True).start()
    return {'status': 'started', 'output': out}

@app.get("/logs")
def get_logs(lines: int = Query(50)):
    log_file = os.path.join(state['output_dir'], 'training_log.csv')
    if os.path.exists(log_file):
        with open(log_file) as f:
            content = f.read()
        return {'csv': content.split('\\n')[-lines:]}
    return {'last_log': state['last_log']}

@app.get("/download")
def download_results(output: str = Query(None)):
    import glob as g
    out = output or state['output_dir']
    files = []
    for f in g.glob(f"{out}/**/*.*", recursive=True):
        if os.path.isfile(f):
            files.append(f.replace(out, ''))
    return {'files': files}

# Abrir tunel ngrok
public_url = ngrok.connect(8000).public_url
print(f"\\n{'='*60}")
print(f"  MATERIA API activa en: {public_url}")
print(f"  Endpoints:")
print(f"    GET  {public_url}/status")
print(f"    GET  {public_url}/train?config=configs/100M.yaml&dataset=hf:allenai/c4:en&max_lines=200000")
print(f"    GET  {public_url}/logs?lines=50")
print(f"    GET  {public_url}/download")
print(f"{'='*60}\\n")

# Iniciar servidor
uvicorn.run(app, host='0.0.0.0', port=8000)
"""

print("Copia el codigo de abajo y pegalo en tu notebook de Colab")
print("=" * 60)
print(colab_api_setup)
