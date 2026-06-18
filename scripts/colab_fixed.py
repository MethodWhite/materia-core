"""
MATERIA V3 - Colab Runner (arreglado)
Pega TODO en UNA celda y ejecuta.

Al final te da una URL ngrok. Pégamela para que monitoree.
"""
# ═══════════════════════════════════════════════════════════════════════
import os, sys, subprocess, threading, time, json, warnings
warnings.filterwarnings('ignore')

NGROK_TOKEN = "3FHwtEpF5UExWE6ijI4qb2KjpQS_4LmPK6vMffvYx4SJhyTun"
OUTPUT = "/content/drive/MyDrive/materia-100m-v2"

# ── Setup ──
if not os.path.exists('/content/materia-core'):
    !git clone https://github.com/MethodWhite/materia-core
    os.chdir('/content/materia-core')
    !pip install -q torch numpy pyyaml matplotlib psutil sentencepiece datasets pyngrok flask

os.chdir('/content/materia-core')
!mkdir -p "$OUTPUT"

# ── Tokenizer ──
tok = '/content/materia-core/data/multilingual/tokenizer/materia_multilingual_v3.model'
if not os.path.exists(tok):
    !python scripts/prepare_data.py --train-tokenizer --vocab-size 32768

# ── Start training ──
ck = [f for f in os.listdir(OUTPUT) if f.startswith('checkpoint_epoch')]
resume = f"--resume {OUTPUT}/{sorted(ck)[-1]}" if ck else ""

cmd = f"python scripts/train.py --config configs/100M.yaml --dataset hf:allenai/c4:en --max-lines 500000 --memory-limit 0.85 --output {OUTPUT} {resume}"
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

def log_output():
    for line in proc.stdout:
        print(line, end='', flush=True)

t = threading.Thread(target=log_output, daemon=True)
t.start()

# ── API server via background thread ──
def run_api():
    from flask import Flask, jsonify
    from pyngrok import ngrok, conf
    conf.get_default().auth_token = NGROK_TOKEN

    app = Flask(__name__)

    @app.route('/status')
    def status():
        return jsonify(running=proc.poll() is None)

    @app.route('/resume')
    def resume():
        global proc
        if proc.poll() is None:
            return jsonify(status='already running')
        ck = [f for f in os.listdir(OUTPUT) if f.startswith('checkpoint_epoch')]
        rs = f"--resume {OUTPUT}/{sorted(ck)[-1]}" if ck else ""
        proc = subprocess.Popen(
            f"python scripts/train.py --config configs/100M.yaml --dataset hf:allenai/c4:en --max-lines 500000 --memory-limit 0.85 --output {OUTPUT} {rs}",
            shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        threading.Thread(target=lambda: [print(l, end='') for l in proc.stdout], daemon=True).start()
        return jsonify(status='resumed')

    url = ngrok.connect(8000).public_url
    print(f"\n✅ URL ACTIVA: {url}")
    print(f"   Comparte esta URL conmigo\n")
    app.run(host='0.0.0.0', port=8000, debug=False, use_reloader=False)

api_thread = threading.Thread(target=run_api, daemon=True)
api_thread.start()

# Mantener la celda viva
while True:
    time.sleep(1)
