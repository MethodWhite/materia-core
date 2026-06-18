"""
MATERIA V3 - Colab Remote Control via GitHub
Pega en UNA celda de Colab.

Sin ngrok, sin SSH. Yo mando comandos via GitHub y Colab responde.
"""
# ═══════════════════════════════════════════════════════════════════════
import os, subprocess, threading, time, json, requests, warnings, glob
warnings.filterwarnings('ignore')

GITHUB_TOKEN = "3FHwtEpF5UExWE6ijI4qb2KjpQS_4LmPK6vMffvYx4SJhyTun"  # reemplazar con tu token
GITHUB_REPO = "MethodWhite/materia-core"
OUTPUT = "/content/drive/MyDrive/materia-100m-v2"
STATUS_FILE = "colab_status.json"

# Setup
if not os.path.exists('/content/materia-core'):
    !git clone https://github.com/MethodWhite/materia-core
os.chdir('/content/materia-core')

# ── Iniciar entrenamiento ──
os.makedirs(OUTPUT, exist_ok=True)
ck = [f for f in os.listdir(OUTPUT) if f.startswith('checkpoint_epoch')]
rs = f"--resume {OUTPUT}/{sorted(ck)[-1]}" if ck else ""

proc = subprocess.Popen(
    f"python scripts/train.py --config configs/100M.yaml --dataset hf:allenai/c4:en --max-lines 500000 --memory-limit 0.85 --output {OUTPUT} {rs}",
    shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
)
threading.Thread(target=lambda: [print(l, end='', flush=True) for l in proc.stdout], daemon=True).start()

# ── Funciones GitHub API ──
def gh_get(path):
    r = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
                     headers={"Authorization": f"token {GITHUB_TOKEN}"})
    if r.status_code == 200:
        import base64
        return base64.b64decode(r.json()['content']).decode()
    return None

def gh_put(path, content, message):
    import base64
    # Verificar si existe
    r = requests.get(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
                     headers={"Authorization": f"token {GITHUB_TOKEN}"})
    sha = r.json().get('sha') if r.status_code == 200 else None
    data = {"message": message, "content": base64.b64encode(content.encode()).decode()}
    if sha: data["sha"] = sha
    requests.put(f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}",
                 headers={"Authorization": f"token {GITHUB_TOKEN}"}, json=data)

def update_status(estado, info=""):
    # Sube estado actual a GitHub
    s = {
        'running': proc.poll() is None,
        'epoch': info,
        'time': time.strftime('%H:%M:%S'),
        'memory': f"{os.popen('free -g | head -2').read().split()[-2]}GB" if os.name == 'posix' else '',
    }
    if not proc.poll() is None:
        s['exit'] = proc.returncode
    gh_put(STATUS_FILE, json.dumps(s, indent=2), f"status: {s['time']}")

# Enviar heartbeat inicial
update_status('started')

# ── Polling de comandos ──
last_command = ''
while True:
    try:
        # Leer comando desde GitHub
        raw = gh_get('colab_command.json')
        if raw and raw != last_command:
            cmd = json.loads(raw)
            last_command = raw
            action = cmd.get('action', '')

            if action == 'status':
                update_status('ok')
            elif action == 'stop':
                if proc.poll() is None:
                    proc.kill()
                update_status('stopped')
            elif action == 'resume':
                ck = [f for f in os.listdir(OUTPUT) if f.startswith('checkpoint_epoch')]
                r = f"--resume {OUTPUT}/{sorted(ck)[-1]}" if ck else ""
                proc = subprocess.Popen(f"python scripts/train.py --config configs/100M.yaml --dataset hf:allenai/c4:en --max-lines 500000 --memory-limit 0.85 --output {OUTPUT} {r}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
                threading.Thread(target=lambda: [print(l, end='') for l in proc.stdout], daemon=True).start()
                update_status('resumed')
            elif action == 'metrics':
                fs = sorted(glob.glob(f"{OUTPUT}/*.csv"))
                update_status('metrics', open(fs[-1]).read() if fs else 'no csv yet')
    except:
        pass
    time.sleep(15)  # Poll cada 15 segundos
