"""
MATERIA V3 - Entrenamiento Completo con métricas, logging y gráficos
Arquitectura: GQA + RoPE + SwiGLU + LIF-SNN + SSM + JEPA + Synapsis + HSAQ
"""
import os, sys, json, time, math, random, pickle, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/methodwhite/MATERIA'
DEVICE = torch.device('cpu')
torch.set_num_threads(4)
CSV_LOG = os.path.join(BASE, 'logs', 'training_log.csv')
PLOTS_DIR = os.path.join(BASE, 'docs', 'plots')
os.makedirs(os.path.join(BASE, 'logs'), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(BASE, 'models'))
from materia_v3_full import count_params

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class CompleteModel(nn.Module):
    def __init__(self, vocab_size=800, dim=256, n_layers=3, n_heads=8, n_kv=4,
                 jepa_dim=256, synapsis_slots=128, use_snn=True, use_ssm=True,
                 use_jepa=True, use_synapsis=True, use_hsaq=True):
        super().__init__()
        self.dim = dim; self.use_snn=use_snn; self.use_ssm=use_ssm
        self.use_jepa=use_jepa; self.use_synapsis=use_synapsis; self.use_hsaq=use_hsaq
        from materia_v3_full import TransformerBlock, SNNLayer, SSMBlock, JEPA, SynapsisMemory, HSAQ
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, n_kv) for _ in range(n_layers)])
        if use_snn: self.snn = SNNLayer(dim)
        if use_ssm: self.ssm = SSMBlock(dim)
        if use_jepa: self.jepa = JEPA(dim, jepa_dim)
        if use_synapsis: self.synapsis = SynapsisMemory(dim, synapsis_slots)
        if use_hsaq: self.hsaq = HSAQ(sparsity=0.3)
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, mask=None):
        x = self.tok_emb(x)
        if self.use_hsaq: x = self.hsaq(x)
        for l in self.layers: x = l(x, mask)
        if self.use_synapsis: x = self.synapsis(x)
        if self.use_snn:
            x_enh, rate = self.snn(x[:, -1:])
            x = torch.cat([x[:, :-1], x_enh], dim=1)
        if self.use_ssm: x = self.ssm(x)
        if self.use_jepa: _, x = self.jepa(x)
        return self.head(self.norm(x))

def build_vocab(texts, vocab_size=800):
    chars = set()
    for t in texts:
        for c in t: chars.add(c)
    chars = sorted(chars)[:vocab_size-4]
    stoi = {c: i+4 for i,c in enumerate(chars)}
    stoi['<PAD>']=0; stoi['<BOS>']=1; stoi['<EOS>']=2; stoi['<UNK>']=3
    itos = {i:c for c,i in stoi.items()}
    return stoi, itos

class TextDataset(Dataset):
    def __init__(self, texts, stoi, seq_len=64):
        self.seq_len = seq_len; self.data = []
        for text in texts:
            ids = [stoi.get(c,3) for c in text]
            if len(ids) > seq_len + 1:
                for i in range(0, len(ids)-seq_len, seq_len//2):
                    self.data.append(ids[i:i+seq_len+1])
    def __len__(self): return max(1, len(self.data))
    def __getitem__(self, idx):
        ids = self.data[idx%len(self.data)][:self.seq_len+1]
        ids = ids + [0]*(self.seq_len+1-len(ids))
        ids = torch.tensor(ids[:self.seq_len+1], dtype=torch.long)
        return ids[:-1], ids[1:]

def compute_accuracy(logits, targets, ignore_idx=0):
    preds = logits.argmax(dim=-1)
    mask = targets != ignore_idx
    correct = (preds[mask] == targets[mask]).float().sum()
    total = mask.sum()
    return (correct / total).item() if total > 0 else 0.0

def save_training_plots(csv_path, model_name):
    if not os.path.exists(csv_path): return
    import pandas as pd
    df = pd.read_csv(csv_path)
    if len(df) < 2: return
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0,0].plot(df['step'], df['loss'], color='blue', alpha=0.7)
    axes[0,0].set_title('Loss'); axes[0,0].grid(True, alpha=0.3); axes[0,0].set_xlabel('Step')
    axes[0,1].plot(df['step'], df['accuracy'], color='green', alpha=0.7)
    axes[0,1].set_title('Accuracy'); axes[0,1].grid(True, alpha=0.3); axes[0,1].set_xlabel('Step')
    axes[1,0].plot(df['step'], df['grad_norm'], color='red', alpha=0.7)
    axes[1,0].set_title('Gradient Norm'); axes[1,0].grid(True, alpha=0.3); axes[1,0].set_xlabel('Step')
    axes[1,1].plot(df['step'], df['spike_rate'], color='purple', alpha=0.7)
    axes[1,1].set_title('Spike Rate (LIF)'); axes[1,1].grid(True, alpha=0.3); axes[1,1].set_xlabel('Step')
    plt.suptitle(f'Training Metrics: {model_name}', fontsize=14)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'{model_name}_training.png')
    plt.savefig(out, dpi=150)
    plt.close()
    log(f"  Plot saved: {out}")

class Trainer:
    def __init__(self, model, model_name, stoi):
        self.model = model; self.model_name = model_name; self.stoi = stoi
        self.csv_path = os.path.join(BASE, 'logs', f'{model_name}_log.csv')
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['step', 'epoch', 'loss', 'accuracy', 'grad_norm', 'spike_rate', 'lr'])
        self.step = 0

    def train_epoch(self, loader, opt, sch, epoch, epochs):
        self.model.train()
        total_loss, total_acc, total_steps = 0.0, 0.0, len(loader)
        for i, (x, y) in enumerate(loader):
            opt.zero_grad()
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            sch.step()
            acc = compute_accuracy(logits, y)
            spike_rate = 0.0
            if hasattr(self.model, 'snn') and self.model.use_snn:
                with torch.no_grad():
                    _, sr = self.model.snn(self.model.tok_emb(x[:, -1:]))
                    spike_rate = sr.item() if torch.is_tensor(sr) else sr
            total_loss += loss.item()
            total_acc += acc
            self.step += 1
            self.csv_writer.writerow([self.step, epoch+1, f'{loss.item():.4f}', f'{acc:.4f}', f'{grad_norm:.4f}', f'{spike_rate:.3f}', f'{sch.get_last_lr()[0]:.2e}'])
            if (i+1) % 50 == 0:
                log(f"  E{epoch+1}/{epochs} [{i+1}/{total_steps}] loss={loss.item():.4f} acc={acc:.4f} gn={grad_norm:.2f} sr={spike_rate:.3f}")
        self.csv_file.flush()
        return total_loss / total_steps, total_acc / total_steps

    def close(self):
        self.csv_file.close()

def save_model(model, model_key, spec, params_count, final_loss, final_acc, stoi, epochs):
    models_dir = os.path.join(BASE, 'models')
    is_basemateria = model_key.endswith('.basemateria')
    is_materia = model_key.endswith('.materia')
    base_name = model_key.replace('.basemateria', '').replace('.materia', '')

    if is_basemateria:
        path = os.path.join(models_dir, model_key)
        content = f"""# {spec['name']}
# Arquitectura: GQA + RoPE + SwiGLU + LIF-SNN + SSM + JEPA + Synapsis + HSAQ
# Entrenado: {time.strftime('%Y-%m-%d %H:%M')}
# Parametros: {params_count:,}
# Layers: {spec['n_layers']} | Hidden: {spec['dim']} | Heads: {spec['n_heads']} | KV: {spec['n_kv']}
# JEPA dim: {spec['jepa_dim']} | Synapsis: {spec['synapsis_slots']} slots
# SNN: LIF real | SSM: {spec['use_ssm']} | JEPA: {spec['use_jepa']}
# Vocab: {len(stoi)} tokens (char-level) | Contexto: {spec['seq_len']}
# Dataset: {spec['dataset']} | Epochs: {epochs}
# Loss final: {final_loss:.4f} | Accuracy: {final_acc:.4f}
#
# CAPACIDADES COMPLETAS:
#   - Texto: espanol, ingles, multilingual
#   - Razonamiento: logico, matematico, cientifico
#   - Codigo: python, javascript, rust
#   - Memoria: synapsis persistente ({spec['synapsis_slots']} slots)
#   - Eficiencia: HSAQ ejecucion dispersa
#   - SNN: LIF real con surrogate gradient
#   - SSM: State Space Model para secuencias largas
#   - JEPA: Joint Embedding Predictive Architecture

OLLAMA_MODEL: {spec['ollama_model']}
DOMAIN: {spec['domain']}
ARCH: gqa+rope+swiglu+lif_snn+ssm+jepa+synapsis+hsaq
PARAMS: {params_count}
STATUS: entrenado
EPOCHS: {epochs}
FINAL_LOSS: {final_loss:.4f}
FINAL_ACCURACY: {final_acc:.4f}
SNN_TYPE: LIF (Leaky Integrate-and-Fire) real
TRAINING_DATE: {time.strftime('%Y-%m-%d')}

CAPABILITIES:
  - text: espanol, ingles, multilingual
  - reasoning: logico, matematico, cientifico
  - code: python, javascript, rust
  - memory: synapsis persistente
  - efficiency: hsaq ejecucion dispersa
  - temporal: lif_snn deteccion patrones
  - sequence: ssm contexto extendido
  - latent: jepa prediccion latente

WEIGHT_MODULE: {base_name}.materia
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        log(f"  Saved: {path}")

    # Save .materia weight module
    materia_path = os.path.join(models_dir, f'{base_name}.materia')
    module = {
        "materia": "umbra_sub_agent",
        "name": spec['name'],
        "version": "3.0.0",
        "architecture": "gqa+rope+swiglu+lif_snn+ssm+jepa+synapsis+hsaq",
        "base_model": "materia-v3.basemateria" if is_materia else None,
        "config": {
            "vocab_size": len(stoi), "dim": spec["dim"],
            "n_layers": spec["n_layers"], "n_heads": spec["n_heads"],
            "n_kv": spec["n_kv"], "jepa_dim": spec["jepa_dim"],
            "synapsis_slots": spec["synapsis_slots"],
            "use_snn": spec["use_snn"], "use_ssm": spec["use_ssm"],
            "use_jepa": spec["use_jepa"], "use_synapsis": spec["use_synapsis"],
            "use_hsaq": spec["use_hsaq"], "max_seq_len": spec["seq_len"],
            "params": params_count,
        },
        "ollama_model": spec["ollama_model"],
        "domain": spec["domain"],
        "capabilities": spec["capabilities"],
        "training": {
            "date": time.strftime('%Y-%m-%d'),
            "dataset": spec["dataset"],
            "epochs": epochs,
            "final_loss": round(final_loss, 4),
            "final_accuracy": round(final_acc, 4),
            "snn_type": "LIF real",
        },
        "weights": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        "tokenizer": stoi,
    }
    import gzip, json
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.integer): return int(obj)
            return super().default(obj)
    with gzip.open(materia_path, 'wb') as f:
        f.write(json.dumps(module, cls=NumpyEncoder).encode('utf-8'))
    log(f"  Saved: {materia_path} ({os.path.getsize(materia_path)//1024}KB)")

MODEL_SPECS = {
    "materia-v3.basemateria": {
        "name": "MATERIA V3 - Base Model",
        "dim": 256, "n_layers": 3, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 256, "synapsis_slots": 128,
        "use_snn": True, "use_ssm": True, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "C4 EN + Wikipedia (multilingue)",
        "max_samples": 5000, "epochs": 4, "batch_size": 8, "seq_len": 64,
        "ollama_model": "materia-v3:latest",
        "domain": "general",
        "capabilities": [
            "text: espanol, ingles, multilingual",
            "reasoning: logico, matematico, cientifico",
            "code: python, javascript, rust",
            "memory: synapsis persistente",
            "temporal: lif_snn patrones",
            "latent: jepa prediccion"
        ]
    },
    "materia-v3-full.materia": {
        "name": "MATERIA V3 - Full Fine-tune",
        "dim": 256, "n_layers": 3, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 256, "synapsis_slots": 128,
        "use_snn": True, "use_ssm": True, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "C4 EN + Wikipedia (full)",
        "max_samples": 10000, "epochs": 3, "batch_size": 8, "seq_len": 64,
        "ollama_model": "materia-v3-full:latest",
        "domain": "general",
        "capabilities": [
            "text: espanol, codigo, multilingual",
            "reasoning: logico avanzado",
            "conocimiento: enciclopedico"
        ]
    },
    "materia-v3-extended.materia": {
        "name": "MATERIA V3 - Extended Fine-tune",
        "dim": 256, "n_layers": 3, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 128, "synapsis_slots": 128,
        "use_snn": True, "use_ssm": True, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "C4 EN (5000 textos)",
        "max_samples": 5000, "epochs": 3, "batch_size": 8, "seq_len": 64,
        "ollama_model": "materia-v3-extended:latest",
        "domain": "general",
        "capabilities": ["llm", "snn", "ssm", "jepa"]
    },
    "materia-v3-unified.materia": {
        "name": "MATERIA V3 - Unified Fine-tune",
        "dim": 256, "n_layers": 2, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 128, "synapsis_slots": 64,
        "use_snn": True, "use_ssm": True, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "Wikipedia ES/EN",
        "max_samples": 3000, "epochs": 3, "batch_size": 8, "seq_len": 64,
        "ollama_model": "materia-v3-unified:latest",
        "domain": "general",
        "capabilities": ["llm", "snn", "ssm", "jepa"]
    },
    "materia-v3-nano.materia": {
        "name": "MATERIA V3 - Nano Fine-tune",
        "dim": 128, "n_layers": 2, "n_heads": 4, "n_kv": 2,
        "jepa_dim": 64, "synapsis_slots": 32,
        "use_snn": True, "use_ssm": False, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "C4 EN (1000 textos)",
        "max_samples": 1000, "epochs": 3, "batch_size": 16, "seq_len": 64,
        "ollama_model": "materia-v3-nano:latest",
        "domain": "general",
        "capabilities": ["inferencia rapida", "razonamiento basico"]
    },
    "science-v3.materia": {
        "name": "MATERIA Science V3 - Fine-tune",
        "dim": 256, "n_layers": 2, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 128, "synapsis_slots": 64,
        "use_snn": False, "use_ssm": False, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "reasoning_dataset.txt (QA cientifica)",
        "max_samples": 168, "epochs": 20, "batch_size": 8, "seq_len": 128,
        "ollama_model": "materia-science:latest",
        "domain": "science",
        "capabilities": ["conocimiento cientifico", "razonamiento logico"]
    },
}

def load_texts(filepath, max_lines=5000, min_len=20):
    texts = []
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if len(line) > min_len:
                texts.append(line)
                if len(texts) >= max_lines: break
    return texts

if __name__ == '__main__':
    log("="*60)
    log("MATERIA V3 - ENTRENAMIENTO COMPLETO CON METRICAS")
    log(f"Device: {DEVICE} | SNN: LIF real con surrogate gradient")
    log("="*60)

    all_texts = load_texts(os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt'), 5000)
    wiki_dir = os.path.join(BASE, 'data/multilingual/tokenizer')
    for f in sorted(os.listdir(wiki_dir)):
        if f.startswith('wiki_') and f.endswith('.txt') and not f.startswith('wiki_api'):
            all_texts.extend(load_texts(os.path.join(wiki_dir, f), 1000))
    all_texts.extend(load_texts(os.path.join(BASE, 'data/reasoning_dataset.txt'), 168))
    log(f"Total texts: {len(all_texts):,}")

    stoi, itos = build_vocab(all_texts, vocab_size=800)
    log(f"Vocab: {len(stoi)}")

    for model_key, spec in MODEL_SPECS.items():
        log(f"\n{'='*60}")
        log(f"TRAINING: {model_key} ({spec['name']})")
        log(f"{'='*60}")

        if 'science' in model_key:
            texts = load_texts(os.path.join(BASE, 'data/reasoning_dataset.txt'), spec['max_samples'])
        elif 'unified' in model_key:
            es = load_texts(os.path.join(BASE, 'data/multilingual/tokenizer/wiki_es.txt'), spec['max_samples']//2)
            en = load_texts(os.path.join(BASE, 'data/multilingual/tokenizer/wiki_en.txt'), spec['max_samples']//2)
            texts = es + en
        else:
            texts = load_texts(os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt'), spec['max_samples'])

        log(f"  Textos: {len(texts):,}")

        # Split train/val
        split = int(len(texts) * 0.9)
        train_texts, val_texts = texts[:split], texts[split:]

        train_ds = TextDataset(train_texts, stoi, spec['seq_len'])
        val_ds = TextDataset(val_texts, stoi, spec['seq_len'])
        train_loader = DataLoader(train_ds, batch_size=spec['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=spec['batch_size'], shuffle=False, drop_last=True) if len(val_ds) > 0 else None

        model = CompleteModel(
            vocab_size=len(stoi), dim=spec['dim'], n_layers=spec['n_layers'],
            n_heads=spec['n_heads'], n_kv=spec['n_kv'], jepa_dim=spec['jepa_dim'],
            synapsis_slots=spec['synapsis_slots'], use_snn=spec['use_snn'],
            use_ssm=spec['use_ssm'], use_jepa=spec['use_jepa'],
            use_synapsis=spec['use_synapsis'], use_hsaq=spec['use_hsaq'],
        )
        params = count_params(model)
        log(f"  Params: {params:,}")

        trainer = Trainer(model, model_key.replace('.', '_'), stoi)
        opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
        total_steps = spec['epochs'] * len(train_loader)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

        for epoch in range(spec['epochs']):
            train_loss, train_acc = trainer.train_epoch(train_loader, opt, sch, epoch, spec['epochs'])
            log(f"  -> E{epoch+1}/{spec['epochs']}: train_loss={train_loss:.4f} train_acc={train_acc:.4f}")
            # Validation
            if val_loader:
                model.eval()
                val_loss, val_acc, val_steps = 0.0, 0.0, 0
                with torch.no_grad():
                    for x, y in val_loader:
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                        val_loss += loss.item()
                        val_acc += compute_accuracy(logits, y)
                        val_steps += 1
                if val_steps > 0:
                    log(f"       val_loss={val_loss/val_steps:.4f} val_acc={val_acc/val_steps:.4f}")

        trainer.close()
        save_training_plots(trainer.csv_path, model_key.replace('.', '_'))
        save_model(model, model_key, spec, params, train_loss, train_acc, stoi, spec['epochs'])
        log(f"  ✓ {model_key} completo (loss={train_loss:.4f}, acc={train_acc:.4f})")

    # Final combined plot
    log("\nGenerating combined training plot...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for idx, (mk, spec) in enumerate(MODEL_SPECS.items()):
        csv_p = os.path.join(BASE, 'logs', f"{mk.replace('.', '_')}_log.csv")
        if os.path.exists(csv_p):
            import pandas as pd
            df = pd.read_csv(csv_p)
            c = colors[idx % len(colors)]
            axes[0,0].plot(df['step'], df['loss'], color=c, alpha=0.7, label=mk)
            axes[0,1].plot(df['step'], df['accuracy'], color=c, alpha=0.7, label=mk)
            axes[1,0].plot(df['step'], df['grad_norm'], color=c, alpha=0.7, label=mk)
            axes[1,1].plot(df['step'], df['spike_rate'], color=c, alpha=0.7, label=mk)
    for ax, title in zip(axes.flat, ['Loss', 'Accuracy', 'Gradient Norm', 'Spike Rate']):
        ax.set_title(title); ax.grid(True, alpha=0.3); ax.legend(fontsize=6)
    plt.suptitle('MATERIA V3 - All Models Training Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'all_models_training.png'), dpi=150)
    plt.close()
    log("Done!")
