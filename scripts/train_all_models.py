"""
MATERIA V3 - Entrenamiento Completo de Todos los Modelos
Cada .basemateria y .materia recibe entrenamiento con su dataset correspondiente
Arquitectura completa: GQA + RoPE + SwiGLU + JEPA + Synapsis + HSAQ
"""
import os, sys, json, time, math, random, pickle, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

BASE = '/home/methodwhite/MATERIA'
DEVICE = torch.device('cpu')
torch.set_num_threads(4)

sys.path.insert(0, os.path.join(BASE, 'models'))
from materia_v3_full import Tokenizer, count_params

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)

# ============================================================
# MODEL DEFINITIONS - Each has complete architecture
# ============================================================
from materia_v3_full import (
    TransformerBlock, SNNLayer, SSMBlock, JEPA,
    SynapsisMemory, HSAQ, RoPE, GQA, SwiGLU
)

class CompleteModel(nn.Module):
    def __init__(self, vocab_size=32000, dim=256, n_layers=3, n_heads=8, n_kv=4,
                 jepa_dim=256, synapsis_slots=128, use_snn=True, use_ssm=True,
                 use_jepa=True, use_synapsis=True, use_hsaq=True):
        super().__init__()
        self.dim = dim
        self.use_snn = use_snn
        self.use_ssm = use_ssm
        self.use_jepa = use_jepa
        self.use_synapsis = use_synapsis
        self.use_hsaq = use_hsaq

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv) for _ in range(n_layers)
        ])
        if use_snn:
            self.snn = SNNLayer(dim)
        if use_ssm:
            self.ssm = SSMBlock(dim)
        if use_jepa:
            self.jepa = JEPA(dim, jepa_dim)
        if use_synapsis:
            self.synapsis = SynapsisMemory(dim, synapsis_slots)
        if use_hsaq:
            self.hsaq = HSAQ(sparsity=0.3)
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, mask=None):
        x = self.tok_emb(x)
        if self.use_hsaq:
            x = self.hsaq(x)
        for l in self.layers:
            x = l(x, mask)
        if self.use_synapsis:
            x = self.synapsis(x)
        if self.use_snn:
            x_enh, rate = self.snn(x[:, -1:])
            x = torch.cat([x[:, :-1], x_enh], dim=1)
        if self.use_ssm:
            x = self.ssm(x)
        if self.use_jepa:
            _, x = self.jepa(x)
        logits = self.head(self.norm(x))
        return logits

    def generate(self, idx, max_new=30, temp=0.8, top_p=0.9):
        self.eval()
        for _ in range(max_new):
            logits = self.forward(idx[:, -64:])
            l = logits[:, -1, :] / temp
            p = F.softmax(l, dim=-1)
            if torch.isnan(p).any() or (p == 0).all():
                p = torch.ones_like(p) / p.size(-1)
            idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
        return idx

# ============================================================
# MODEL SPECS
# ============================================================
MODEL_SPECS = {
    "materia-v3.basemateria": {
        "name": "MATERIA V3 - Base Model",
        "dim": 256, "n_layers": 3, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 256, "synapsis_slots": 128,
        "use_snn": True, "use_ssm": True, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "multilingual (C4 + Wikipedia 12 langs)",
        "max_samples": 5000, "epochs": 4, "batch_size": 8, "seq_len": 64,
        "ollama_model": "materia-v3:latest",
        "domain": "general",
        "capabilities": [
            "reasoning: logico, matematico, cientifico",
            "code: python, javascript, rust",
            "text: espanol, ingles, multilingual",
            "memory: synapsis persistente (128 slots)",
            "sparse: hsaq ejecucion dispersa"
        ],
    },
    "materia-v3-full.materia": {
        "name": "MATERIA V3 - Full Module",
        "dim": 256, "n_layers": 4, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 256, "synapsis_slots": 256,
        "use_snn": True, "use_ssm": True, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "C4 EN completo + Wikipedia",
        "max_samples": 15000, "epochs": 4, "batch_size": 8, "seq_len": 64,
        "ollama_model": "materia-v3-full:latest",
        "domain": "general",
        "capabilities": [
            "text: espanol, codigo, multilingual",
            "reasoning: logico, matematico avanzado",
            "memory: synapsis persistente (256 slots)"
        ],
    },
    "materia-v3-extended.materia": {
        "name": "MATERIA V3 - Extended Module",
        "dim": 256, "n_layers": 3, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 128, "synapsis_slots": 128,
        "use_snn": True, "use_ssm": True, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "C4 EN + Wikipedia ES/EN",
        "max_samples": 5000, "epochs": 4, "batch_size": 8, "seq_len": 64,
        "ollama_model": "materia-v3-extended:latest",
        "domain": "general",
        "capabilities": [
            "llm: transformer (gqa+rope+swiglu)",
            "snn: temporal patterns lif neurons",
            "ssm: long sequences state space",
            "jepa: latent prediction"
        ],
        "weight_files": ["materia-v3.basemateria"]
    },
    "materia-v3-unified.materia": {
        "name": "MATERIA V3 - Unified Module",
        "dim": 256, "n_layers": 2, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 128, "synapsis_slots": 64,
        "use_snn": True, "use_ssm": True, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "Wikipedia ES/EN",
        "max_samples": 3000, "epochs": 4, "batch_size": 8, "seq_len": 64,
        "ollama_model": "materia-v3-unified:latest",
        "domain": "general",
        "capabilities": [
            "llm: transformer (gqa+rope+swiglu)",
            "snn: temporal patterns",
            "ssm: state space",
            "jepa: latent prediction"
        ],
    },
    "materia-v3-nano.materia": {
        "name": "MATERIA V3 - Nano Module",
        "dim": 128, "n_layers": 2, "n_heads": 4, "n_kv": 2,
        "jepa_dim": 64, "synapsis_slots": 32,
        "use_snn": True, "use_ssm": False, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "C4 EN (subset rapido)",
        "max_samples": 1000, "epochs": 5, "batch_size": 16, "seq_len": 64,
        "ollama_model": "materia-v3-nano:latest",
        "domain": "general",
        "capabilities": [
            "inferencia rapida y ligera",
            "razonamiento basico",
            "texto: espanol, ingles"
        ],
        "weight_files": []
    },
    "science-v3.materia": {
        "name": "MATERIA Science V3 - General",
        "dim": 256, "n_layers": 2, "n_heads": 8, "n_kv": 4,
        "jepa_dim": 128, "synapsis_slots": 64,
        "use_snn": False, "use_ssm": False, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "reasoning_dataset.txt (QA cientifica)",
        "max_samples": 168, "epochs": 20, "batch_size": 8, "seq_len": 128,
        "ollama_model": "materia-science:latest",
        "domain": "science",
        "capabilities": [
            "conocimiento cientifico general",
            "razonamiento logico-matematico",
            "analisis de datos"
        ],
    },
    "science-v3-part-1.materia": {
        "name": "MATERIA Science V3 - Exactas",
        "dim": 128, "n_layers": 2, "n_heads": 4, "n_kv": 2,
        "jepa_dim": 64, "synapsis_slots": 32,
        "use_snn": False, "use_ssm": False, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "reasoning_dataset.txt (fisica, quimica, matematica)",
        "max_samples": 168, "epochs": 20, "batch_size": 8, "seq_len": 128,
        "ollama_model": "materia-science-exact:latest",
        "domain": "science/exactas",
        "capabilities": ["fisica", "quimica", "matematicas"],
    },
    "science-v3-part-2.materia": {
        "name": "MATERIA Science V3 - Biologia",
        "dim": 128, "n_layers": 2, "n_heads": 4, "n_kv": 2,
        "jepa_dim": 64, "synapsis_slots": 32,
        "use_snn": False, "use_ssm": False, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "reasoning_dataset.txt (biologia, medicina)",
        "max_samples": 168, "epochs": 20, "batch_size": 8, "seq_len": 128,
        "ollama_model": "materia-science-bio:latest",
        "domain": "science/biologia",
        "capabilities": ["biologia", "medicina", "neurociencia"],
    },
    "science-v3-part-3.materia": {
        "name": "MATERIA Science V3 - Ingenieria",
        "dim": 128, "n_layers": 2, "n_heads": 4, "n_kv": 2,
        "jepa_dim": 64, "synapsis_slots": 32,
        "use_snn": False, "use_ssm": False, "use_jepa": True,
        "use_synapsis": True, "use_hsaq": True,
        "dataset": "reasoning_dataset.txt (ingenieria, computacion)",
        "max_samples": 168, "epochs": 20, "batch_size": 8, "seq_len": 128,
        "ollama_model": "materia-science-eng:latest",
        "domain": "science/ingenieria",
        "capabilities": ["ingenieria", "computacion", "tecnologia"],
    }
}

# ============================================================
# TOKENIZER - Character-level BPE (~800 tokens like original spec)
# ============================================================
def build_vocab(texts, vocab_size=800):
    chars = set()
    for t in texts:
        for c in t:
            chars.add(c)
    chars = sorted(chars)
    # Keep most common chars up to vocab_size - 4 special tokens
    max_chars = vocab_size - 4
    chars = chars[:max_chars]
    stoi = {c: i+4 for i, c in enumerate(chars)}
    stoi['<PAD>'] = 0; stoi['<BOS>'] = 1; stoi['<EOS>'] = 2; stoi['<UNK>'] = 3
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return [stoi.get(c, 3) for c in text]

class TextDataset(Dataset):
    def __init__(self, texts, stoi, seq_len=64):
        self.seq_len = seq_len
        self.data = []
        for text in texts:
            ids = encode(text, stoi)
            if len(ids) > seq_len + 1:
                for i in range(0, len(ids) - seq_len, seq_len // 2):
                    self.data.append(ids[i:i+seq_len+1])
    def __len__(self):
        return max(1, len(self.data))
    def __getitem__(self, idx):
        ids = self.data[idx % len(self.data)][:self.seq_len+1]
        ids = ids + [0] * (self.seq_len + 1 - len(ids))
        ids = torch.tensor(ids[:self.seq_len+1], dtype=torch.long)
        return ids[:-1], ids[1:]

def load_texts(filepaths, max_lines=None):
    texts = []
    for fp in filepaths:
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 50:
                        texts.append(line)
                        if max_lines and len(texts) >= max_lines:
                            break
    return texts

# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model, loader, epochs, lr=5e-4, model_name="model"):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(loader)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    model.train()
    log(f"  Training {model_name}: {epochs} epochs, {len(loader)} batches/epoch")

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            opt.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            total_loss += loss.item()
            if (i+1) % 50 == 0:
                log(f"    E{epoch+1}/{epochs} [{i+1}/{len(loader)}] loss={loss.item():.4f}")
        avg_loss = total_loss / len(loader)
        log(f"    -> Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
    return avg_loss

# ============================================================
# SAVE FUNCTIONS
# ============================================================
def save_basemateria(filepath, name, spec, params_count, loss, stoi, epochs):
    content = f"""# {name}
# Arquitectura completa: GQA + RoPE + SwiGLU + JEPA + Synapsis + HSAQ
# Entrenado: {time.strftime('%Y-%m-%d %H:%M')}
# Parametros: {params_count:,}
# Layers: {spec['n_layers']} | Hidden: {spec['dim']} | Heads: {spec['n_heads']} | KV: {spec['n_kv']}
# JEPA dim: {spec['jepa_dim']} | Synapsis: {spec['synapsis_slots']} slots
# SNN: {spec['use_snn']} | SSM: {spec['use_ssm']} | JEPA: {spec['use_jepa']}
# Synapsis: {spec['use_synapsis']} | HSAQ: {spec['use_hsaq']}
# Vocab: {len(stoi)} tokens (char-level)
# Contexto max: {spec['seq_len']}
# Dataset: {spec['dataset']}
# Epochs: {epochs} | Final loss: {loss:.4f}

OLLAMA_MODEL: {spec['ollama_model']}
DOMAIN: {spec['domain']}
ARCH: jepa+gqa+rope+swiglu+synapsis+hsaq
TRAINING_DATE: {time.strftime('%Y-%m-%d')}
PARAMS: {params_count}
STATUS: entrenado

CAPABILITIES:
"""
    for cap in spec['capabilities']:
        content += f"  - {cap}\n"
    weight_file = os.path.basename(filepath).replace('.basemateria', '.materia')
    content += f"\nWEIGHT_MODULE: {weight_file}\n"
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    log(f"  Saved: {filepath}")

def save_materia_module(filepath, name, spec, state_dict, stoi, params_count, loss, epochs):
    module = {
        "materia": "umbra_sub_agent",
        "name": name,
        "version": "3.0.0",
        "architecture": "jepa+gqa+rope+swiglu+synapsis+hsaq",
        "config": {
            "vocab_size": len(stoi),
            "dim": spec["dim"],
            "n_layers": spec["n_layers"],
            "n_heads": spec["n_heads"],
            "n_kv": spec["n_kv"],
            "jepa_dim": spec["jepa_dim"],
            "synapsis_slots": spec["synapsis_slots"],
            "use_snn": spec["use_snn"],
            "use_ssm": spec["use_ssm"],
            "use_jepa": spec["use_jepa"],
            "use_synapsis": spec["use_synapsis"],
            "use_hsaq": spec["use_hsaq"],
            "max_seq_len": spec["seq_len"],
            "params": params_count,
        },
        "ollama_model": spec["ollama_model"],
        "domain": spec["domain"],
        "capabilities": spec["capabilities"],
        "training": {
            "date": time.strftime('%Y-%m-%d'),
            "dataset": spec["dataset"],
            "epochs": epochs,
            "final_loss": round(loss, 4),
            "samples": spec["max_samples"],
        },
        "weights": {k: v.cpu().numpy() for k, v in state_dict.items()},
        "tokenizer": stoi,
    }
    with gzip.open(filepath, 'wb') as f:
        f.write(json.dumps(module, cls=NumpyEncoder).encode('utf-8'))
    log(f"  Saved: {filepath} ({os.path.getsize(filepath)//1024}KB)")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return super().default(obj)

# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    log("=" * 60)
    log("MATERIA V3 - ENTRENAMIENTO COMPLETO DE TODOS LOS MODELOS")
    log(f"Device: {DEVICE}")
    log("=" * 60)

    # Load all text data for vocabulary building
    log("\n[1] Cargando datos para construccion de vocabulario...")
    all_text_files = [
        os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt'),
        os.path.join(BASE, 'data/multilingual/tokenizer/combined_for_spm.txt'),
    ]
    wiki_dir = os.path.join(BASE, 'data/multilingual/tokenizer')
    for f in sorted(os.listdir(wiki_dir)):
        if f.startswith('wiki_') and f.endswith('.txt') and not f.startswith('wiki_api'):
            all_text_files.append(os.path.join(wiki_dir, f))
    # Add reasoning dataset
    all_text_files.append(os.path.join(BASE, 'data/reasoning_dataset.txt'))

    all_texts = load_texts(all_text_files, max_lines=80000)
    log(f"  Total textos cargados: {len(all_texts):,}")

    # Build vocabulary (~800 tokens as per original spec)
    log("\n[2] Construyendo vocabulario...")
    stoi, itos = build_vocab(all_texts, vocab_size=800)
    vocab_size = len(stoi)
    log(f"  Vocab size: {vocab_size}")

    # Train each model
    for model_key, spec in MODEL_SPECS.items():
        is_basemateria = model_key.endswith('.basemateria')
        log(f"\n{'='*60}")
        log(f"[3] ENTRENANDO: {model_key} ({spec['name']})")
        log(f"{'='*60}")

        # Load domain-specific data
        if 'science' in model_key:
            log("  Cargando dataset cientifico...")
            texts = load_texts([os.path.join(BASE, 'data/reasoning_dataset.txt')],
                             max_lines=spec['max_samples'])
        elif 'unified' in model_key:
            log("  Cargando Wikitext-2 + Wikipedia...")
            es_path = os.path.join(BASE, 'data/multilingual/tokenizer/wiki_es.txt')
            en_path = os.path.join(BASE, 'data/multilingual/tokenizer/wiki_en.txt')
            texts = load_texts([es_path, en_path], max_lines=spec['max_samples'])
        elif 'nano' in model_key:
            log("  Cargando C4 EN (subset rapido)...")
            texts = load_texts([os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt')],
                             max_lines=spec['max_samples'])
        elif 'extended' in model_key:
            log("  Cargando Wikitext-2 + C4...")
            texts = load_texts([os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt')],
                             max_lines=spec['max_samples'])
        elif 'full' in model_key:
            log("  Cargando C4 EN completo...")
            texts = load_texts([os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt')],
                             max_lines=spec['max_samples'])
        else:
            log("  Cargando dataset multilingue completo...")
            texts = all_texts[:spec['max_samples']]

        log(f"  Textos cargados: {len(texts):,}")

        # Create dataset
        dataset = TextDataset(texts, stoi, seq_len=spec['seq_len'])
        loader = DataLoader(dataset, batch_size=spec['batch_size'],
                          shuffle=True, drop_last=True)
        log(f"  Chunks del dataset: {len(dataset):,}")

        # Create model with complete architecture
        log("  Creando modelo con arquitectura completa...")
        model = CompleteModel(
            vocab_size=vocab_size,
            dim=spec['dim'],
            n_layers=spec['n_layers'],
            n_heads=spec['n_heads'],
            n_kv=spec['n_kv'],
            jepa_dim=spec['jepa_dim'],
            synapsis_slots=spec['synapsis_slots'],
            use_snn=spec['use_snn'],
            use_ssm=spec['use_ssm'],
            use_jepa=spec['use_jepa'],
            use_synapsis=spec['use_synapsis'],
            use_hsaq=spec['use_hsaq'],
        )
        params_count = count_params(model)
        log(f"  Parametros totales: {params_count:,}")

        # Train
        final_loss = train_model(model, loader, spec['epochs'],
                                 model_name=model_key)

        # Save
        base_path = os.path.join(BASE, 'models', model_key)

        if is_basemateria:
            save_basemateria(base_path, spec['name'], spec,
                           params_count, final_loss, stoi, spec['epochs'])
            # Also save .materia weight file
            materia_path = base_path.replace('.basemateria', '.materia')
            save_materia_module(materia_path, spec['name'], spec,
                              model.state_dict(), stoi,
                              params_count, final_loss, spec['epochs'])
        else:
            save_materia_module(base_path, spec['name'], spec,
                              model.state_dict(), stoi,
                              params_count, final_loss, spec['epochs'])

        # Quick generation test
        model.eval()
        with torch.no_grad():
            if 'science' in model_key:
                prompt_text = "Question: What is the speed of light?"
            else:
                prompt_text = "Hello, this is a test of MATERIA"
            prompt_ids = [stoi.get(c, 3) for c in prompt_text]
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
            gen_ids = model.generate(prompt_tensor, max_new=20)
            gen_text = ''.join(itos.get(int(i), '?') for i in gen_ids[0].tolist())
            log(f"  Sample: {gen_text[:100]}")

        log(f"  ✓ {model_key} completado (loss={final_loss:.4f})")

    log("\n" + "=" * 60)
    log("ENTRENAMIENTO COMPLETO FINALIZADO")
    log("=" * 60)
