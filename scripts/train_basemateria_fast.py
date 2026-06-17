"""
MATERIA V3 - Entrenamiento rapido del BaseModel (materia-v3.basemateria)
Version ligera para CPU/4GB-VRAM: modelo 128-dim + datos reducidos + gradient accum.

Usa MateriaV3Full con config reducida (no MiniMateria duplicado).
"""
import os, sys, time, pickle, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from materia_v3_full import MateriaV3Full, count_params

BASE = '/home/methodwhite/MATERIA'

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.cuda.set_per_process_memory_fraction(0.9)
else:
    DEVICE = torch.device('cpu')
    torch.set_num_threads(max(1, os.cpu_count() - 1))

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_char_tokenizer(texts, vocab_size=1024):
    chars = set()
    for t in texts:
        chars.update(t)
    chars = sorted(chars)[:vocab_size - 4]
    stoi = {ch: i + 4 for i, ch in enumerate(chars)}
    stoi['<PAD>'] = 0
    stoi['<BOS>'] = 1
    stoi['<EOS>'] = 2
    stoi['<UNK>'] = 3
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


class CharTextDataset(Dataset):
    def __init__(self, texts, stoi, seq_len=64):
        self.seq_len = seq_len
        self.data = []
        for text in texts:
            ids = [stoi.get(c, 3) for c in text]
            if len(ids) > seq_len + 1:
                for i in range(0, len(ids) - seq_len, seq_len // 2):
                    self.data.append(ids[i:i + seq_len + 1])
        del texts

    def __len__(self):
        return max(1, len(self.data))

    def __getitem__(self, idx):
        ids = self.data[idx % len(self.data)][:self.seq_len + 1]
        ids = ids + [0] * (self.seq_len + 1 - len(ids))
        ids = torch.tensor(ids[:self.seq_len + 1], dtype=torch.long)
        return ids[:-1], ids[1:]


if __name__ == '__main__':
    log(f"=== MATERIA V3 - BaseModel Training (rapido) ===")
    log(f"Device: {DEVICE}")

    models_dir = os.path.join(BASE, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # --- Load small sample ---
    data_path = os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt')
    log(f"Loading data from {data_path}...")
    texts = []
    with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if len(line) > 50:
                texts.append(line)
                if len(texts) >= 3000:
                    break
    log(f"Loaded {len(texts):,} texts")

    stoi, itos = build_char_tokenizer(texts, vocab_size=1024)
    vocab_size = len(stoi)
    log(f"Vocab size: {vocab_size}")

    dataset = CharTextDataset(texts, stoi, seq_len=64)
    del texts
    gc.collect()
    log(f"Dataset chunks: {len(dataset):,}")

    loader = DataLoader(
        dataset, batch_size=16, shuffle=True, drop_last=True,
        num_workers=0,
    )

    # --- Modelo: MateriaV3Full con dim reducido (no MiniMateria) ---
    with torch.device('cpu'):
        model = MateriaV3Full(
            vocab_size=vocab_size,
            dim=128,
            n_layers=2,
            n_heads=4,
            n_kv=2,
            synapsis_slots=64,
        )
    model.to(DEVICE)
    total = count_params(model)
    log(f"Model params: {total:,} ({total*4/1024**2:.1f}MB fp32)")

    # --- Train ---
    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    epochs = 5
    total_steps = epochs * len(loader)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            logits, rate = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                log(f"  E{epoch+1}/{epochs} [{i+1}/{len(loader)}] "
                    f"loss={loss.item():.4f} spike={rate:.3f}")
            del logits, loss, x, y
        avg_loss = total_loss / len(loader)
        log(f"  -> Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log(f"Training complete! Final avg loss: {avg_loss:.4f}")

    # --- Save ---
    materia_path = os.path.join(models_dir, 'materia-v3.materia')
    weight_data = {
        'config': {
            'vocab_size': vocab_size,
            'dim': 128,
            'n_layers': 2,
            'n_heads': 4,
            'n_kv': 2,
            'ffn_size': 512,
            'max_seq_len': 64,
            'jepa_dim': 128,
            'synapsis_slots': 64,
            'hsaq_sparse_threshold': 0.01,
            'hsaq_delta_encode': True,
            'params': total,
            'snn': 'lif_real',
        },
        'state_dict': {k: v.cpu().numpy() for k, v in model.state_dict().items()},
        'tokenizer': stoi,
    }
    with open(materia_path, 'wb') as f:
        pickle.dump(weight_data, f)
    log(f"Weight module: {materia_path} ({os.path.getsize(materia_path)//1024}KB)")

    basemateria_path = os.path.join(models_dir, 'materia-v3.basemateria')
    content = f"""# M.A.T.E.R.I.A. V3 - Base Model
# SNN: LIF real (Leaky Integrate-and-Fire, no sigmoid)
# Entrenado: {time.strftime('%Y-%m-%d %H:%M')}
# Parametros: {total:,}
# Layers: 2 | Hidden: 128 | Heads: 4 | KV: 2
# Vocab: {vocab_size} (char-level)
# JEPA dim: 128 | Synapsis: 64 slots | HSAQ: 0.3 sparsity
# Contexto max: 64 | Device: {DEVICE}
# Dataset: C4 EN (3000 textos, ~150K chars)
# Epochs: {epochs} | Final loss: {avg_loss:.4f}

OLLAMA_MODEL: materia-v3:latest
DOMAIN: general
ARCH: gqa+rope+swiglu+lif_snn+ssm+jepa+synapsis+hsaq
TRAINING_DATE: {time.strftime('%Y-%m-%d')}
PARAMS: {total}
STATUS: entrenado
"""
    with open(basemateria_path, 'w', encoding='utf-8') as f:
        f.write(content)
    log(f".basemateria: {basemateria_path}")

    # --- Quick generation test ---
    model.eval()
    with torch.no_grad():
        prompt = "Hello, this is a test"
        ids = torch.tensor([[stoi.get(c, 3) for c in prompt]], dtype=torch.long)
        for _ in range(30):
            logits, _ = model(ids[:, -64:])
            p = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            ids = torch.cat([ids, torch.multinomial(p, 1)], dim=1)
        generated = ''.join(itos.get(i, '?') for i in ids[0].tolist())
        log(f"Sample generation: {generated[:100]}")

    log("\n✓ Training complete!")
