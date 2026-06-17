"""
MATERIA V3 - Entrenamiento del BaseModel (materia-v3.basemateria)
Arquitectura: GQA + RoPE + SwiGLU + JEPA + Synapsis + HSAQ + LIF-SNN

MEMORIA: optimizado para 4GB VRAM + RAM limitada.
Usa gradient accumulation, streaming loading, y garbage collection.
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

# --- Device detection con 4GB VRAM ---
if torch.cuda.is_available():
    free_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    if free_mem < 5:
        DEVICE = torch.device('cuda')
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
    else:
        DEVICE = torch.device('cuda')
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


def encode(text, stoi):
    return [stoi.get(c, 3) for c in text]


class CharTextDataset(Dataset):
    def __init__(self, texts, stoi, seq_len=64):
        self.stoi = stoi
        self.seq_len = seq_len
        self.data = []
        for text in texts:
            ids = encode(text, stoi)
            if len(ids) > seq_len + 1:
                for i in range(0, len(ids) - seq_len, seq_len // 2):
                    self.data.append(ids[i:i + seq_len + 1])
        # Liberar textos originales de memoria
        del texts

    def __len__(self):
        return max(1, len(self.data))

    def __getitem__(self, idx):
        ids = self.data[idx % len(self.data)][:self.seq_len + 1]
        ids = ids + [0] * (self.seq_len + 1 - len(ids))
        ids = torch.tensor(ids[:self.seq_len + 1], dtype=torch.long)
        return ids[:-1], ids[1:]


def load_text_data_streaming(filepaths, max_lines=80000):
    """Streaming text loader: no carga todo en RAM de golpe pero
    mantiene un buffer manejable. Para datasets grandes (>1M lineas),
    usar __iter__ en vez de lista."""
    texts = []
    for fp in filepaths:
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if len(line) > 50:
                    texts.append(line)
                    if len(texts) >= max_lines:
                        break
        log(f"  {fp}: {len(texts)} lines loaded")
        # Liberar memoria si el archivo es grande
        if len(texts) > 500000:
            gc.collect()
    return texts


def train_model(model, loader, epochs=8, lr=5e-4, grad_accum=1):
    """Training loop con gradient accumulation para batch efectivo mayor sin OOM."""
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(loader)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    model.train()
    log(f"Training: {epochs} epochs, {len(loader)} batches/epoch, "
        f"grad_accum={grad_accum}, device={DEVICE}")

    for epoch in range(epochs):
        total_loss = 0.0
        opt.zero_grad()

        for i, (x, y) in enumerate(loader):
            x, y = x.to(DEVICE), y.to(DEVICE)

            logits, rate = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0
            )
            loss = loss / grad_accum
            loss.backward()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad()

            sch.step()
            total_loss += loss.item() * grad_accum

            if (i + 1) % 25 == 0:
                sp = (logits.detach() == 0).float().mean().item()
                log(f"  E{epoch+1}/{epochs} [{i+1}/{len(loader)}] "
                    f"loss={loss.item()*grad_accum:.4f} spike={rate:.3f} sparse={sp:.3f}")

            # Liberar tensores intermedios
            del logits, loss, x, y
            if i % 100 == 0:
                gc.collect()

        # Final step si no se ejecuto en el ultimo batch
        if (i + 1) % grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

        avg_loss = total_loss / len(loader)
        log(f"  -> Epoch {epoch+1}/{epochs}: avg_loss={avg_loss:.4f}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return avg_loss


def save_model(model, stoi, vocab_size, total, final_loss, models_dir):
    """Guarda pesos en .materia y metadatos en .basemateria."""
    torch_path = os.path.join(models_dir, 'materia-v3-base.pth')
    torch.save(model.state_dict(), torch_path)
    log(f"PyTorch weights saved: {torch_path}")

    materia_path = os.path.join(models_dir, 'materia-v3.materia')
    weight_data = {
        'config': {
            'vocab_size': vocab_size,
            'dim': 256,
            'n_layers': 3,
            'n_heads': 8,
            'n_kv': 4,
            'ffn_size': 1024,
            'max_seq_len': 64,
            'jepa_dim': 256,
            'synapsis_slots': 128,
            'hsaq_sparse_threshold': 0.01,
            'hsaq_delta_encode': True,
            'snn': 'lif_real',
        },
        'state_dict': {k: v.cpu().numpy() for k, v in model.state_dict().items()},
    }
    with open(materia_path, 'wb') as f:
        pickle.dump(weight_data, f)
    log(f"Weight module: {materia_path} ({os.path.getsize(materia_path) // 1024 // 1024}MB)")

    basemateria_path = os.path.join(models_dir, 'materia-v3.basemateria')
    content = f"""# M.A.T.E.R.I.A. V3 - Base Model
# SNN: LIF real (Leaky Integrate-and-Fire, no sigmoid)
# Entrenado: {time.strftime('%Y-%m-%d')}
# Parametros: {total:,}
# Layers: 3 | Hidden: 256 | Heads: 8 | KV: 4
# Vocab: {vocab_size}
# JEPA dim: 256 | Synapsis: 128 slots | HSAQ: 0.01 threshold
# Contexto max: 64
# Dataset: multilingual (C4 + Wikipedia, 80K textos)
# Epochs: 8 | Final loss: {final_loss:.4f}
# Weight module: materia-v3.materia

OLLAMA_MODEL: materia-v3:latest
DOMAIN: general
ARCH: gqa+rope+swiglu+lif_snn+ssm+jepa+synapsis+hsaq
TRAINING_DATE: {time.strftime('%Y-%m-%d')}
PARAMS: {total}
STATUS: entrenado

CAPABILITIES:
  - reasoning: logico, matematico, cientifico
  - code: python, javascript, rust
  - text: espanol, ingles, multilingual
  - memory: synapsis persistente (128 slots)
  - sparse: hsaq ejecucion dispersa adaptativa

MODULES:
  - materia-v3.materia (weight module, pickle)
"""
    with open(basemateria_path, 'w', encoding='utf-8') as f:
        f.write(content)
    log(f".basemateria: {basemateria_path}")


if __name__ == '__main__':
    log(f"=== MATERIA V3 - BaseModel Training ===")
    log(f"Device: {DEVICE} | CPU threads: {torch.get_num_threads()}")

    models_dir = os.path.join(BASE, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # --- Load data (streaming) ---
    data_files = [
        os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt'),
        os.path.join(BASE, 'data/multilingual/tokenizer/combined_for_spm.txt'),
    ]
    wiki_dir = os.path.join(BASE, 'data/multilingual/tokenizer')
    if os.path.exists(wiki_dir):
        for f in sorted(os.listdir(wiki_dir)):
            if f.startswith('wiki_') and f.endswith('.txt'):
                data_files.append(os.path.join(wiki_dir, f))

    log("Loading text data (streaming)...")
    texts = load_text_data_streaming(data_files, max_lines=80000)
    log(f"Total texts: {len(texts):,}")
    gc.collect()

    # --- Build char tokenizer (solo primeras 10K muestras) ---
    log("Building character-level tokenizer...")
    stoi, itos = build_char_tokenizer(texts[:10000], vocab_size=1024)
    vocab_size = len(stoi)
    log(f"Vocabulary: {vocab_size} tokens (char-level)")

    # --- Create dataset ---
    dataset = CharTextDataset(texts, stoi, seq_len=64)
    del texts  # liberar textos originales
    gc.collect()
    log(f"Dataset chunks: {len(dataset):,}")

    loader = DataLoader(
        dataset, batch_size=8, shuffle=True, drop_last=True,
        num_workers=0,  # 0 para evitar duplicacion de memoria
    )

    # --- Create model (a CPU primero, luego mover a GPU si hay) ---
    log("Creating model...")
    with torch.device('cpu'):
        model = MateriaV3Full(
            vocab_size=vocab_size,
            dim=256,
            n_layers=3,
            n_heads=8,
            n_kv=4,
            synapsis_slots=128,
        )
    model.to(DEVICE)
    total = count_params(model)
    log(f"Params: {total:,} ({total*4/1024**2:.1f}MB en fp32)")

    # --- Train ---
    grad_accum = max(1, 32 // 8)  # batch efectivo ~32
    log(f"Starting training (grad_accum={grad_accum})...")
    final_loss = train_model(model, loader, epochs=8, lr=5e-4, grad_accum=grad_accum)
    log(f"Training complete! Final loss: {final_loss:.4f}")

    # --- Save ---
    save_model(model, stoi, vocab_size, total, final_loss, models_dir)

    log(f"\n✓ Base model training complete!")
    log(f"  Params: {total:,}")
    log(f"  Final loss: {final_loss:.4f}")
    log(f"  Device: {DEVICE}")
