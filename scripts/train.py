"""
MATERIA V3 - Universal Training Pipeline
Uso: python scripts/train.py --config configs/100M.yaml

Lee config YAML, construye modelo, entrena, genera plots.
Funciona headless en cualquier cloud (RunPod, Modal, Colab, etc).

MEMORIA: monitoreo activo de RAM/VRAM, liberacion automatica,
manejo graceful de OOM con reduccion de batch_size.
"""
import os, sys, yaml, time, pickle, gc, argparse, signal, warnings
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from materia_v3_full import MateriaV3Full, count_params

MATERIA_HOME = os.environ.get(
    'MATERIA_HOME',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
)

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
warn = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] ⚠ {msg}", flush=True)


# ─── Memoria ────────────────────────────────────────────────────────────

def get_memory_usage():
    """Retorna (ram_gb, vram_gb, ram_pct) actual."""
    ram = psutil_process() or {}
    ram_gb = ram.get('rss', 0) / 1024**3
    ram_pct = ram.get('percent', 0)
    vram_gb = 0
    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / 1024**3
    return ram_gb, vram_gb, ram_pct


def psutil_process():
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info()._asdict() | {'percent': proc.memory_percent()}
    except ImportError:
        return None


def log_memory(tag=''):
    ram_gb, vram_gb, ram_pct = get_memory_usage()
    vram_str = f" | VRAM: {vram_gb:.2f}GB" if vram_gb > 0 else ""
    log(f"[MEM]{' ' + tag if tag else ''} RAM: {ram_gb:.2f}GB ({ram_pct:.1f}%){vram_str}")


def free_memory(device, force=False):
    """Libera memoria caché y recolecta basura."""
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        if force:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()


def memory_safe_batch_size(model, loader, device, start_bs, max_retries=3):
    """Prueba el batch_size, reduce si hay OOM. Retorna batch_size seguro."""
    bs = start_bs
    for attempt in range(max_retries):
        try:
            test_x, test_y = next(iter(loader))
            test_x = test_x[:bs].to(device)
            test_y = test_y[:bs].to(device)
            with torch.no_grad():
                model(test_x)
            del test_x, test_y
            free_memory(device)
            return bs
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if 'out of memory' in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                old_bs = bs
                bs = max(1, bs // 2)
                warn(f"OOM con batch_size={old_bs}, reduciendo a {bs}")
                free_memory(device, force=True)
            else:
                raise
    warn(f"No se pudo determinar batch_size seguro, usando {bs}")
    return bs


# ─── Deteccion de dispositivo ────────────────────────────────────────────

def detect_device(memory_limit=0.85):
    """Detecta GPU y calcula memoria usable.
    memory_limit: fraccion de la memoria total a usar (0.0-1.0).
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        usable = total * memory_limit
        log(f"GPU: {props.name} ({total:.1f}GB total, {usable:.1f}GB usable)")
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(memory_limit)
        return torch.device('cuda'), usable
    import psutil
    mem = psutil.virtual_memory()
    usable = mem.available / 1024**3 * memory_limit
    cores = max(1, os.cpu_count() - 1)
    torch.set_num_threads(cores)
    log(f"CPU: {cores} threads, {mem.available/1024**3:.1f}GB RAM disponible, {usable:.1f}GB usable")
    return torch.device('cpu'), usable


# ─── Tokenizer ────────────────────────────────────────────────────────────

class TextDatasetSeq(Dataset):
    """Dataset para tokenizer BPE/SentencePiece."""
    def __init__(self, texts, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.data = []
        for text in texts:
            ids = tokenizer.encode(text)
            if len(ids) > seq_len + 1:
                for i in range(0, len(ids) - seq_len, seq_len // 2):
                    self.data.append(ids[i:i + seq_len + 1])
        del texts

    def __len__(self):
        return max(1, len(self.data))

    def __getitem__(self, idx):
        ids = self.data[idx % len(self.data)]
        ids = torch.tensor(ids[:self.seq_len + 1], dtype=torch.long)
        if len(ids) < self.seq_len + 1:
            pad = torch.full((self.seq_len + 1 - len(ids),),
                             self.tokenizer.pad_id, dtype=torch.long)
            ids = torch.cat([ids, pad])
        return ids[:-1], ids[1:]


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


# ─── Dataset ────────────────────────────────────────────────────────────

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


def load_text_data(filepaths, max_lines=80000):
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
        log(f"  {fp}: {len(texts)} lines")
        if len(texts) > 200000:
            log_memory('post-file-load')
            gc.collect()
    return texts


# ─── Tokenizer BPE / HF ───────────────────────────────────────────────

class BPETokenizer:
    def __init__(self, model_path=None):
        import sentencepiece as spm
        if model_path is None:
            base = os.path.join(MATERIA_HOME, 'data/multilingual/tokenizer')
            model_path = os.path.join(base, 'materia_multilingual_v2.model')
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text):
        return self.sp.EncodeAsIds(text)

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()

    @property
    def pad_id(self):
        return self.sp.pad_id() if self.sp.pad_id() >= 0 else 0

    @property
    def unk_id(self):
        return self.sp.unk_id()


def load_hf_dataset(dataset_spec, max_lines=500_000, split='train'):
    """Carga dataset desde HuggingFace en modo streaming.
    dataset_spec: 'hf:org/dataset' o 'hf:org/dataset:config'
    """
    from datasets import load_dataset
    parts = dataset_spec.split(':')
    path = parts[1]
    config = parts[2] if len(parts) > 2 else None

    log(f"Loading HF dataset: {path}" + (f' ({config})' if config else ''))
    dataset = load_dataset(path, config, split=split, streaming=True)

    texts = []
    for i, example in enumerate(dataset):
        text = example.get('text', example.get('content', ''))
        if isinstance(text, str) and len(text.strip()) > 50:
            texts.append(text.strip())
            if len(texts) >= max_lines:
                break
        if (i + 1) % 100000 == 0:
            log(f"  {i+1} items, {len(texts)} saved")
            gc.collect()

    log(f"HF dataset loaded: {len(texts)} texts")
    return texts


def select_tokenizer(cfg):
    """Retorna tokenizer segun config. Busca .model disponible si no encuentra el path exacto."""
    tok_type = cfg.get('tokenizer', {}).get('type', 'char')
    if tok_type == 'bpe':
        model_path = cfg.get('tokenizer', {}).get('model_path')
        if model_path and not os.path.isabs(model_path):
            model_path = os.path.join(MATERIA_HOME, model_path)
        if not os.path.exists(model_path or ''):
            # Buscar cualquier .model disponible
            tok_dir = os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
            models = sorted(glob(os.path.join(tok_dir, '*.model')))
            if models:
                model_path = models[-1]
                log(f"BPE model not found at configured path, using {model_path}")
            else:
                log(f"No BPE model found. Train with: python scripts/prepare_data.py --train-tokenizer")
                log(f"Falling back to char-level tokenizer")
                return None, None, None
        return BPETokenizer(model_path), None, None
    return None, None, None  # char-level: returns (None, stoi, itos)


# ─── Training ────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        acc = (logits.argmax(-1) == y).float().mean()
        total_loss += loss.item() * x.size(0)
        total_acc += acc.item() * x.size(0)
        n += x.size(0)
        del logits, loss, x, y
    model.train()
    return total_loss / max(1, n), total_acc / max(1, n)


def train_epoch(model, loader, opt, sch, device, epoch, epochs, cfg):
    total_loss, total_acc = 0.0, 0.0
    grad_accum = cfg['training'].get('grad_accum', 1)
    log_interval = cfg['logging'].get('log_interval', 25)
    mem_interval = max(100, log_interval * 4)
    model.train()

    for i, (x, y) in enumerate(loader):
        try:
            x, y = x.to(device), y.to(device)
            logits, rate = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0
            )
            loss = loss / grad_accum
            loss.backward()

            if (i + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['training'].get('clip_grad_norm', 1.0)
                )
                opt.step()
                opt.zero_grad(set_to_none=True)

            sch.step()
            total_loss += loss.item() * grad_accum
            acc = (logits.argmax(-1) == y).float().mean().item()
            total_acc += acc * x.size(0)

            if (i + 1) % log_interval == 0:
                sp = (logits.detach() == 0).float().mean().item()
                log(f"  E{epoch+1}/{epochs} [{i+1}/{len(loader)}] "
                    f"loss={loss.item()*grad_accum:.4f} acc={acc:.4f} "
                    f"spike={rate:.4f} sparse={sp:.3f}")

            del logits, loss, x, y

            if (i + 1) % mem_interval == 0:
                free_memory(device)

        except torch.cuda.OutOfMemoryError:
            warn(f"CUDA OOM en batch {i}. Liberando memoria...")
            free_memory(device, force=True)
            grad_accum = max(1, grad_accum // 2)
            warn(f"Gradient accumulation reducido a {grad_accum}")
            opt.zero_grad(set_to_none=True)
            continue

    if (i + 1) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg['training'].get('clip_grad_norm', 1.0)
        )
        opt.step()
        opt.zero_grad(set_to_none=True)

    n = len(loader.dataset)
    return total_loss / max(1, n), total_acc / max(1, n)


# ─── Checkpoint / Save ────────────────────────────────────────────────────

def save_checkpoint(model, opt, epoch, stats, output_dir):
    path = os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pt')
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    opt_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in opt.state_dict().items()}
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_cpu,
        'optimizer_state_dict': opt_cpu,
        'stats': stats,
    }, path)
    del model_cpu, opt_cpu
    log(f"  Checkpoint: {path} ({os.path.getsize(path)//1024**2}MB)")
    return path


def save_weights(model, stoi, vocab_size, total, final_stats, output_dir, tokenizer=None):
    model.cpu()
    free_memory(model.tok_emb.weight.device, force=True)
    materia_path = os.path.join(output_dir, 'materia-v3.materia')
    tok_data = stoi if not tokenizer else {
        'type': 'bpe',
        'vocab_size': tokenizer.vocab_size,
        'pad_id': tokenizer.pad_id,
    }
    weight_data = {
        'config': {'vocab_size': vocab_size, 'dim': model.dim, 'snn': 'lif_real'},
        'state_dict': {k: v.numpy() for k, v in model.state_dict().items()},
        'tokenizer': tok_data,
        'stats': final_stats,
    }
    with open(materia_path, 'wb') as f:
        pickle.dump(weight_data, f)
    log(f"Weights: {materia_path} ({os.path.getsize(materia_path)//1024}KB)")


# ─── Plots ────────────────────────────────────────────────────────────────

def generate_plots(stats, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        epochs = range(1, len(stats['train_loss']) + 1)

        axes[0, 0].plot(epochs, stats['train_loss'], 'b-', label='Train')
        if stats.get('val_loss'):
            axes[0, 0].plot(epochs, stats['val_loss'], 'r--', label='Val')
        axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].grid(True)

        axes[0, 1].plot(epochs, stats['train_acc'], 'b-', label='Train')
        if stats.get('val_acc'):
            axes[0, 1].plot(epochs, stats['val_acc'], 'r--', label='Val')
        axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy'); axes[0, 1].legend(); axes[0, 1].grid(True)

        if stats.get('spike_rate'):
            axes[1, 0].plot(epochs, stats['spike_rate'], 'g-')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Spike Rate')
            axes[1, 0].set_title('LIF Neuron Activity')
            axes[1, 0].grid(True)

        if stats.get('lr'):
            axes[1, 1].plot(epochs, stats['lr'], 'm-')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_title('LR Schedule')
            axes[1, 1].grid(True)

        plt.tight_layout()
        path = os.path.join(output_dir, 'training_curves.png')
        plt.savefig(path, dpi=150)
        plt.close()
        log(f"  Plots: {path}")
    except ImportError:
        log("  matplotlib no instalado, skipping plots")
    except Exception as e:
        log(f"  Plot error: {e}")


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--data', '-d', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset HF: hf:allenai/c4, hf:allenai/c4:en')
    parser.add_argument('--max-lines', type=int, default=None,
                        help='Max lineas a cargar')
    parser.add_argument('--memory-limit', '-m', type=float, default=0.80,
                        help='Fraccion de memoria a usar (0.0-1.0, default: 0.80)')
    parser.add_argument('--batch-size', '-b', type=int, default=None,
                        help='Override batch_size del config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    experiment_name = f"{cfg['name']}-{time.strftime('%Y%m%d_%H%M')}"
    output_dir = args.output or os.path.join(MATERIA_HOME, 'outputs', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    log(f"=== MATERIA V3 - Train: {cfg['name']} ===")
    device, usable_mem = detect_device(args.memory_limit)
    log(f"Output: {output_dir}")
    log(f"Config: {cfg['description']} ({cfg['params_m']}M params)")
    log_memory('init')

    # ── Data ──
    max_lines = args.max_lines or cfg['training'].get('max_lines', 80000)
    tok_type = cfg.get('tokenizer', {}).get('type', 'char')

    if args.dataset and args.dataset.startswith('hf:'):
        ds_spec = args.dataset
        # Si no incluye config, agregar desde YAML
        if ds_spec.count(':') < 2:
            ds_config = cfg.get('dataset', {}).get('config', '')
            if ds_config:
                ds_spec = f"{ds_spec}:{ds_config}"
        texts = load_hf_dataset(ds_spec, max_lines=max_lines)
        if tok_type == 'bpe':
            tokenizer, stoi, itos = select_tokenizer(cfg)
            vocab_size = tokenizer.vocab_size
        else:
            stoi, itos = build_char_tokenizer(texts, cfg['model'].get('vocab_size', 1024))
            vocab_size = len(stoi)
            tokenizer = None
    else:
        data_dir = args.data or os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
        data_files = [os.path.join(data_dir, 'c4_en.txt'),
                      os.path.join(data_dir, 'combined_for_spm.txt')]
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.startswith('wiki_') and f.endswith('.txt'):
                    data_files.append(os.path.join(data_dir, f))
        log("Loading local data...")
        texts = load_text_data(data_files, max_lines)
        if tok_type == 'bpe':
            tokenizer, stoi, itos = select_tokenizer(cfg)
            vocab_size = tokenizer.vocab_size
        else:
            stoi, itos = build_char_tokenizer(texts[:10000],
                                               cfg['model'].get('vocab_size', 1024))
            vocab_size = len(stoi)
            tokenizer = None

    log(f"Texts: {len(texts):,} | Vocab: {vocab_size} tokens | Tokenizer: {tok_type}")
    free_memory(device)

    # Split
    val_split = cfg['training'].get('val_split', 0.1)
    n_val = max(1, int(len(texts) * val_split))
    train_texts = texts[n_val:]
    val_texts = texts[:n_val]
    del texts; free_memory(device)

    seq_len = cfg['training'].get('seq_len', 64)
    if tokenizer:
        from materia_v3_full import TextDatasetSeq
        train_ds = TextDatasetSeq(train_texts, tokenizer, seq_len)
        val_ds = TextDatasetSeq(val_texts, tokenizer, seq_len)
    else:
        train_ds = CharTextDataset(train_texts, stoi, seq_len)
        val_ds = CharTextDataset(val_texts, stoi, seq_len)
    del train_texts, val_texts; free_memory(device)
    log(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} chunks")

    bs = args.batch_size or cfg['training'].get('batch_size', 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                               drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                             drop_last=False, num_workers=0)

    # ── Model ──
    log("Creating model...")
    with torch.device('cpu'):
        model = MateriaV3Full(
            vocab_size=vocab_size,
            dim=cfg['model'].get('dim', 256),
            n_layers=cfg['model'].get('n_layers', 3),
            n_heads=cfg['model'].get('n_heads', 8),
            n_kv=cfg['model'].get('n_kv', 4),
            synapsis_slots=cfg['model'].get('synapsis_slots', 128),
            hsaq_sparsity=cfg['model'].get('hsaq_sparsity', 0.3),
        )
    model.to(device)
    total = count_params(model)
    model_mb = total * (4 if model.tok_emb.weight.dtype == torch.float32 else 2) / 1024**2
    log(f"Params: {total:,} ({model_mb:.1f}MB)")

    # Safe batch
    safe_bs = memory_safe_batch_size(model, train_loader, device, bs)
    if safe_bs != bs:
        warn(f"Batch_size ajustado: {bs} → {safe_bs}")
        bs = safe_bs
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                   drop_last=True, num_workers=0)
    log_memory('model-loaded')

    # ── Optimizer ──
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg['training'].get('lr', 5e-4),
        weight_decay=cfg['training'].get('weight_decay', 0.01),
        foreach=False,
    )

    epochs = cfg['training'].get('epochs', 8)
    total_steps = epochs * len(train_loader)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        log(f"Resumed from epoch {start_epoch}")
        del ckpt; free_memory(device)

    # ── Train ──
    stats = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'spike_rate': [], 'lr': [],
    }

    for epoch in range(start_epoch, epochs):
        log(f"Epoch {epoch+1}/{epochs}")
        log_memory(f'epoch-{epoch+1}-start')

        train_loss, train_acc = train_epoch(
            model, train_loader, opt, sch, device, epoch, epochs, cfg
        )
        free_memory(device)

        val_loss, val_acc = validate(model, val_loader, device)

        stats['train_loss'].append(train_loss)
        stats['train_acc'].append(train_acc)
        stats['val_loss'].append(val_loss)
        stats['val_acc'].append(val_acc)
        stats['spike_rate'].append(
            model.spike_rate if hasattr(model, 'spike_rate') else 0
        )
        stats['lr'].append(sch.get_last_lr()[0])

        log(f"  => train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        log_memory(f'epoch-{epoch+1}-end')

        if (epoch + 1) % cfg['logging'].get('save_interval', 1) == 0:
            save_checkpoint(model, opt, epoch, stats, output_dir)

        free_memory(device, force=True)

    # ── Finalize ──
    if cfg['logging'].get('plot_metrics', True):
        generate_plots(stats, output_dir)

    tok_obj = tokenizer if tok_type == 'bpe' else None
    save_weights(model, stoi, vocab_size, total, stats, output_dir, tokenizer=tok_obj)

    csv_path = os.path.join(output_dir, 'training_log.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,train_acc,val_loss,val_acc,spike_rate,lr\n')
        for i in range(len(stats['train_loss'])):
            f.write(f"{i+1},{stats['train_loss'][i]:.6f},{stats['train_acc'][i]:.6f},"
                    f"{stats['val_loss'][i]:.6f},{stats['val_acc'][i]:.6f},"
                    f"{stats['spike_rate'][i]:.6f},{stats['lr'][i]:.6e}\n")
    log(f"CSV: {csv_path}")
    log_memory('final')

    log(f"\n✓ Training complete!")
    log(f"  Model: {cfg['name']} ({total:,} params)")
    log(f"  Final val_loss: {stats['val_loss'][-1]:.6f}")
    log(f"  Final val_acc: {stats['val_acc'][-1]:.4f}")
    log(f"  Output: {output_dir}")


if __name__ == '__main__':
    main()
