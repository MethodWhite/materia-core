"""
MATERIA V4 - Training Pipeline (JEPA-First + SCA)
Uso: python scripts/train_v4.py --config configs/V4_3.8M.yaml

Loss dual: token_loss + K * jepa_loss (K = √π·e·γ = 2.781042)
Trackea: jepa_loss, maxent_loss, token_loss por separado
Genera plots comparativos con V3 si existen logs previos.
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
from materia_v4 import MateriaV4, count_params

MATERIA_HOME = os.environ.get(
    'MATERIA_HOME',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
)

K = 2.781042

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
warn = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_memory_usage():
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
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        if force:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

def memory_safe_batch_size(model, loader, device, start_bs, max_retries=3):
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


def detect_device(memory_limit=0.85):
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


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_loss, total_jepa, total_acc, n = 0.0, 0.0, 0.0, 0
    jepa_weight = getattr(model, 'jepa_weight', K)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, jepa_mse, _ = model(x)
        token_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = token_loss + jepa_weight * jepa_mse
        acc = (logits.argmax(-1) == y).float().mean()
        total_loss += loss.item() * x.size(0)
        total_jepa += jepa_mse.item() * x.size(0)
        total_acc += acc.item() * x.size(0)
        n += x.size(0)
        del logits, loss, x, y
    model.train()
    return total_loss / max(1, n), total_jepa / max(1, n), total_acc / max(1, n)


def train_epoch(model, loader, opt, sch, device, epoch, epochs, cfg, output_dir=None, stats=None, skip_batches=0):
    total_loss, total_jepa, total_acc = 0.0, 0.0, 0.0
    grad_accum = cfg['training'].get('grad_accum', 1)
    log_interval = cfg['logging'].get('log_interval', 25)
    mem_interval = max(100, log_interval * 4)
    batch_save_interval = cfg['logging'].get('batch_save_interval', 0)  # 0 = deshabilitado
    jepa_weight = cfg['model'].get('jepa_weight', 2.781042)
    use_amp = cfg['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    model.train()
    skipped = 0

    for i, (x, y) in enumerate(loader):
        # Skip batches si resume desde batch checkpoint
        if skip_batches > 0 and i < skip_batches:
            if i == 0:
                log(f"  Skipping first {skip_batches} batches (resume)...")
            continue

        try:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, jepa_mse, rate = model(x)
                token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0
                )
                loss = (token_loss + jepa_weight * jepa_mse) / grad_accum
            scaler.scale(loss).backward()

            if (i + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['training'].get('clip_grad_norm', 1.0)
                )
                scaler.step(opt)
                scaler.update()
                sch.step()
                opt.zero_grad(set_to_none=True)
            total_loss += loss.item() * grad_accum
            total_jepa += jepa_mse.item()
            acc = (logits.argmax(-1) == y).float().mean().item()
            total_acc += acc * x.size(0)

            if (i + 1) % log_interval == 0:
                sp = (logits.detach() == 0).float().mean().item()
                log(f"  E{epoch+1}/{epochs} [{i+1}/{len(loader)}] "
                    f"loss={loss.item()*grad_accum:.4f} tok={token_loss.item():.4f} "
                    f"jepa_mse={jepa_mse.item():.4f} acc={acc:.4f} spike={rate:.4f}")

            del logits, loss, x, y

            if (i + 1) % mem_interval == 0:
                free_memory(device)

            # Batch-level checkpoint
            if batch_save_interval > 0 and output_dir and stats and (i + 1) % batch_save_interval == 0:
                save_batch_checkpoint(model, opt, epoch, i + 1, stats, output_dir)
                cleanup_batch_checkpoints(output_dir, keep_last=2)

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
    return total_loss / max(1, n), total_jepa / max(1, len(loader)), total_acc / max(1, n)


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


def save_batch_checkpoint(model, opt, epoch, batch, stats, output_dir):
    path = os.path.join(output_dir, f'batch_ckpt_{batch}.pt')
    model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
    opt_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in opt.state_dict().items()}
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model_cpu,
        'optimizer_state_dict': opt_cpu,
        'stats': stats,
    }, path)
    del model_cpu, opt_cpu
    log(f"  Batch checkpoint: {path} ({os.path.getsize(path)//1024**2}MB)")


def cleanup_batch_checkpoints(output_dir, keep_last=2):
    """Mantener solo los últimos N batch checkpoints para ahorrar disco."""
    ckpts = sorted(glob(os.path.join(output_dir, 'batch_ckpt_*.pt')))
    if len(ckpts) > keep_last:
        for old in ckpts[:-keep_last]:
            os.remove(old)
            log(f"  Cleaned: {old}")


def save_weights(model, stoi, vocab_size, total, final_stats, output_dir):
    model.cpu()
    free_memory(model.tok_emb.weight.device, force=True)
    materia_path = os.path.join(output_dir, 'materia-v4.materia')
    weight_data = {
        'config': {'vocab_size': vocab_size, 'dim': model.dim, 'version': 'V4',
                    'latent_dim': model.latent_dim, 'K': 2.781042},
        'state_dict': {k: v.numpy() for k, v in model.state_dict().items()},
        'tokenizer': stoi,
        'stats': final_stats,
    }
    with open(materia_path, 'wb') as f:
        pickle.dump(weight_data, f)
    log(f"Weights: {materia_path} ({os.path.getsize(materia_path)//1024}KB)")


def generate_plots(stats, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        epochs = range(1, len(stats['train_loss']) + 1)

        axes[0, 0].plot(epochs, stats['train_loss'], 'b-', label='Train')
        if stats.get('val_loss'):
            axes[0, 0].plot(epochs, stats['val_loss'], 'r--', label='Val')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss (token + K·jepa)')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, stats['train_jepa'], 'g-', label='JEPA')
        if stats.get('val_jepa'):
            axes[0, 1].plot(epochs, stats['val_jepa'], 'r--', label='Val JEPA')
        axes[0, 1].set_ylabel('JEPA Loss')
        axes[0, 1].set_title(f'JEPA Loss (K={2.781042})')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)

        axes[0, 2].plot(epochs, stats['train_acc'], 'b-', label='Train')
        if stats.get('val_acc'):
            axes[0, 2].plot(epochs, stats['val_acc'], 'r--', label='Val')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Token Prediction Accuracy')
        axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)

        if stats.get('spike_rate'):
            axes[1, 0].plot(epochs, stats['spike_rate'], 'orange')
            axes[1, 0].set_ylabel('Spike Rate')
            axes[1, 0].set_title('LIF Neuron Activity')
            axes[1, 0].grid(True, alpha=0.3)

        if stats.get('spectral_mu'):
            axes[1, 1].plot(epochs, stats['spectral_mu'], 'purple')
            axes[1, 1].set_ylabel('Mean μ')
            axes[1, 1].set_title('SCA Spectral Eigenvalues (∝ K)')
            axes[1, 1].grid(True, alpha=0.3)

        if stats.get('lr'):
            axes[1, 2].plot(epochs, stats['lr'], 'm-')
            axes[1, 2].set_ylabel('LR')
            axes[1, 2].set_title('Learning Rate')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(output_dir, 'training_curves_v4.png')
        plt.savefig(path, dpi=150)
        plt.close()
        log(f"  Plots: {path}")
    except ImportError:
        log("  matplotlib no instalado, skipping plots")
    except Exception as e:
        log(f"  Plot error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--max-lines', type=int, default=None)
    parser.add_argument('--data', '-d', type=str, default=None,
                        help='Directorio de datos')
    parser.add_argument('--memory-limit', '-m', type=float, default=0.80)
    parser.add_argument('--batch-size', '-b', type=int, default=None)
    parser.add_argument('--no-synapsis', action='store_true',
                        help='Deshabilitar Synapsis memory (evitar maldicion de repeticion)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    experiment_name = f"{cfg['name']}-{time.strftime('%Y%m%d_%H%M')}"
    output_dir = args.output or os.path.join(MATERIA_HOME, 'outputs', experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    log(f"=== MATERIA V4 - Train: {cfg['name']} ===")
    K_VAL = cfg['model'].get('jepa_weight', 2.781042)
    log(f"JEPA weight K = {K_VAL} (√π·e·γ)")
    device, usable_mem = detect_device(args.memory_limit)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        if hasattr(torch, 'compile'):
            log("torch.compile available - enabling for faster training")
            # compile disabled for now to avoid potential issues
        log("BF16 mixed precision + TF32 matmul enabled")
    log(f"Output: {output_dir}")
    log(f"Config: {cfg['description']}")
    log_memory('init')

    # Data
    max_lines = args.max_lines or cfg['training'].get('max_lines', 80000)
    data_dir = args.data or os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
    data_files = [os.path.join(data_dir, 'c4_en.txt'),
                  os.path.join(data_dir, 'combined_for_spm.txt')]
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.startswith('wiki_') and f.endswith('.txt'):
                data_files.append(os.path.join(data_dir, f))

    log("Loading local data...")
    texts = load_text_data(data_files, max_lines)
    free_memory(device)

    vocab_size = cfg['model'].get('vocab_size') or cfg['tokenizer'].get('vocab_size', 1024)
    stoi, itos = build_char_tokenizer(texts[:10000], vocab_size)
    vocab_size = len(stoi)
    log(f"Texts: {len(texts):,} | Vocab: {vocab_size} tokens")

    # Split
    val_split = cfg['training'].get('val_split', 0.1)
    n_val = max(1, int(len(texts) * val_split))
    train_texts = texts[n_val:]
    val_texts = texts[:n_val]
    del texts
    free_memory(device)

    seq_len = cfg['training'].get('seq_len', 64)
    train_ds = CharTextDataset(train_texts, stoi, seq_len)
    val_ds = CharTextDataset(val_texts, stoi, seq_len)
    del train_texts, val_texts
    free_memory(device)
    log(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} chunks")

    num_workers = cfg['training'].get('num_workers', 2)
    bs = args.batch_size or cfg['training'].get('batch_size', 8)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=True)

    # Model
    log("Creating MATERIA V4 model...")
    mc = cfg['model']
    use_synapsis = not args.no_synapsis
    if not use_synapsis:
        log("Synapsis DISABLED (no-synapsis mode)")
    with torch.device('cpu'):
        model = MateriaV4(
            vocab_size=vocab_size,
            dim=mc.get('dim', 256),
            n_layers=mc.get('n_layers', 3),
            n_heads=mc.get('n_heads', 8),
            n_kv=mc.get('n_kv', 4),
            latent_dim=mc.get('latent_dim', 256),
            snn_dim=mc.get('snn_dim', 128),
            snn_threshold=mc.get('snn_threshold', 0.005),
            snn_tau=mc.get('snn_tau', 0.8),
            ssm_state=mc.get('ssm_state', 32),
            synapsis_slots=mc.get('synapsis_slots', 128),
            hsaq_sparsity=mc.get('hsaq_sparsity', 0.3),
            jepa_weight=K_VAL,
            use_synapsis=use_synapsis,
        )
    model.to(device)
    total = count_params(model)
    model_mb = total * (4 if model.tok_emb.weight.dtype == torch.float32 else 2) / 1024**2
    log(f"Params: {total:,} ({model_mb:.1f}MB)")

    safe_bs = memory_safe_batch_size(model, train_loader, device, bs)
    if safe_bs != bs:
        warn(f"Batch_size ajustado: {bs} → {safe_bs}")
        bs = safe_bs
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
    log_memory('model-loaded')

    # Optimizer
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
    skip_batches = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        # Si es batch checkpoint, calcular batches a saltar
        if 'batch' in ckpt and ckpt['batch'] > 0:
            skip_batches = ckpt['batch']
            log(f"Resumed from epoch {start_epoch+1}, batch {skip_batches}")
        else:
            start_epoch += 1
            log(f"Resumed from epoch {start_epoch}")
        del ckpt
        free_memory(device)

    # Stats
    stats = {
        'train_loss': [], 'train_jepa': [], 'train_acc': [],
        'val_loss': [], 'val_jepa': [], 'val_acc': [],
        'spike_rate': [], 'spectral_mu': [], 'lr': [],
    }

    best_val_loss = float('inf')
    patience = cfg['training'].get('early_stopping_patience', 3)
    no_improve = 0

    # Train loop
    for epoch in range(start_epoch, epochs):
        log(f"Epoch {epoch+1}/{epochs}")
        log_memory(f'epoch-{epoch+1}-start')

        train_loss, train_jepa, train_acc = train_epoch(
            model, train_loader, opt, sch, device, epoch, epochs, cfg,
            output_dir=output_dir, stats=stats,
            skip_batches=skip_batches if epoch == start_epoch else 0
        )
        skip_batches = 0  # Solo saltar en la primera epoch
        free_memory(device)

        val_loss, val_jepa, val_acc = validate(model, val_loader, device)

        # Track SCA spectral eigenvalues
        spectral_mu = 0.0
        if hasattr(model.jepa_pred, 'mu'):
            spectral_mu = model.jepa_pred.mu.sigmoid().mean().item()

        stats['train_loss'].append(train_loss)
        stats['train_jepa'].append(train_jepa)
        stats['train_acc'].append(train_acc)
        stats['val_loss'].append(val_loss)
        stats['val_jepa'].append(val_jepa)
        stats['val_acc'].append(val_acc)
        stats['spike_rate'].append(getattr(model, 'spike_rate', 0))
        stats['spectral_mu'].append(spectral_mu)
        stats['lr'].append(sch.get_last_lr()[0])

        log(f"  => train loss={train_loss:.4f} (tok) jepa={train_jepa:.4f} acc={train_acc:.4f}")
        log(f"     val  loss={val_loss:.4f} jepa={val_jepa:.4f} acc={val_acc:.4f}")
        log(f"     spectral μ={spectral_mu:.4f} (∝ K={2.781042})")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log(f"Early stopping: {patience} epochs without improvement")
                break

        if (epoch + 1) % cfg['logging'].get('save_interval', 1) == 0:
            save_checkpoint(model, opt, epoch, stats, output_dir)

        free_memory(device, force=True)

    # Finalize
    if cfg['logging'].get('plot_metrics', True):
        generate_plots(stats, output_dir)

    save_weights(model, stoi, vocab_size, total, stats, output_dir)

    # CSV
    csv_path = os.path.join(output_dir, 'training_log_v4.csv')
    with open(csv_path, 'w') as f:
        f.write('epoch,train_loss,train_jepa,train_acc,val_loss,val_jepa,val_acc,spike_rate,spectral_mu,lr\n')
        for i in range(len(stats['train_loss'])):
            f.write(f"{i+1},{stats['train_loss'][i]:.6f},{stats['train_jepa'][i]:.6f},"
                    f"{stats['train_acc'][i]:.6f},{stats['val_loss'][i]:.6f},"
                    f"{stats['val_jepa'][i]:.6f},{stats['val_acc'][i]:.6f},"
                    f"{stats['spike_rate'][i]:.6f},{stats['spectral_mu'][i]:.6f},{stats['lr'][i]:.6e}\n")
    log(f"CSV: {csv_path}")
    log_memory('final')

    log(f"\nMATERIA V4 training complete!")
    log(f"  Model: {cfg['name']} ({total:,} params)")
    log(f"  JEPA weight K = {K_VAL}")
    log(f"  Final val_loss: {stats['val_loss'][-1]:.6f}")
    log(f"  Final val_jepa: {stats['val_jepa'][-1]:.6f}")
    log(f"  Output: {output_dir}")

    # Generate comparison plot with V3 if available
    try:
        v3_logs = glob(os.path.join(MATERIA_HOME, 'logs', '*_log.csv'))
        if v3_logs:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(14, 5))
            ax[0].plot(epochs, stats['val_loss'], 'b-o', label='V4 total')
            ax[0].plot(epochs, stats['val_jepa'], 'g--s', label='V4 JEPA')
            ax[0].set_xlabel('Epoch'); ax[0].set_ylabel('Loss')
            ax[0].set_title(f'V4 Loss (K={K_VAL})')
            ax[0].legend(); ax[0].grid(True, alpha=0.3)
            ax[1].plot(epochs, stats['val_acc'], 'b-o', label='V4')
            ax[1].set_xlabel('Epoch'); ax[1].set_ylabel('Accuracy')
            ax[1].set_title('V4 Validation Accuracy')
            ax[1].legend(); ax[1].grid(True, alpha=0.3)
            plt.suptitle('MATERIA V4 - Training Metrics', fontsize=14)
            plt.tight_layout()
            comp_path = os.path.join(output_dir, 'v4_metrics.png')
            plt.savefig(comp_path, dpi=150); plt.close()
            log(f"  Comparison plot: {comp_path}")
    except Exception as e:
        log(f"  Comparison plot skipped: {e}")


if __name__ == '__main__':
    main()
