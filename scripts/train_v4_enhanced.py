"""
MATERIA V4 Enhanced - Training Pipeline con mejoras
- Métricas: perplexity, generación durante entrenamiento
- Regularización: dropout, label smoothing
- LR schedule: warmup + cosine decay + restarts
- Evaluación: genera texto cada N batches para monitorear calidad
"""
import os, sys, yaml, time, pickle, gc, argparse, math
from glob import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import sentencepiece as spm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from materia_v4 import MateriaV4, count_params

MATERIA_HOME = os.environ.get(
    'MATERIA_HOME',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
)

K = 2.781042

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
warn = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─── Memoria ────────────────────────────────────────────────────────────

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


# ─── Métricas mejoradas ──────────────────────────────────────────────────

def compute_perplexity(loss):
    """Calcula perplexity desde cross-entropy loss."""
    return math.exp(min(loss, 20))  # Cap para evitar overflow


def generate_text(model, stoi, itos, prompt, max_new=50, temp=0.8, top_p=0.9, sp=None):
    """Genera texto desde un prompt para evaluación.
    Usa SentencePiece (sp) si está disponible, sino char-level."""
    model.eval()
    if sp is not None:
        ids = sp.EncodeAsIds(prompt)
    else:
        ids = [stoi.get(c, 3) for c in prompt]
    if not ids:
        ids = [stoi.get(' ', 3)] if sp is None else [sp.PieceToId('<unk>')]
    x = torch.tensor([ids], dtype=torch.long)
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        out = model.generate(x, max_new=max_new, temp=temp, top_p=top_p)
    gen_ids = out[0].tolist()
    if sp is not None:
        result = sp.DecodeIds(gen_ids[len(ids):])
    else:
        result = ''.join(itos.get(i, '?') for i in gen_ids[len(ids):])
    model.train()
    return result


def evaluate_generation(model, stoi, itos, device, sp=None):
    """Evalúa calidad de generación con múltiples prompts."""
    prompts = [
        'The meaning of life is',
        'Artificial intelligence',
        'Hello, how are you',
    ]
    results = []
    for prompt in prompts:
        try:
            result = generate_text(model, stoi, itos, prompt, max_new=30, sp=sp)
            # Calidad básica: ¿genera caracteres únicos?
            unique_chars = len(set(result))
            repetition_ratio = 1.0 - (unique_chars / max(1, len(result)))
            results.append({
                'prompt': prompt,
                'output': result,
                'unique_chars': unique_chars,
                'repetition_ratio': repetition_ratio,
            })
        except Exception as e:
            results.append({'prompt': prompt, 'error': str(e)})
    return results


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


class BPETextDataset(Dataset):
    """Dataset que tokeniza texto con SentencePiece (BPE) y concatena textos cortos."""
    def __init__(self, texts, sp, seq_len=64):
        self.seq_len = seq_len
        self.data = []
        # Concatenar textos en un solo flujo con separador
        buffer = []
        for text in texts:
            ids = sp.EncodeAsIds(text)
            if len(ids) > 10:  # Ignorar textos muy cortos
                buffer.extend(ids + [sp.eos_id()] if sp.eos_id() > 0 else ids + [0])
        # Crear chunks con stride
        for i in range(0, len(buffer) - seq_len, seq_len // 2):
            self.data.append(buffer[i:i + seq_len + 1])
        del texts, buffer

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
    return texts


# ─── LR Schedule mejorado ────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Warmup + Cosine Decay + Restarts."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            scale = self.step_count / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * progress))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = max(self.min_lr, base_lr * scale)

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


# ─── Entrenamiento mejorado ──────────────────────────────────────────────

def train_epoch(model, loader, opt, sch, device, epoch, epochs, cfg,
                output_dir=None, stats=None, skip_batches=0, stoi=None, itos=None, sp=None,
                is_main=True):
    total_loss, total_jepa, total_acc = 0.0, 0.0, 0.0
    n_total = 0
    grad_accum = cfg['training'].get('grad_accum', 1)
    log_interval = cfg['logging'].get('log_interval', 25)
    gen_interval = cfg['logging'].get('gen_interval', 500)
    mem_interval = max(100, log_interval * 4)
    batch_save_interval = cfg['logging'].get('batch_save_interval', 0)
    jepa_weight = cfg['model'].get('jepa_weight', 2.781042)
    spike_target = cfg['training'].get('spike_target', 0.30)
    spike_weight = cfg['training'].get('spike_weight', 0.5)
    label_smoothing = cfg['training'].get('label_smoothing', 0.0)
    use_amp = cfg['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    model.train()
    skipped = 0
    epoch_start = time.time()

    for i, (x, y) in enumerate(loader):
        # Skip batches si resume
        if skip_batches > 0 and i < skip_batches:
            if i == 0 and is_main:
                log(f"  Skipping first {skip_batches} batches (resume)...")
            continue

        try:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                logits, jepa_mse, rate = model(x)
                # Label smoothing
                if label_smoothing > 0:
                    vocab_size = logits.size(-1)
                    log_probs = F.log_softmax(logits.view(-1, vocab_size), dim=-1)
                    smooth_loss = -log_probs.mean(dim=-1)
                    token_loss = (1 - label_smoothing) * F.cross_entropy(
                        logits.view(-1, vocab_size), y.view(-1), ignore_index=0
                    ) + label_smoothing * smooth_loss.mean()
                else:
                    token_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0
                    )
                loss = (token_loss + jepa_weight * jepa_mse) / grad_accum
                # Spike regularizer: fuerza al SNN a mantener tasa de disparo
                spike_reg = spike_weight * (rate - spike_target) ** 2
                task_loss = loss  # sin regularizador (para logging)
                loss = loss + spike_reg
                # Clamp total loss para evitar NaN en backward por pérdida extrema
                loss = loss.clamp(max=100.0)
            scaler.scale(loss).backward()
            
            # Escalar gradientes de HSAQ sparsity_logits (gradiente muy atenuado por STE)
            # Escalamiento adaptativo: si logit está en zona muerta, escala más
            for name, p in model.named_parameters():
                if 'sparsity_logit' in name and p.grad is not None:
                    # Gradiente adaptativo: más escala si logit cerca de bordes [-5,5]
                    logit_val = p.data.clamp(-5, 5)
                    dead_zone = 1.0 - (logit_val.abs() / 5.0).mean()
                    scale = 10.0 + dead_zone * 90.0  # 10× en medio, 100× en bordes
                    p.grad.copy_(p.grad.clamp(-0.5, 0.5) * scale)

            if (i + 1) % grad_accum == 0:
                scaler.unscale_(opt)
                # Clamp gradientes antes de check NaN
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg['training'].get('clip_grad_norm', 1.0)
                )
                # Skip step si hay NaN/Inf
                has_nan = any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters())
                if has_nan:
                    if is_main:
                        log(f"  WARNING: NaN/Inf gradiente en batch {i+1}, saltando step + reseteando momentum")
                    opt.zero_grad(set_to_none=True)
                    for group in opt.param_groups:
                        for p in group['params']:
                            if p.grad is not None:
                                state = opt.state.get(p)
                                if state and 'momentum_buffer' in state and state['momentum_buffer'] is not None:
                                    if not torch.isfinite(state['momentum_buffer']).all():
                                        state['momentum_buffer'].zero_()
                    skipped += 1
                    if skipped >= 500:
                        if is_main:
                            log(f"  AUTO-RECUPERACIÓN: {skipped} NaN seguidos, cargando último checkpoint...")
                        break
                    continue
                scaler.step(opt)
                scaler.update()
                sch.step()
                opt.zero_grad(set_to_none=True)

            total_loss += task_loss.item()
            total_jepa += jepa_mse.item()
            acc = (logits.argmax(-1) == y).float().mean().item()
            total_acc += acc * x.size(0)

            # Log con métricas mejoradas
            if is_main and (i + 1) % log_interval == 0:
                avg_loss = total_loss / (i + 1)
                perplexity = compute_perplexity(avg_loss)
                lr = sch.get_last_lr()[0] if hasattr(sch, 'get_last_lr') else opt.param_groups[0]['lr']
                # HSAQ metrics (usar primer HSAQ disponible como referencia global)
                hsaq_ref = getattr(model, 'hsaq_emb', None) or getattr(model, 'hsaq_jepa_in', None)
                hsaq_stats = hsaq_ref.get_stats() if hsaq_ref is not None else {}
                hsaq_sparse = hsaq_stats.get('actual_sparsity', 0)
                hsaq_target = hsaq_stats.get('sparsity_scale', 0)
                hsaq_thresh = hsaq_stats.get('threshold', 0)
                # Per-layer HSAQ stats (cada N muestras)
                hsaq_per_layer = ""
                if hasattr(model, '_hsaq_log') and model._hsaq_log and (i + 1) % (log_interval * 4) == 0:
                    hsaq_per_layer = " | " + " ".join(
                        f"{name}:{s.get('actual_sparsity',0):.2f}"
                        for name, s in model._hsaq_log
                    )
                log(f"  E{epoch+1}/{epochs} [{i+1}/{len(loader)}] "
                    f"loss={task_loss.item():.4f} tok={token_loss.item():.4f} "
                    f"jepa={jepa_mse.item():.6f} acc={acc:.4f} "
                    f"ppl={perplexity:.2f} spike={rate:.4f} "
                    f"spk_reg={spike_reg:.4f} "
                    f"hsaq_sp={hsaq_sparse:.3f}(tgt={hsaq_target:.3f}) th={hsaq_thresh:.4f}"
                    f"{hsaq_per_layer} lr={lr:.2e}")

            # Generar texto para evaluación
            if is_main and gen_interval > 0 and stoi and itos and (i + 1) % gen_interval == 0:
                gen_results = evaluate_generation(model, stoi, itos, device, sp=sp)
                for r in gen_results:
                    if 'error' not in r:
                        log(f"  [GEN] '{r['prompt']}' → '{r['output'][:50]}' "
                            f"(unique={r['unique_chars']}, repeat={r['repetition_ratio']:.2f})")

            del logits, loss, x, y

            if (i + 1) % mem_interval == 0:
                free_memory(device)

            # Batch checkpoint (solo rank 0)
            if is_main and batch_save_interval > 0 and output_dir and stats and (i + 1) % batch_save_interval == 0:
                save_batch_checkpoint(model, opt, epoch, i + 1, stats, output_dir)
                cleanup_batch_checkpoints(output_dir, keep_last=2)

        except torch.cuda.OutOfMemoryError:
            if is_main:
                warn(f"CUDA OOM en batch {i}. Liberando memoria...")
            free_memory(device, force=True)
            if is_main:
                warn(f"Gradient accumulation reducido a {grad_accum}")
            opt.zero_grad(set_to_none=True)
            continue

    if (i + 1) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg['training'].get('clip_grad_norm', 1.0)
        )
        opt.step()
        opt.zero_grad(set_to_none=True)

    epoch_time = time.time() - epoch_start
    n = len(loader.dataset)
    log(f"  Epoch time: {epoch_time/60:.1f} min")
    return total_loss / max(1, n), total_jepa / max(1, len(loader)), total_acc / max(1, n)


# ─── Validación mejorada ─────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, device, world_size=1):
    model.eval()
    total_loss, total_jepa, total_acc, n = 0.0, 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, jepa_mse, rate = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
        acc = (logits.argmax(-1) == y).float().mean()
        total_loss += loss.item() * x.size(0)
        total_jepa += jepa_mse.item()
        total_acc += acc.item() * x.size(0)
        n += x.size(0)
        del logits, loss, x, y
    model.train()
    # Sync DDP: sum metrics across all GPUs
    if world_size > 1 and dist.is_initialized():
        t = torch.tensor([total_loss, total_jepa, total_acc, n], device=device)
        dist.all_reduce(t)
        total_loss, total_jepa, total_acc, n = t.tolist()
    return total_loss / max(1, n), total_jepa / max(1, len(loader)), total_acc / max(1, n)


# ─── Checkpoints ─────────────────────────────────────────────────────────

def save_checkpoint(model, opt, epoch, stats, output_dir):
    path = os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pt')
    # Guardado directo sin copia explícita a CPU (torch.save maneja GPU→CPU internamente)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'stats': stats,
    }, path, pickle_protocol=4, _use_new_zipfile_serialization=True)
    size = os.path.getsize(path) // 1024**2
    log(f"  Checkpoint: {path} ({size}MB)")


def save_batch_checkpoint(model, opt, epoch, batch, stats, output_dir):
    path = os.path.join(output_dir, f'batch_ckpt_{batch}.pt')
    torch.save({
        'epoch': epoch,
        'batch': batch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'stats': stats,
    }, path, pickle_protocol=4, _use_new_zipfile_serialization=True)
    size = os.path.getsize(path) // 1024**2
    log(f"  Batch checkpoint: {path} ({size}MB)")


def cleanup_batch_checkpoints(output_dir, keep_last=2):
    ckpts = sorted(glob(os.path.join(output_dir, 'batch_ckpt_*.pt')))
    if len(ckpts) > keep_last:
        for old in ckpts[:-keep_last]:
            os.remove(old)
            log(f"  Cleaned: {old}")


def save_weights(model, stoi, vocab_size, total, final_stats, output_dir,
                 sp_model_path=None):
    model.cpu()
    free_memory(model.tok_emb.weight.device, force=True)
    materia_path = os.path.join(output_dir, 'materia-v4.materia')
    weight_data = {
        'config': {'vocab_size': vocab_size, 'dim': model.dim, 'version': 'V4-enhanced',
                    'latent_dim': model.latent_dim, 'K': 2.781042},
        'state_dict': {k: v.numpy() for k, v in model.state_dict().items()},
        'tokenizer': stoi,
        'tokenizer_type': 'bpe' if sp_model_path else 'char',
        'stats': final_stats,
    }
    if sp_model_path and os.path.exists(sp_model_path):
        with open(sp_model_path, 'rb') as fsp:
            weight_data['sp_model_bytes'] = fsp.read()
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
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend(); axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(epochs, stats['train_jepa'], 'g-', label='JEPA')
        if stats.get('val_jepa'):
            axes[0, 1].plot(epochs, stats['val_jepa'], 'r--', label='Val JEPA')
        axes[0, 1].set_ylabel('JEPA Loss')
        axes[0, 1].set_title(f'JEPA Loss (K={K})')
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

        if stats.get('perplexity'):
            axes[1, 1].plot(epochs, stats['perplexity'], 'purple')
            axes[1, 1].set_ylabel('Perplexity')
            axes[1, 1].set_title('Perplexity (lower = better)')
            axes[1, 1].grid(True, alpha=0.3)

        if stats.get('lr'):
            axes[1, 2].plot(epochs, stats['lr'], 'm-')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_title('LR Schedule')
            axes[1, 2].grid(True, alpha=0.3)

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
    # ── DDP init ──
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    is_main = local_rank == 0
    if world_size > 1:
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device, _ = detect_device(0.85)
    log = lambda msg: is_main and print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    warn = lambda msg: is_main and print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--max-lines', type=int, default=None)
    parser.add_argument('--data', '-d', type=str, default=None)
    parser.add_argument('--memory-limit', '-m', type=float, default=0.80)
    parser.add_argument('--batch-size', '-b', type=int, default=None)
    parser.add_argument('--no-synapsis', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    experiment_name = f"{cfg['name']}-{time.strftime('%Y%m%d_%H%M')}"
    output_dir = args.output or os.path.join(MATERIA_HOME, 'outputs', experiment_name)
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
            yaml.dump(cfg, f)

    if is_main:
        log(f"=== MATERIA V4 Enhanced - Train: {cfg['name']} ===")
        log(f"GPUs: {world_size} | Output: {output_dir}")
        log_memory('init')

    # ── Data ──
    max_lines = args.max_lines or cfg['training'].get('max_lines', 80000)
    data_dir = args.data or os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
    data_files = [os.path.join(data_dir, 'c4_en.txt'),
                  os.path.join(data_dir, 'combined_for_spm.txt')]
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.startswith('wiki_') and f.endswith('.txt'):
                data_files.append(os.path.join(data_dir, f))

    if is_main:
        log("Loading data...")
    texts = load_text_data(data_files, max_lines)

    tokenizer_type = cfg['tokenizer'].get('type', 'char')
    sp_model = None
    sp_model_path = None
    if tokenizer_type == 'bpe':
        sp_model_path = cfg['tokenizer'].get('sp_model',
                          os.path.join(MATERIA_HOME, 'models', 'materia_multilingual_v2.model'))
        if is_main:
            log(f"Loading SentencePiece model: {sp_model_path}")
        sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
        stoi = {sp_model.IdToPiece(i): i for i in range(sp_model.GetPieceSize())}
        itos = {i: sp_model.IdToPiece(i) for i in range(sp_model.GetPieceSize())}
        vocab_size = sp_model.GetPieceSize()
        if is_main:
            log(f"Texts: {len(texts):,} | Vocab: {vocab_size} BPE tokens (type=bpe)")
    else:
        stoi, itos = build_char_tokenizer(texts, cfg['tokenizer'].get('vocab_size', 1024))
        vocab_size = len(stoi)
        if is_main:
            log(f"Texts: {len(texts):,} | Vocab: {vocab_size} tokens (type=char)")

    val_split = cfg['training'].get('val_split', 0.1)
    n_val = max(1, int(len(texts) * val_split))
    train_texts = texts[n_val:]
    val_texts = texts[:n_val]
    del texts; free_memory(device)

    seq_len = cfg['training'].get('seq_len', 64)
    if tokenizer_type == 'bpe':
        train_ds = BPETextDataset(train_texts, sp_model, seq_len)
        val_ds = BPETextDataset(val_texts, sp_model, seq_len)
    else:
        train_ds = CharTextDataset(train_texts, stoi, seq_len)
        val_ds = CharTextDataset(val_texts, stoi, seq_len)
    del train_texts, val_texts; free_memory(device)
    if is_main:
        log(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} chunks")

    bs = args.batch_size or cfg['training'].get('batch_size', 8)
    num_workers = cfg['training'].get('num_workers', 0)
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=local_rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(train_ds, batch_size=bs, sampler=train_sampler, shuffle=(train_sampler is None),
                              drop_last=True, num_workers=num_workers, pin_memory=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=local_rank, shuffle=False) if world_size > 1 else None
    val_loader = DataLoader(val_ds, batch_size=bs, sampler=val_sampler, shuffle=False,
                            drop_last=False, num_workers=num_workers, pin_memory=True)
    # Loader completo sin DDP para validación en rank 0 (evita deadlock all_reduce)
    val_loader_full = DataLoader(val_ds, batch_size=bs, shuffle=False,
                                 drop_last=False, num_workers=min(2, num_workers), pin_memory=True)

    # ── Model ──
    if is_main:
        log("Creating MATERIA V4 Enhanced model...")
    mc = cfg['model']
    use_synapsis = not args.no_synapsis
    if not use_synapsis:
        log("Synapsis DISABLED")
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
            jepa_weight=K,
            n_cycles=mc.get('n_cycles', 3),
            use_synapsis=use_synapsis,
            use_checkpointing=mc.get('use_checkpointing', False),
            use_flash=mc.get('use_flash', False),
        )
    model.to(device)
    # DDP wrap
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    total = count_params(model.module if world_size > 1 else model)
    if is_main:
        log(f"Params: {total:,} | Synapsis: {use_synapsis}")
        log_memory('model-loaded')

    # ── Optimizer (SGD con momentum = HSAQ optimizer) ──
    raw_model = model.module if world_size > 1 else model
    opt = optim.SGD(
        raw_model.parameters(),
        lr=cfg['training'].get('lr', 5e-4),
        momentum=0.9,
        weight_decay=cfg['training'].get('weight_decay', 0.01),
        nesterov=True,
    )
    if is_main:
        log(f"HSAQ optimizer: SGD Nesterov, lr={cfg['training'].get('lr', 5e-4)}")
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    epochs = cfg['training'].get('epochs', 10)
    total_steps = epochs * len(train_loader)
    warmup_steps = cfg['training'].get('warmup_steps', 100)
    sch = WarmupCosineScheduler(opt, warmup_steps, total_steps)

    # ── Resume ──
    start_epoch = 0
    skip_batches = 0
    if args.resume and os.path.exists(args.resume):
        if is_main:
            log("Loading checkpoint for resume...")
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        raw_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        if 'batch' in ckpt and ckpt['batch'] > 0:
            skip_batches = ckpt['batch']
            if is_main:
                log(f"Resumed from epoch {start_epoch+1}, batch {skip_batches}")
        else:
            start_epoch += 1
            if is_main:
                log(f"Resumed from epoch {start_epoch}")
        if hasattr(raw_model, 'snn') and hasattr(raw_model, 'snn_to_jepa'):
            def _reset_linear(m):
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
            raw_model.snn.apply(_reset_linear)
            raw_model.snn_to_jepa.apply(_reset_linear)
            with torch.no_grad():
                if hasattr(raw_model.snn, 'alpha'):
                    raw_model.snn.alpha.fill_(0.3)
                if hasattr(raw_model.snn, 'w_in'):
                    raw_model.snn.w_in.weight.mul_(5.0)
            if is_main:
                log("  SNN path + alpha reinitialized (w_in scale=5×, alpha=0.3)")
        del ckpt
        free_memory(device)
    # Broadcast parámetros a todas las GPUs post-resume
    if world_size > 1:
        for param in raw_model.parameters():
            dist.broadcast(param.data, src=0)
        dist.barrier()
        if is_main:
            log("  DDP broadcast post-resume completed")

    # ── Stats ──
    stats = {
        'train_loss': [], 'train_jepa': [], 'train_acc': [],
        'val_loss': [], 'val_jepa': [], 'val_acc': [],
        'spike_rate': [], 'spectral_mu': [], 'lr': [],
        'perplexity': [], 'gen_samples': [],
    }

    best_val_loss = float('inf')
    patience = cfg['training'].get('early_stopping_patience', 5)
    no_improve = 0

    # ── Train ──
    for epoch in range(start_epoch, epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if is_main:
            log(f"Epoch {epoch+1}/{epochs}")
            log_memory(f'epoch-{epoch+1}-start')

        train_loss, train_jepa, train_acc = train_epoch(
            raw_model if world_size > 1 else model, train_loader, opt, sch, device, epoch, epochs, cfg,
            output_dir=output_dir if is_main else None, stats=stats if is_main else None,
            skip_batches=skip_batches if epoch == start_epoch else 0,
            stoi=stoi if is_main else None, itos=itos if is_main else None, sp=sp_model,
            is_main=is_main,
        )
        skip_batches = 0
        free_memory(device)

        # Barrier: sincronizar antes de guardar checkpoint
        if world_size > 1:
            dist.barrier()

        if is_main:
            spike_rate = raw_model.last_spike_rate.item() if hasattr(raw_model, 'last_spike_rate') else 0
            stats['train_loss'].append(train_loss)
            stats['train_jepa'].append(train_jepa)
            stats['train_acc'].append(train_acc)
            stats['spike_rate'].append(spike_rate)
            stats['lr'].append(sch.get_last_lr()[0])

            log(f"  => epoch={epoch+1}/{epochs} loss={train_loss:.4f} acc={train_acc:.4f} "
                f"spike={spike_rate:.4f} lr={sch.get_last_lr()[0]:.2e}")
            log_memory(f'epoch-{epoch+1}-end')

            log("  Saving checkpoint...")
            save_checkpoint(raw_model, opt, epoch, stats, output_dir)

        # Barrier: todas esperan que termine el checkpoint
        if world_size > 1:
            dist.barrier()

        free_memory(device, force=True)

    # ── Finalize (solo rank 0) ──
    if is_main:
        if cfg['logging'].get('plot_metrics', True):
            generate_plots(stats, output_dir)

        save_weights(raw_model, stoi, vocab_size, total, stats, output_dir, sp_model_path=sp_model_path)

        csv_path = os.path.join(output_dir, 'training_log.csv')
        with open(csv_path, 'w') as f:
            f.write('epoch,train_loss,train_acc,val_loss,val_acc,spike_rate,lr,perplexity\n')
            for i in range(len(stats['train_loss'])):
                f.write(f"{i+1},{stats['train_loss'][i]:.6f},{stats['train_acc'][i]:.6f},"
                        f"{stats['val_loss'][i]:.6f},{stats['val_acc'][i]:.6f},"
                        f"{stats['spike_rate'][i]:.6f},{stats['lr'][i]:.6e},"
                        f"{stats['perplexity'][i]:.4f}\n")
        log(f"CSV: {csv_path}")
        log_memory('final')

        log(f"\nTraining complete!")
        log(f"  Model: {cfg['name']} ({total:,} params)")
        log(f"  Final val_loss: {stats['val_loss'][-1]:.6f}")
        log(f"  Final val_acc: {stats['val_acc'][-1]:.4f}")
        log(f"  Final perplexity: {stats['perplexity'][-1]:.2f}")
        log(f"  Output: {output_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
