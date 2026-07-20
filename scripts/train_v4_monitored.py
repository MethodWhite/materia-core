"""
MATERIA V4 - Training supervisado con monitor inteligente
Auto-checkpoint por batches, deteccion de sobreentrenamiento,
extraccion de estado interno, reporte de aprendizaje.
"""
import os, sys, yaml, time, gc, argparse, json, pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from materia_v4 import MateriaV4, count_params
from core.monitor import TrainingMonitor

MATERIA_HOME = os.environ.get('MATERIA_HOME',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..')))

K = 2.781042
log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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
    stoi['<PAD>'] = 0; stoi['<BOS>'] = 1; stoi['<EOS>'] = 2; stoi['<UNK>'] = 3
    return stoi, {i: ch for ch, i in stoi.items()}

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
    return texts

def detect_device(memory_limit=0.80):
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total = props.total_memory / 1024**3
        usable = total * memory_limit
        log(f"GPU: {props.name} ({total:.1f}GB total, {usable:.1f}GB usable)")
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(memory_limit)
        return torch.device('cuda'), usable
    return torch.device('cpu'), 0

def free_memory(device, force=False):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        if force:
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()

def generate_plots(monitor, output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib no instalado, skipping plots")
        return

    s = monitor.stats
    epochs = range(1, len(s['train_loss']) + 1)

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle(f'MATERIA V4 - {monitor.cfg.get("name", "training")}', fontsize=14)

    # 1 - Loss
    ax = axes[0, 0]
    ax.plot(epochs, s['train_loss'], 'b-o', label='Train', markersize=3)
    ax.plot(epochs, s['val_loss'], 'r--s', label='Val', markersize=3)
    ax.set_ylabel('Total Loss')
    ax.set_title('Loss (token + K·jepa)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2 - Token & JEPA loss
    ax = axes[0, 1]
    if s.get('train_tok'):
        ax.plot(epochs, s['train_tok'], 'c-^', label='Token', markersize=3)
    ax.plot(epochs, s['train_jepa'], 'g-v', label='JEPA', markersize=3)
    ax.set_title('Loss Components')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 3 - Accuracy
    ax = axes[0, 2]
    ax.plot(epochs, s['train_acc'], 'b-o', label='Train', markersize=3)
    ax.plot(epochs, s['val_acc'], 'r--s', label='Val', markersize=3)
    ax.set_ylabel('Accuracy')
    ax.set_title('Token Accuracy')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 4 - Overfit gap
    ax = axes[1, 0]
    if s.get('overfit_gap'):
        ax.plot(epochs, s['overfit_gap'], 'm-o', markersize=3)
        ax.axhline(y=-0.2, color='r', linestyle='--', alpha=0.5, label='Sobreentrenamiento')
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax.set_title('Overfit Gap (train-val)')
        ax.legend(); ax.grid(True, alpha=0.3)

    # 5 - Perplexity
    ax = axes[1, 1]
    if s.get('perplexity'):
        ax.plot(epochs, s['perplexity'], 'g-o', markersize=3)
        ax.set_title('Perplexity')
        ax.grid(True, alpha=0.3)

    # 6 - Spectral mu
    ax = axes[1, 2]
    if s.get('spectral_mu'):
        ax.plot(epochs, s['spectral_mu'], 'purple', marker='o', markersize=3)
        ax.set_title(f'Spectral Eigenvalues (∝ K={K})')
        ax.grid(True, alpha=0.3)

    # 7 - SNN Spike Rate
    ax = axes[2, 0]
    if s.get('spike_rate'):
        ax.plot(epochs, s['spike_rate'], 'orange', marker='o', markersize=3)
        ax.set_title('LIF-SNN Spike Rate')
        ax.grid(True, alpha=0.3)

    # 8 - LR Schedule
    ax = axes[2, 1]
    if s.get('lr'):
        ax.plot(epochs, s['lr'], 'm-o', markersize=3)
        ax.set_title('Learning Rate')
        ax.grid(True, alpha=0.3)

    # 9 - Synapsis
    ax = axes[2, 2]
    if s.get('synapsis_usage'):
        ax.plot(epochs, s['synapsis_usage'], 'brown', marker='o', markersize=3)
        ax.set_title('Synapsis Memory Usage (%)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'training_monitor.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"[PLOTS] {path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--resume', '-r', type=str, default=None)
    parser.add_argument('--memory-limit', '-m', type=float, default=0.80)
    parser.add_argument('--batch-size', '-b', type=int, default=None)
    parser.add_argument('--patience', type=int, default=2,
                        help='Early stopping patience (default: 2)')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override max epochs from config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    experiment_name = f"{cfg['name']}-{time.strftime('%Y%m%d_%H%M')}"
    output_dir = args.output or os.path.join(MATERIA_HOME, 'outputs', experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)

    log(f"=== MATERIA V4 - Training Supervisado: {cfg['name']} ===")
    device, usable_mem = detect_device(args.memory_limit)
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
    log(f"Output: {output_dir}")
    log(f"Config: {cfg['description']}")

    # Data
    max_lines = cfg['training'].get('max_lines', 80000)
    data_dir = os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
    data_files = [os.path.join(data_dir, 'c4_en.txt'),
                  os.path.join(data_dir, 'combined_for_spm.txt')]
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.startswith('wiki_') and f.endswith('.txt'):
                data_files.append(os.path.join(data_dir, f))

    log("Loading data...")
    texts = load_text_data(data_files, max_lines)
    free_memory(device)

    vocab_size = cfg['model'].get('vocab_size', 1024)
    stoi, itos = build_char_tokenizer(texts[:10000], vocab_size)
    vocab_size = len(stoi)
    log(f"Texts: {len(texts):,} | Vocab: {vocab_size} tokens")

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

    bs = args.batch_size or cfg['training'].get('batch_size', 4)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, drop_last=False, num_workers=0, pin_memory=True)

    # Model
    log("Creating model...")
    mc = cfg['model']
    with torch.device('cpu'):
        model = MateriaV4(
            vocab_size=vocab_size, dim=mc.get('dim', 256),
            n_layers=mc.get('n_layers', 3), n_heads=mc.get('n_heads', 8),
            n_kv=mc.get('n_kv', 4), latent_dim=mc.get('latent_dim', 256),
            snn_dim=mc.get('snn_dim', 128), snn_threshold=mc.get('snn_threshold', 0.005),
            snn_tau=mc.get('snn_tau', 0.8), ssm_state=mc.get('ssm_state', 32),
            synapsis_slots=mc.get('synapsis_slots', 128),
            hsaq_sparsity=mc.get('hsaq_sparsity', 0.3),
            jepa_weight=mc.get('jepa_weight', K),
        )
    model.to(device)
    total = count_params(model)
    log(f"Params: {total:,} ({total*4/1024**2:.1f}MB)")

    # Monitor
    monitor = TrainingMonitor(model, output_dir, cfg,
                              patience=args.patience)

    # Optimizer
    opt = optim.AdamW(model.parameters(),
                      lr=cfg['training'].get('lr', 5e-4),
                      weight_decay=cfg['training'].get('weight_decay', 0.01),
                      foreach=False)

    epochs = args.max_epochs or cfg['training'].get('epochs', 5)
    total_steps = epochs * len(train_loader)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    use_amp = cfg['training'].get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Resume
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            opt.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', -1) + 1
        log(f"Resumed from epoch {start_epoch}")
        del ckpt
        free_memory(device)

    # Train loop
    grad_accum = cfg['training'].get('grad_accum', 4)
    clip_norm = cfg['training'].get('clip_grad_norm', 1.0)
    jepa_weight = mc.get('jepa_weight', K)

    log(f"\n{'='*60}")
    log(f"INICIANDO ENTRENAMIENTO SUPERVISADO")
    log(f"Modelo: {total:,} params | Data: {len(train_ds):,} chunks")
    log(f"Batch: {bs} | Grad accum: {grad_accum} | Epochs: {epochs}")
    log(f"Patience: {args.patience} | Mixed precision: BF16={use_amp}")
    log(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        log(f"Epoch {epoch+1}/{epochs}")
        model.train()
        total_loss = total_tok = total_jepa = total_acc = 0.0
        n_batches = 0

        for i, (x, y) in enumerate(train_loader):
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    scaler.step(opt)
                    scaler.update()
                    sch.step()
                    opt.zero_grad(set_to_none=True)

                batch_loss = loss.item() * grad_accum
                total_loss += batch_loss
                total_tok += token_loss.item()
                total_jepa += jepa_mse.item()
                total_acc += (logits.argmax(-1) == y).float().mean().item()
                n_batches += 1

                monitor.log_batch({
                    'loss': batch_loss, 'tok': token_loss.item(),
                    'jepa': jepa_mse.item(), 'acc': (logits.argmax(-1) == y).float().mean().item(),
                    'spike': rate,
                })

                if (i + 1) % 50 == 0:
                    log(f"  E{epoch+1} [{i+1}/{len(train_loader)}] "
                        f"loss={batch_loss:.4f} tok={token_loss.item():.4f} "
                        f"jepa={jepa_mse.item():.4f} acc={total_acc/max(1,n_batches):.4f}")

                del logits, loss, x, y

            except torch.cuda.OutOfMemoryError:
                log(f"OOM batch {i}. Liberando...")
                free_memory(device, force=True)
                opt.zero_grad(set_to_none=True)
                continue

        # Validation
        model.eval()
        val_loss = val_jepa = val_acc = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, jepa_mse, _ = model(x)
                tok_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = tok_loss + jepa_weight * jepa_mse
                val_loss += loss.item()
                val_jepa += jepa_mse.item()
                val_acc += (logits.argmax(-1) == y).float().mean().item()
                n_val_batches += 1
                del logits, x, y

        n_val_batches = max(1, n_val_batches)
        avg_val_loss = val_loss / n_val_batches
        avg_val_jepa = val_jepa / n_val_batches
        avg_val_acc = val_acc / n_val_batches

        avg_train_loss = total_loss / max(1, n_batches)
        avg_train_tok = total_tok / max(1, n_batches)
        avg_train_jepa = total_jepa / max(1, n_batches)
        avg_train_acc = total_acc / max(1, n_batches)

        spectral_mu = 0.0
        if hasattr(model, 'jepa_pred') and hasattr(model.jepa_pred, 'mu'):
            spectral_mu = model.jepa_pred.mu.sigmoid().mean().item()

        metrics = {
            'train_loss': avg_train_loss, 'train_tok': avg_train_tok,
            'train_jepa': avg_train_jepa, 'train_acc': avg_train_acc,
            'val_loss': avg_val_loss, 'val_jepa': avg_val_jepa,
            'val_acc': avg_val_acc, 'spike_rate': rate.item() if torch.is_tensor(rate) else rate,
            'spectral_mu': spectral_mu, 'lr': sch.get_last_lr()[0],
        }

        should_stop = monitor.log_epoch(epoch, metrics)

        log(f"  => train loss={avg_train_loss:.4f} val loss={avg_val_loss:.4f} "
            f"acc={avg_val_acc:.4f} ppl={np.exp(min(avg_val_loss, 20)):.2f}")
        status = monitor.get_summary_line()
        log(f"  => {status}")

        # Save checkpoint
        ckpt_path = os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pt')
        model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        opt_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in opt.state_dict().items()}
        torch.save({
            'epoch': epoch, 'model_state_dict': model_cpu,
            'optimizer_state_dict': opt_cpu, 'stats': monitor.stats,
        }, ckpt_path)
        del model_cpu, opt_cpu
        log(f"  Checkpoint: {ckpt_path} ({os.path.getsize(ckpt_path)//1024**2}MB)")

        if should_stop:
            log(f"[PARADA TEMPRANA] Modelo estancado despues de epoch {epoch+1}")
            break

        free_memory(device, force=True)

    # Finalize
    log(f"\n{'='*60}")
    log(f"ENTRENAMIENTO COMPLETADO")
    log(f"{'='*60}")

    report = monitor.generate_report()
    print(report)
    log(f"\nResumen final:")
    log(f"  Fase: {report['learning_status']['phase']}")
    log(f"  Val Loss: {report['final_metrics']['val_loss']:.4f}")
    log(f"  Perplexity: {report['final_metrics']['perplexity']:.2f}")
    log(f"  Necesita mas datos: {report['learning_status']['needs_more_data']}")
    log(f"  Sugerencia: {report['learning_status'].get('suggestion', 'N/A')}")
    log(f"  Estado interno: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in report['internal_state'].items() if isinstance(v, (int, float))}, default=str)}")

    generate_plots(monitor, output_dir)

    # Export weights
    model.cpu()
    free_memory(device, force=True)
    materia_path = os.path.join(output_dir, 'materia-v4.materia')
    weight_data = {
        'config': {'vocab_size': vocab_size, 'dim': model.dim, 'version': 'V4',
                    'latent_dim': model.latent_dim, 'K': K, 'params': total},
        'state_dict': {k: v.numpy() for k, v in model.state_dict().items()},
        'tokenizer': stoi,
        'report': report,
    }
    with open(materia_path, 'wb') as f:
        pickle.dump(weight_data, f)
    log(f"Weights: {materia_path} ({os.path.getsize(materia_path)//1024}KB)")
    log(f"Output: {output_dir}")

if __name__ == '__main__':
    main()
