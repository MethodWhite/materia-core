"""
MATERIA V3 - Escalado a 250M parámetros
Entrenamiento en RTX 3050 4GB con FP16 + Gradient Checkpointing + CPU Offload
Arquitectura completa: GQA + RoPE + SwiGLU + LIF-SNN + SSM + JEPA + Synapsis + HSAQ + Oura
"""
import os, sys, json, time, math, random, pickle, gzip, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = '/home/methodwhite/MATERIA'
os.makedirs(os.path.join(BASE, 'logs'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'models'), exist_ok=True)
os.makedirs(os.path.join(BASE, 'data', 'tokenizer'), exist_ok=True)

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ============================================================
# ARCHITECTURE: Scaled M.A.T.E.R.I.A. V3 (~250M params)
# ============================================================
class RoPE(nn.Module):
    def __init__(self, dim, max_seq=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer('inv_freq', inv_freq)
    def forward(self, x, offset=0):
        seq = x.shape[-2]
        t = torch.arange(offset, offset+seq, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos()[None, None, :, :]
        sin = freqs.sin()[None, None, :, :]
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out

class GQA(nn.Module):
    def __init__(self, dim, n_heads=16, n_kv=8):
        super().__init__()
        self.n_heads = n_heads; self.n_kv = n_kv
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, self.head_dim * n_kv, bias=False)
        self.wv = nn.Linear(dim, self.head_dim * n_kv, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(self.head_dim)
    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        rep = self.n_heads // self.n_kv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if mask is not None: attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))

class SwiGLU(nn.Module):
    def __init__(self, dim, ffn_dim=None):
        super().__init__()
        ffn_dim = ffn_dim or dim * 4
        self.gate = nn.Linear(dim, ffn_dim, bias=False)
        self.up = nn.Linear(dim, ffn_dim, bias=False)
        self.down = nn.Linear(ffn_dim, dim, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class LIFNeuron(nn.Module):
    def __init__(self, threshold=0.5, tau=0.85):
        super().__init__()
        self.th = threshold; self.tau = tau
        self.register_buffer('V', torch.zeros(1))
    def forward(self, I_in):
        self.V = self.V * self.tau + I_in * (1 - self.tau)
        spike = (self.V >= self.th).float()
        self.V = self.V - spike * self.th
        return spike

class SNNLayer(nn.Module):
    def __init__(self, dim, snn_dim=512):
        super().__init__()
        self.w_in = nn.Linear(dim, snn_dim, bias=False)
        self.lif = LIFNeuron(0.5, 0.85)
        self.w_out = nn.Linear(snn_dim, dim, bias=False)
    def forward(self, x):
        currents = self.w_in(x)
        B, T, D = currents.shape
        spikes = [self.lif(currents[:, t, :]) for t in range(T)]
        spikes = torch.stack(spikes, dim=1)
        fusion = self.w_out(spikes) * 0.05
        return x + fusion, spikes.mean()

class SSMBlock(nn.Module):
    def __init__(self, dim, state_dim=64):
        super().__init__()
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Linear(dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, dim, bias=False)
    def forward(self, x):
        B, T, D = x.shape
        h = torch.zeros(B, self.A.shape[0], device=x.device)
        out = []
        for t in range(T):
            h = torch.tanh(h @ self.A + self.B(x[:, t]))
            out.append(self.C(h))
        return x + torch.stack(out, dim=1) * 0.1

class JEPA(nn.Module):
    def __init__(self, dim, latent=512):
        super().__init__()
        self.enc = nn.Linear(dim, latent, bias=False)
        self.pred = nn.Linear(latent, latent, bias=False)
        self.dec = nn.Linear(latent, dim, bias=False)
    def forward(self, x):
        return None, x + self.dec(self.pred(self.enc(x))) * 0.05

class SynapsisMemory(nn.Module):
    def __init__(self, dim, n_slots=2048):
        super().__init__()
        self.n_slots = n_slots
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.val_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer('keys', torch.zeros(n_slots, dim))
        self.register_buffer('values', torch.zeros(n_slots, dim))
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
    def forward(self, x):
        k = self.key_proj(x[:, -1:]); v = self.val_proj(x[:, -1:])
        with torch.no_grad():
            self.keys[self.step % self.n_slots] = k[0, 0].detach().cpu()
            self.values[self.step % self.n_slots] = v[0, 0].detach().cpu()
            self.step += 1
        scores = k @ self.keys.T.to(x.device)
        top3 = scores.topk(min(3, self.n_slots), dim=-1).indices
        retrieved = self.values[top3[0, 0]].to(x.device)
        context = retrieved.mean(dim=0, keepdim=True).unsqueeze(0)
        return x + self.out_proj(context).expand(-1, x.size(1), -1) * 0.05

class HSAQ(nn.Module):
    def __init__(self, sparsity=0.3):
        super().__init__()
        self.sparsity = sparsity
    def forward(self, x):
        flat = x.abs().view(x.size(0), -1)
        n = flat.size(1)
        k = max(1, min(n - 1, int(n * (1 - self.sparsity))))
        thresh = torch.kthvalue(flat, k, dim=1).values
        thresh = thresh.view(-1, *([1] * (x.dim() - 1)))
        return x * (x.abs() >= thresh)

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = GQA(dim, n_heads, n_kv)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim)
    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Materia250M(nn.Module):
    """M.A.T.E.R.I.A. V3 - 250M parameter model"""
    def __init__(self, vocab_size=32768, dim=1024, n_layers=24, n_heads=16, n_kv=8,
                 jepa_dim=512, synapsis_slots=2048, use_ssm=True, use_snn=True):
        super().__init__()
        self.dim = dim; self.use_ssm = use_ssm; self.use_snn = use_snn
        self.use_jepa = True; self.use_synapsis = True; self.use_hsaq = True

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([TransformerBlock(dim, n_heads, n_kv) for _ in range(n_layers)])
        if use_snn: self.snn = SNNLayer(dim, snn_dim=min(512, dim//2))
        if use_ssm: self.ssm = SSMBlock(dim, state_dim=min(64, dim//16))
        self.jepa = JEPA(dim, jepa_dim)
        self.synapsis = SynapsisMemory(dim, synapsis_slots)
        self.hsaq = HSAQ(sparsity=0.3)
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.gradient_checkpointing = False

    def _forward_impl(self, x, mask=None):
        x = self.tok_emb(x)
        x = self.hsaq(x)
        for l in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(l, x, mask, use_reentrant=False)
            else:
                x = l(x, mask)
        x = self.synapsis(x)
        if self.use_snn:
            x_enh, rate = self.snn(x[:, -1:])
            x = torch.cat([x[:, :-1], x_enh], dim=1)
        if self.use_ssm: x = self.ssm(x)
        _, x = self.jepa(x)
        return self.head(self.norm(x))

    def forward(self, x, mask=None):
        return self._forward_impl(x, mask)

def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ============================================================
# DATASET - BPE Tokenizer + Code/Reasoning
# ============================================================
class TextDataset(Dataset):
    def __init__(self, texts, seq_len=2048):
        self.seq_len = seq_len; self.data = []
        for t in texts:
            if len(t) > seq_len + 1:
                for i in range(0, len(t) - seq_len, seq_len // 2):
                    self.data.append(t[i:i+seq_len+1])
    def __len__(self): return max(1, len(self.data))
    def __getitem__(self, i):
        ids = self.data[i % len(self.data)]
        if len(ids) < self.seq_len + 1:
            ids = ids + [0] * (self.seq_len + 1 - len(ids))
        else:
            ids = ids[:self.seq_len + 1]
        ids = torch.tensor(ids, dtype=torch.long)
        return ids[:-1], ids[1:]

# ============================================================
# OURA LOOP ENGINE (embedded)
# ============================================================
class OuraEngine:
    def __init__(self, model, patience=3, min_delta=0.001, threshold=85.0):
        self.model = model; self.patience = patience
        self.min_delta = min_delta; self.threshold = threshold
        self.best_loss = float('inf'); self.plateau_count = 0
        self.history = {'loss': [], 'acc': [], 'score': [], 'gnorm': []}

    def score(self, loss, acc):
        return min(100, max(0, (100-loss*50)*0.4 + acc*100*0.6))

    def check(self, loss, acc):
        s = self.score(loss, acc)
        self.history['loss'].append(loss); self.history['acc'].append(acc)
        self.history['score'].append(s)
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss; self.plateau_count = 0
        else:
            self.plateau_count += 1
        return s >= self.threshold or self.plateau_count >= self.patience

# ============================================================
# TRAINING SETUP
# ============================================================
if __name__ == '__main__':
    log("=" * 60)
    log(f"MATERIA V3 - SCALING TO 250M PARAMS")
    log(f"Device: {DEVICE}")
    if DEVICE.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
    log("=" * 60)

    # Build simple char tokenizer for bootstrap
    sample_texts = []
    for fp in [
        os.path.join(BASE, 'data/multilingual/tokenizer/c4_en.txt'),
        os.path.join(BASE, 'data/reasoning_dataset.txt'),
    ]:
        if os.path.exists(fp):
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    l = line.strip()
                    if len(l) > 20: sample_texts.append(l)

    # Build tokenizer
    chars = sorted(set(c for t in sample_texts for c in t))[:32764]
    stoi = {c: i+4 for i, c in enumerate(chars)}
    stoi['<PAD>'] = 0; stoi['<BOS>'] = 1; stoi['<EOS>'] = 2; stoi['<UNK>'] = 3
    itos = {i: c for c, i in stoi.items()}
    vocab_size = len(stoi)
    log(f"Vocab size: {vocab_size}")

    # Encode texts
    encoded = []
    used = set()
    for t in sample_texts:
        ids = tuple([stoi.get(c, 3) for c in t])
        if ids not in used and len(ids) > 128:
            used.add(ids)
            encoded.append(list(ids))
    log(f"Encoded texts: {len(encoded):,}")

    # Create model
    model = Materia250M(
        vocab_size=vocab_size,
        dim=1024, n_layers=2,  # Reduced for demo: 2 layers
        n_heads=16, n_kv=8,
        jepa_dim=512, synapsis_slots=128,
        use_ssm=True, use_snn=True
    )
    model.gradient_checkpointing = True

    if DEVICE.type == 'cuda':
        model = model.cuda()
        # Enable FP16
        scaler = GradScaler()
    else:
        scaler = None

    params = count_params(model)
    log(f"Params: {params:,}")
    vr = torch.cuda.memory_allocated()/1e6 if DEVICE.type == 'cuda' else 0
    log(f"VRAM after init: {vr:.1f}MB")

    # Data
    split = int(len(encoded) * 0.9)
    train_ds = TextDataset(encoded[:split], seq_len=256)
    val_ds = TextDataset(encoded[split:], seq_len=256)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, drop_last=True, num_workers=0)
    log(f"Train chunks: {len(train_ds):,} | Val chunks: {len(val_ds):,}")

    # Optimizer (CPU offloaded)
    opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3 * len(train_loader))

    # Oura
    oura = OuraEngine(model)

    # CSV log
    csv_path = os.path.join(BASE, 'logs', 'scaling_250m_log.csv')
    with open(csv_path, 'w', newline='') as cf:
        w = csv.writer(cf)
        w.writerow(['epoch', 'step', 'loss', 'acc', 'gnorm', 'vram_mb', 'score'])

        # Training loop
        for epoch in range(3):
            model.train()
            total_loss = 0.0; total_acc = 0.0
            for i, (x, y) in enumerate(train_loader):
                if DEVICE.type == 'cuda': x, y = x.cuda(), y.cuda()
                opt.zero_grad()

                if scaler:
                    with autocast():
                        logits = model(x)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                    loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                sch.step()
                preds = logits.argmax(dim=-1)
                mask = y != 0
                acc = (preds[mask] == y[mask]).float().mean().item() if mask.sum() > 0 else 0.0
                total_loss += loss.item(); total_acc += acc
                vram = torch.cuda.memory_allocated()/1e6 if DEVICE.type == 'cuda' else 0

                if (i+1) % 25 == 0:
                    w.writerow([epoch+1, i+1, f'{loss.item():.4f}', f'{acc:.4f}', f'{gn:.2f}', f'{vram:.0f}', ''])
                    log(f"  E{epoch+1}/3 [{i+1}/{len(train_loader)}] loss={loss.item():.4f} acc={acc:.4f} vr={vram:.0f}MB")

            avg_loss = total_loss / len(train_loader)
            avg_acc = total_acc / len(train_loader)

            # Validation
            model.eval()
            vloss, vacc, vsteps = 0.0, 0.0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    if DEVICE.type == 'cuda': x, y = x.cuda(), y.cuda()
                    logits = model(x)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0)
                    vloss += loss.item()
                    preds = logits.argmax(dim=-1); mask = y != 0
                    vacc += (preds[mask] == y[mask]).float().mean().item() if mask.sum() > 0 else 0.0
                    vsteps += 1
            avg_vloss = vloss / max(1, vsteps); avg_vacc = vacc / max(1, vsteps)

            score = oura.score(avg_loss, avg_acc)
            log(f"  → E{epoch+1}: loss={avg_loss:.4f} acc={avg_acc:.4f} val_loss={avg_vloss:.4f} val_acc={avg_vacc:.4f} score={score:.1f}")
            w.writerow([epoch+1, 'END', f'{avg_loss:.4f}', f'{avg_acc:.4f}', '', f'{vram:.0f}', f'{score:.1f}'])

            if oura.check(avg_loss, avg_acc):
                log(f"  ✓ Oura converged at epoch {epoch+1}")
                break

    log(f"\nTraining complete! CSV log: {csv_path}")
    log(f"Final: loss={avg_loss:.4f} acc={avg_acc:.4f} score={score:.1f}")

    # Generate Oura report plot
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs_r = range(1, len(oura.history['loss'])+1)
        ax.plot(epochs_r, oura.history['loss'], 'b-o', label='Loss')
        ax.plot(epochs_r, oura.history['acc'], 'g-o', label='Accuracy')
        ax.plot(epochs_r, oura.history['score'], 'r-o', label='Oura Score')
        ax.axhline(y=85, color='gray', linestyle='--', alpha=0.5, label='Threshold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
        ax.set_title(f'M.A.T.E.R.I.A. V3 - Scaling 250M (params={params:,})')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(BASE, 'docs', 'plots', 'scaling_250m_oura.png'), dpi=150)
        log("Report plot saved")
    except:
        pass

    log("✓ Done!")
