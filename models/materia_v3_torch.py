"""
MATERIA V3 - Modelo torch (compatibilidad hacia atras)
SNN real con LIF (no sigmoid), RMSNorm en vez de LayerNorm.
Componentes importados de core/ — mismo codigo fuente que MateriaV3Full.
"""
from core import (
    RoPE, GQA, SwiGLU, LIFNeuron, SSMBlock, JEPA
)
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim=256, n_heads=8, n_kv=4):
        super().__init__()
        self.attn_norm = nn.RMSNorm(dim)
        self.attn = GQA(dim, n_heads, n_kv)
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim)

    def forward(self, x, mask=None):
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class SNNLayer(nn.Module):
    def __init__(self, dim=256, snn_dim=128, threshold=0.05, tau=0.8):
        super().__init__()
        self.w_in = nn.Linear(dim, snn_dim, bias=False)
        self.lif = LIFNeuron(threshold, tau)
        self.w_out = nn.Linear(snn_dim, dim, bias=False)

    def forward(self, x):
        currents = self.w_in(x)
        B, T, D = currents.shape
        V = currents.new_zeros(B, 1)
        spikes_list = []
        for t in range(T):
            spike, V = self.lif(currents[:, t, :], V)
            spikes_list.append(spike)
        spikes = torch.stack(spikes_list, dim=1)
        fusion = self.w_out(spikes) * 0.05
        return x + fusion, spikes.mean()


class MateriaV3(nn.Module):
    def __init__(self, vocab_size=4096, dim=256, n_layers=4, n_heads=8, n_kv=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv) for _ in range(n_layers)
        ])
        self.snn = SNNLayer(dim)
        self.ssm = SSMBlock(dim)
        self.jepa = JEPA(dim)
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.spike_rate = 0.0
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None):
        x = self.tok_emb(x)
        for l in self.layers:
            x = l(x, mask)
        x_enh, rate = self.snn(x[:, -1:])
        self.spike_rate = rate.item() if torch.is_tensor(rate) else rate
        x = torch.cat([x[:, :-1], x_enh], dim=1)
        x = self.ssm(x)
        _, x = self.jepa(x)
        return self.head(self.norm(x)), rate

    def generate(self, idx, max_new=50, temp=0.8, top_p=0.9):
        self.eval()
        for _ in range(max_new):
            logits, _ = self.forward(idx[:, -64:])
            l = logits[:, -1, :] / temp
            if top_p < 1.0:
                sorted_logits, sorted_idx = l.sort(descending=True)
                cumsum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                cutoff = cumsum <= top_p
                sorted_logits[~cutoff] = float('-inf')
                if not torch.isinf(sorted_logits).all():
                    l = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            p = F.softmax(l, dim=-1)
            if torch.isnan(p).any() or (p == 0).all():
                p = torch.ones_like(p) / p.size(-1)
            idx = torch.cat([idx, torch.multinomial(p, 1)], dim=1)
        return idx
