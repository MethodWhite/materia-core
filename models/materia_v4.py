"""
MATERIA V4 — Toroidal Hexagonal Architecture (JEPA-First)
JEPA como hub central en un toroide de interconexión hexagonal.
Basado en principios de geometría sagrada: flower of life, toro, hexágono sagrado.

Cada arista del hexágono tiene su propia instancia HSAQ con parámetros
aprendibles. La sparsity EMERGE de la dinámica toroidal, no es fija.

Arquitectura:
                ┌─────────────┐
               ╱  Transformer  ╲
              │     ↕ ↕ ↕      │
     ┌───────┐│   ↔ JEPA ↔    │┌───────┐
     │  SSM  │←━━→   ↕   ←━━→││  SNN  │
     └───────┘│  ↔  Hub  ↔   │└───────┘
              │     ↕ ↕ ↕      │
               ╲              ╱
                └─────┬───────┘
                      ↕
                ┌─────┬───────┐
                │ Head/Emb    │
                └─────────────┘
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from core import (
    RoPE, GQA, FlashGQA, SwiGLU, TransformerBlock,
    LIFNeuron, SNNLayer, SSMBlock,
    HSAQ, SynapsisMemory,
)

K = 2.781042  # Constante de acoplamiento espectral


class JEPAEncoder(nn.Module):
    """JEPA Encoder — hub central del toroide hexagonal"""
    def __init__(self, latent_dim):
        super().__init__()
        self.proj = nn.Linear(latent_dim, latent_dim * 2)
        self.norm = nn.RMSNorm(latent_dim * 2)
        self.out = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x):
        x = self.proj(x)
        x = F.silu(self.norm(x))
        return self.out(x)


class SCA_Predictor(nn.Module):
    """
    Predictor JEPA con descomposición espectral SCA.
    Autovalores λ_n = K · σ(μ_n).
    """
    def __init__(self, latent_dim, hidden=None):
        super().__init__()
        hidden = hidden or latent_dim * 2
        self.spectral = nn.Linear(latent_dim, latent_dim, bias=False)
        self.norm = nn.RMSNorm(latent_dim)
        self.gate = nn.Linear(latent_dim, hidden, bias=False)
        self.up = nn.Linear(latent_dim, hidden, bias=False)
        self.down = nn.Linear(hidden, latent_dim, bias=False)
        self.mu = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x):
        lambda_n = K * torch.sigmoid(self.mu)
        x_spec = self.spectral(x)
        x_spec = x_spec * lambda_n
        x_spec = self.norm(x_spec)
        x_out = F.silu(self.gate(x_spec)) * self.up(x_spec)
        return self.down(x_out)


class HexagonalTorus(nn.Module):
    """
    Conexión hexagonal entre componentes del toroide JEPA.
    Proyecta entre latent_dim y el dim de cada componente.
    """
    def __init__(self, latent_dim, component_dim):
        super().__init__()
        self.to_latent = nn.Linear(component_dim, latent_dim, bias=False)
        self.from_latent = nn.Linear(latent_dim, component_dim, bias=False)

    def forward(self, x_latent):
        """Proyecta del espacio latente JEPA al espacio del componente."""
        return self.from_latent(x_latent)

    def fuse(self, x_component):
        """Fusiona la salida del componente de vuelta al espacio latente JEPA."""
        return self.to_latent(x_component)


class MateriaV4(nn.Module):
    def __init__(self, vocab_size=1024, dim=256, n_layers=3, n_heads=8, n_kv=4,
                 latent_dim=256, snn_dim=128, ssm_state=32,
                 synapsis_slots=128, hsaq_sparsity=0.15,
                 jepa_weight=K, snn_threshold=0.001, snn_tau=0.8,
                 n_cycles=3,
                 use_synapsis=True, use_checkpointing=False, use_flash=False,
                 weight_tying=False):
        super().__init__()
        self.dim = dim
        self.latent_dim = latent_dim
        self.jepa_weight = jepa_weight
        self.n_cycles = n_cycles
        self.use_synapsis = use_synapsis
        self.use_checkpointing = use_checkpointing

        # ─── Hexagonal Torus Components ───
        # Cada componente se conecta al hub JEPA mediante proyecciones hexagonales

        # Embedding → JEPA
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.emb_to_jepa = HexagonalTorus(latent_dim, dim)

        # ─── HSAQ instances: cada arista tiene su propio parámetro aprendible ───
        # Los logits iniciales siguen geometría sagrada (~sigmoide inverso de tasa objetivo)
        # pero son libres de evolucionar durante training
        self.hsaq_emb = HSAQ(init_logit=-3.0)        # ~5%
        self.hsaq_jepa_in = HSAQ(init_logit=-3.0)     # ~5%
        self.hsaq_t2 = HSAQ(init_logit=-2.5)          # ~8%
        self.hsaq_t5 = HSAQ(init_logit=-2.0)          # ~12%
        self.hsaq_t8 = HSAQ(init_logit=-1.7)          # ~15%
        self.hsaq_t_cycle = HSAQ(init_logit=-2.5)     # ~8%
        self.hsaq_snn = HSAQ(init_logit=-3.5)         # ~3% (SNN ya es sparse)
        self.hsaq_snn_latent = HSAQ(init_logit=-3.0)  # ~5%
        self.hsaq_ssm = HSAQ(init_logit=-3.0)         # ~5%
        self.hsaq_jepa_int = HSAQ(init_logit=-3.0)    # ~5%

        self._hsaq_log = []

        # Transformer × N → JEPA
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv, use_flash=use_flash) for _ in range(n_layers)
        ])
        self.t_to_jepa = HexagonalTorus(latent_dim, dim)

        # SNN → JEPA
        self.snn = SNNLayer(dim, snn_dim, threshold=snn_threshold, tau=snn_tau)
        self.snn_to_jepa = HexagonalTorus(latent_dim, dim)

        # SSM → JEPA
        self.ssm = SSMBlock(dim, ssm_state)
        self.ssm_to_jepa = HexagonalTorus(latent_dim, dim)

        # JEPA Hub central
        self.jepa_enc = JEPAEncoder(latent_dim)
        self.jepa_pred = SCA_Predictor(latent_dim)

        # Synapsis (memoria toroidal persistente)
        self.synapsis = SynapsisMemory(latent_dim, synapsis_slots) if use_synapsis else None

        # Head
        self.norm = nn.RMSNorm(latent_dim)
        self.head = nn.Linear(latent_dim, vocab_size, bias=False)

        if weight_tying:
            self.head.weight = self.tok_emb.weight

        nn.init.normal_(self.tok_emb.weight, 0, 0.02)
        nn.init.normal_(self.head.weight, 0, 0.02)

    def forward(self, x):
        B, T = x.shape
        self._hsaq_log = []
        _log = self._hsaq_log.append

        # ═══════════════════════════════════════════
        # FASE 1: EMBEDDING → JEPA HUB
        # ═══════════════════════════════════════════
        h = self.tok_emb(x)
        h = self.hsaq_emb(h)
        _log(('emb', self.hsaq_emb.get_stats()))

        latent = self.emb_to_jepa.fuse(h)
        latent = self.jepa_enc(latent)
        latent = self.hsaq_jepa_in(latent)
        _log(('jepa_in', self.hsaq_jepa_in.get_stats()))

        # ═══════════════════════════════════════════
        # FASE 2: CICLOS TOROIDALES HEXAGONALES
        # ═══════════════════════════════════════════
        spike_rate = torch.tensor(0.0, device=x.device)
        prev_t_mask = None   # estado toroidal entre ciclos
        prev_snn_mask = None
        prev_ssm_mask = None

        for cycle in range(self.n_cycles):
            # —————— Arista 1: Transformer ←→ JEPA ——————
            t_in = self.t_to_jepa(latent)
            for i, layer in enumerate(self.layers):
                if self.use_checkpointing and self.training:
                    t_out = torch.utils.checkpoint.checkpoint(
                        layer.__call__, t_in, None, use_reentrant=False,
                    )
                else:
                    t_out = layer(t_in)
                t_in = t_out
                # HSAQ con herencia toroidal entre ciclos
                if i == 2:
                    t_out = self.hsaq_t2(t_out, inherit_mask=prev_t_mask)
                    _log(('t2', self.hsaq_t2.get_stats()))
                elif i == 5:
                    t_out = self.hsaq_t5(t_out, inherit_mask=prev_t_mask)
                    _log(('t5', self.hsaq_t5.get_stats()))
                elif i == 8:
                    t_out = self.hsaq_t8(t_out, inherit_mask=prev_t_mask)
                    _log(('t8', self.hsaq_t8.get_stats()))
            t_latent = self.t_to_jepa.fuse(t_out)
            t_latent = self.hsaq_t_cycle(t_latent, inherit_mask=prev_t_mask)
            prev_t_mask = self.hsaq_t_cycle._prev_mask
            _log((f't_cycle{cycle}', self.hsaq_t_cycle.get_stats()))

            # —————— Arista 2: SNN ←→ JEPA ——————
            snn_in = self.snn_to_jepa(latent)
            snn_noise = 0.1 if self.training else 0.0
            snn_out, spk = self.snn(snn_in, noise=snn_noise)
            spike_rate = spike_rate + spk / self.n_cycles
            # SNN sin HSAQ (spikes ya son intrínsecamente discretos)
            _log(('snn', {'actual_sparsity': 0.0}))
            snn_latent = self.snn_to_jepa.fuse(snn_out)
            snn_latent = self.hsaq_snn_latent(snn_latent, inherit_mask=prev_snn_mask)
            prev_snn_mask = self.hsaq_snn_latent._prev_mask
            _log(('snn_latent', self.hsaq_snn_latent.get_stats()))

            # —————— Arista 3: SSM ←→ JEPA ——————
            ssm_in = self.ssm_to_jepa(latent)
            ssm_out = self.ssm(ssm_in)
            ssm_out = self.hsaq_ssm(ssm_out, inherit_mask=prev_ssm_mask)
            _log(('ssm', self.hsaq_ssm.get_stats()))
            ssm_latent = self.ssm_to_jepa.fuse(ssm_out)
            ssm_latent = self.hsaq_jepa_int(ssm_latent)
            prev_ssm_mask = self.hsaq_jepa_int._prev_mask
            _log(('ssm_latent', self.hsaq_jepa_int.get_stats()))

            # —————— JEPA Hub: Integración hexagonal ——————
            fused = latent + t_latent + snn_latent + ssm_latent
            fused = fused / 4.0
            latent = self.jepa_enc(fused)

        # ═══════════════════════════════════════════
        # FASE 3: PREDICCIÓN JEPA (MSE en espacio latente)
        # ═══════════════════════════════════════════
        jepa_mse = torch.tensor(0.0, device=x.device)
        if T > 1:
            pred = self.jepa_pred(latent[:, :-1])
            target = latent[:, 1:].detach()
            latent_var = target.var(unbiased=False) + 1e-8
            jepa_mse = F.mse_loss(pred, target) / latent_var
            jepa_mse = jepa_mse.clamp(max=5.0)  # Evita explosión por colapso de varianza

        self.last_jepa_mse = jepa_mse.detach()
        self.last_spike_rate = spike_rate.detach()

        # ═══════════════════════════════════════════
        # FASE 4: HEAD → LOGITS
        # ═══════════════════════════════════════════
        out = self.norm(latent)
        logits = self.head(out)

        self._last_logits = logits.detach()

        return logits, jepa_mse, spike_rate

    def generate(self, idx, max_new=50, temp=0.8, top_p=0.9):
        self.eval()
        for _ in range(max_new):
            logits, _, _ = self.forward(idx[:, -128:])
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


def count_params(model):
    return sum(p.numel() for p in model.parameters())
