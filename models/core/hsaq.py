"""
HSAQ — HyperSparse Adaptive Quantization (v2.1)
Optimizador adaptativo toroidal con parámetros aprendibles.

HSAQ NO es un compresor fijo ni un eliminador de información.
Es un optimizador que ayuda a activar los nodos correctos mediante
máscaras dinámicas, donde cada arista del toroide tiene su propio
parámetro aprendible que evoluciona con el training.

La sparsity EMERGE de la interacción entre:
- La distribución de activaciones del batch actual
- El parámetro aprendible de cada arista (sparsity_logit)
- El estado del ciclo toroidal anterior (inherit_mask)

El gradiente fluye vía: soft_mask → threshold → scale → sparsity_logit
"""
import torch
import torch.nn as nn
import math


class HSAQ(nn.Module):
    """HyperSparse Adaptive Quantization — optimizador adaptativo.

    Cada arista del toroide hexagonal tiene su propia instancia HSAQ
    con un parámetro de sparsity aprendible.

    El sparsity_logit controla un umbral relativo a la mediana de las
    activaciones. scale=0 → sin sparsity, scale=1 → sparsity=50%.

    Args:
        init_logit: Logit inicial (sigmoide). init=0 → scale=0.5 (sparsity~50%)
                    init=-3 → scale≈0.05, init=-2 → scale≈0.12, etc.
        temperature: Temperatura para STE sigmoide.
    """
    def __init__(self, init_logit: float = 0.0, temperature: float = 0.01,
                 clamp_min: float = -5, clamp_max: float = 5):
        super().__init__()
        self.sparsity_logit = nn.Parameter(torch.tensor(init_logit))
        self.temperature = temperature
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # Estado toroidal entre ciclos
        self.register_buffer('_prev_mask', None)
        self.register_buffer('_prev_threshold', torch.tensor(0.0))

    def forward(self, x: torch.Tensor, bias: float = 0.0,
                inherit_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Aplica máscara HSAQ adaptativa.

        Args:
            x: Tensor de entrada (B, D) o (B, T, D).
            bias: Sesgo adicional para sparsity (influencia toroidal).
            inherit_mask: Máscara del ciclo anterior (continuidad).
        """
        # ── Sparsity emerge: [0, 1] — 0 = nada, 1 = máximo ──
        logit = torch.clamp(self.sparsity_logit + bias, min=self.clamp_min, max=self.clamp_max)
        scale = torch.sigmoid(logit)

        # ── Umbral de referencia: mediana de |activaciones| ──
        flat = x.abs().view(x.size(0), -1)
        n = flat.size(1)
        median_th = torch.kthvalue(flat, n // 2, dim=1).values
        median_th = median_th.view(-1, *([1] * (x.dim() - 1)))

        # ── Umbral efectivo: aprendible vía scale ──
        # scale=0 → th=0 (todo pasa), scale=1 → th=mediana (pasa 50%)
        # Gradiente: loss → soft_mask → thresh → scale → sparsity_logit
        # Floor 1e-8 evita colapso total (piso numérico para estabilidad)
        thresh = median_th.detach() * scale + 1e-8

        # ── Máscara dura (forward) ──
        mask_hard = (x.abs() >= thresh).float()

        # Integrar máscara del ciclo anterior (continuidad toroidal)
        if inherit_mask is not None and inherit_mask.shape == x.shape:
            mask_hard = mask_hard * 0.7 + inherit_mask * 0.3
            mask_hard = (mask_hard >= 0.5).float()

        # ── STE: forward = duro, backward = suave ──
        if self.training:
            # Soft mask dependiente de thresh → scale → logit
            soft_mask = torch.sigmoid(
                (x.abs() - thresh) / (median_th.abs().mean() * 0.1 + 1e-8)
            )
            mask = mask_hard + soft_mask - soft_mask.detach()
        else:
            mask = mask_hard

        # Guardar estado toroidal
        self._prev_mask = mask.detach()
        self._prev_threshold = thresh.view(-1).mean().detach()

        self._last_sparsity = 1.0 - mask.float().mean().item()
        self._last_threshold = thresh.view(-1).mean().item()
        self._last_sparsity_target = scale.item()

        return x * mask

    def get_stats(self) -> dict:
        """Retorna métricas actuales de HSAQ."""
        scale = torch.sigmoid(self.sparsity_logit).item()
        return {
            'sparsity_logit': self.sparsity_logit.item(),
            'sparsity_scale': scale,
            'actual_sparsity': getattr(self, '_last_sparsity', 0.0),
            'threshold': getattr(self, '_last_threshold', 0.0),
            'type': 'activation_optimizer',
            'method': 'adaptive_threshold_ste',
        }
