"""
Mixture of Experts (MoE) for MATERIA V4.
Sparse MoE con top-2 routing y load balancing loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MOEExpert(nn.Module):
    """
    Experto FFN individual: Linear -> SiLU -> Linear, con expansión 4× hidden_dim.
    Se usa como nn.Sequential compacto.
    """
    def __init__(self, dim: int, hidden_factor: int = 4):
        super().__init__()
        hidden_dim = dim * hidden_factor
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MOERouter(nn.Module):
    """
    Router top-k con load balancing loss.
    - W_r: proyección dim → n_experts
    - top-k routing con softmax sobre los k seleccionados
    - aux_loss: pérdida de load balancing (CV de asignación)
    """
    def __init__(self, dim: int, n_experts: int, top_k: int = 2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.route = nn.Linear(dim, n_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, dim)
        Returns:
            weights: (B, T, top_k) — pesos normalizados de los expertos seleccionados
            indices: (B, T, top_k) — índices de los expertos seleccionados
            aux_loss: scalar — pérdida de load balancing
        """
        B, T, D = x.shape
        # Logits de ruteo
        logits = self.route(x)  # (B, T, n_experts)

        # Top-k routing
        weights, indices = torch.topk(logits, self.top_k, dim=-1)  # (B, T, top_k)
        weights = F.softmax(weights, dim=-1)  # normalizar entre los k seleccionados

        # ── Load balancing loss (aux_loss) ──
        # Fracción de tokens asignados a cada experto (f_i)
        # Usamos one-hot sobre los índices seleccionados
        if self.training:
            with torch.no_grad():
                # Para cada experto e, contar cuántos tokens lo tienen en top-k
                # shape: (B, T, top_k, n_experts) one-hot de índices
                one_hot = torch.zeros(B * T * self.top_k, self.n_experts, device=x.device)
                one_hot.scatter_(1, indices.view(-1, 1), 1)
                one_hot = one_hot.view(B, T, self.top_k, self.n_experts)
                # Fracción de asignación por experto (promedio sobre tokens)
                f_i = one_hot.float().mean(dim=(0, 1, 2))  # (n_experts,)

            # Fracción de peso acumulado por experto (P_i)
            # weights: (B, T, top_k) -> scatter por experto
            route_weights = torch.zeros(B, T, self.n_experts, device=x.device)
            route_weights.scatter_add_(2, indices, weights)
            p_i = route_weights.float().mean(dim=(0, 1))  # (n_experts,)

            # CV loss: n_experts * sum(f_i * p_i) — incentive distribución uniforme
            aux_loss = self.n_experts * (f_i * p_i).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return weights, indices, aux_loss


class SparseMoEBlock(nn.Module):
    """
    Bloque MoE que reemplaza la FFN en TransformerBlock.
    - Router para asignar tokens a expertos
    - Lista de MOEExpert
    - Dispatch con top-2 routing: cada token se envía a sus top-k expertos
    """
    def __init__(self, dim: int, n_experts: int = 8, top_k: int = 2, hidden_factor: int = 4):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = MOERouter(dim, n_experts, top_k)
        self.experts = nn.ModuleList([
            MOEExpert(dim, hidden_factor) for _ in range(n_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim)
        Returns:
            out: (B, T, dim)
        También almacena aux_loss en self.last_aux_loss.
        """
        B, T, D = x.shape
        device = x.device

        weights, indices, aux_loss = self.router(x)
        self.last_aux_loss = aux_loss

        # ── Dispatch tokens a expertos ──
        # Construimos un tensor de salida acumulativo
        out = torch.zeros_like(x)

        # Para cada experto, procesar los tokens asignados
        for expert_idx in range(self.n_experts):
            # Máscara de tokens que seleccionaron este experto
            # indices: (B, T, top_k)
            expert_mask = (indices == expert_idx).any(dim=-1)  # (B, T)
            if not expert_mask.any():
                continue

            # Obtener los pesos correspondientes
            # Necesitamos sumar el peso de este experto cuando fue seleccionado
            # weights: (B, T, top_k), indices: (B, T, top_k)
            expert_weight = torch.zeros(B, T, device=device)
            for k in range(self.top_k):
                k_mask = indices[:, :, k] == expert_idx
                expert_weight = expert_weight + weights[:, :, k] * k_mask.float()

            # Token positions where expert was selected
            flat_indices = expert_mask.view(-1).nonzero(as_tuple=True)[0]  # (n_selected,)

            if len(flat_indices) == 0:
                continue

            # Gather selected tokens
            selected_tokens = x.view(B * T, D)[flat_indices]  # (n_selected, dim)

            # Process through expert
            expert_out = self.experts[expert_idx](selected_tokens)  # (n_selected, dim)

            # Weight and scatter back
            w = expert_weight.view(-1)[flat_indices].unsqueeze(-1)  # (n_selected, 1)
            expert_out = expert_out * w

            out.view(B * T, D).scatter_add_(0, flat_indices.unsqueeze(-1).expand(-1, D), expert_out)

        return out
