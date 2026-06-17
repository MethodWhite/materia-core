"""
Joint Embedding Predictive Architecture
"""
import torch
import torch.nn as nn


class JEPA(nn.Module):
    def __init__(self, dim=256, latent=128):
        super().__init__()
        self.enc = nn.Linear(dim, latent, bias=False)
        self.pred = nn.Linear(latent, latent, bias=False)
        self.dec = nn.Linear(latent, dim, bias=False)

    def forward(self, x):
        latent = self.enc(x)
        pred = self.pred(latent)
        reconstruction = self.dec(pred)
        return latent, x + reconstruction * 0.3
