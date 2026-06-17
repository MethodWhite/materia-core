"""
HyperSparse Adaptive Quantization - ejecucion dispersa adaptativa
"""
import torch
import torch.nn as nn


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
        mask = x.abs() >= thresh
        return x * mask
