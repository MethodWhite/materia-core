"""
Neuro-inspired components: LIF-SNN, SSM
"""
import torch
import torch.nn as nn


class LIFNeuron(nn.Module):
    def __init__(self, threshold=0.05, tau=0.8):
        super().__init__()
        self.th = threshold
        self.tau = tau

    def forward(self, I_in, V=None):
        B = I_in.shape[0]
        if V is None:
            V = I_in.new_zeros(B, 1)
        V = V * self.tau + I_in * (1 - self.tau)
        spike = (V >= self.th).float()
        V = V - spike * self.th
        return spike, V


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


class SSMBlock(nn.Module):
    def __init__(self, dim=256, state_dim=32):
        super().__init__()
        self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.01)
        self.B = nn.Linear(dim, state_dim, bias=False)
        self.C = nn.Linear(state_dim, dim, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = x.new_zeros(B, self.A.shape[0])
        out = []
        for t in range(T):
            h = torch.tanh(h @ self.A + self.B(x[:, t]))
            out.append(self.C(h))
        return x + torch.stack(out, dim=1) * 0.1
