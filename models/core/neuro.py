"""
Neuro-inspired components: LIF-SNN, SSM
"""
import torch
import torch.nn as nn


class SurrogateSpikeFn(torch.autograd.Function):
    """
    Surrogate gradient para LIF spike.
    Forward: umbral duro (binario).
    Backward: gradiente suave vía sigmoide (ancho controlado por beta).
    """
    beta = 5.0  # steepness del surrogate (mayor = más parecido al umbral duro)

    @staticmethod
    def forward(ctx, V, th):
        spike = (V >= th).float()
        ctx.th = th
        ctx.save_for_backward(V)
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        V, = ctx.saved_tensors
        th = ctx.th
        # Derivada de sigmoid(V - th) con steepness beta
        sig = torch.sigmoid(SurrogateSpikeFn.beta * (V - th))
        grad_V = grad_output * SurrogateSpikeFn.beta * sig * (1 - sig)
        return grad_V, None


class LIFNeuron(nn.Module):
    def __init__(self, threshold=0.01, tau=0.8, noise_scale=0.05):
        super().__init__()
        self.th = threshold
        self.tau = tau
        self.noise_scale = noise_scale  # ruido para seedear spikes

    def forward(self, I_in, V=None, noise=0.0):
        B, D = I_in.shape[0], I_in.shape[-1]
        if V is None or V.shape[1] != D:
            V = I_in.new_zeros(B, D)
        V = V * self.tau + I_in * (1 - self.tau)
        # Ruido subthreshold para seedear spikes y permitir gradiente fluir
        V = V + noise * self.noise_scale * I_in.new_ones(B, D).uniform_()
        spike = SurrogateSpikeFn.apply(V, self.th)
        V = V * (1 - spike)  # Reset completo post-spike
        return spike, V


class SNNLayer(nn.Module):
    def __init__(self, dim=256, snn_dim=128, threshold=0.01, tau=0.8):
        super().__init__()
        self.w_in = nn.Linear(dim, snn_dim, bias=False)
        self.lif = LIFNeuron(threshold, tau)
        self.w_out = nn.Linear(snn_dim, dim, bias=False)
        self.alpha = nn.Parameter(torch.tensor(0.3))  # escala de fusión aprendible

    def forward(self, x, noise=0.0):
        currents = self.w_in(x)
        B, T, D = currents.shape
        V = currents.new_zeros(B, D)
        spikes_list = []
        for t in range(T):
            spike, V = self.lif(currents[:, t, :], V, noise=noise)
            spikes_list.append(spike)
        spikes = torch.stack(spikes_list, dim=1)
        fusion = self.w_out(spikes) * torch.sigmoid(self.alpha)
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
        return x + torch.stack(out, dim=1) * 0.3
