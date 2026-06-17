"""
Synapsis Memory - memoria persistente con top-K retrieval
"""
import torch
import torch.nn as nn
import pickle


class SynapsisMemory(nn.Module):
    def __init__(self, dim=256, n_slots=128):
        super().__init__()
        self.n_slots = n_slots
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.val_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer('keys', torch.zeros(n_slots, dim))
        self.register_buffer('values', torch.zeros(n_slots, dim))
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))

    def forward(self, x):
        B, T, D = x.shape
        k = self.key_proj(x[:, -1:])
        v = self.val_proj(x[:, -1:])
        with torch.no_grad():
            for b in range(B):
                slot = (self.step.item() + b) % self.n_slots
                self.keys[slot] = k[b, 0].detach().cpu().to(dtype=self.keys.dtype)
                self.values[slot] = v[b, 0].detach().cpu().to(dtype=self.values.dtype)
            self.step += B
        scores = k @ self.keys.T.to(x.device, dtype=k.dtype)
        top3 = scores.topk(min(3, self.n_slots), dim=-1).indices
        contexts = []
        for b in range(B):
            retrieved = self.values[top3[b, 0]].to(x.device, dtype=x.dtype)
            ctx = retrieved.mean(dim=0, keepdim=True)
            contexts.append(ctx)
        context = torch.stack(contexts, dim=0)
        return x + self.out_proj(context).expand(-1, T, -1) * 0.3

    def save_memory(self, path):
        data = {
            'keys': self.keys.cpu().numpy(),
            'values': self.values.cpu().numpy(),
            'step': self.step.cpu().item(),
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_memory(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.keys = torch.from_numpy(data['keys'])
        self.values = torch.from_numpy(data['values'])
        self.step = torch.tensor([data['step']], dtype=torch.long)
