"""
MATERIA V3 - Unified NumPy Reference Implementation (REFERENCIA)
NO entrenada. Pesos aleatorios. NO cargable desde .materia.
Sirve como referencia de arquitectura en NumPy puro.
NO usar para inferencia real.
"""
import numpy as np

class SwiGLU:
    @staticmethod
    def forward(gate, up):
        return (gate * (1 / (1 + np.exp(-gate)))) * up

class TransformerBlock:
    def __init__(self, dim=256, n_heads=8, n_kv=4, ffn_dim=1024):
        self.wq = np.random.randn(dim, dim) * 0.02
        self.wk = np.random.randn(dim, dim * n_kv // n_heads) * 0.02
        self.wv = np.random.randn(dim, dim * n_kv // n_heads) * 0.02
        self.wo = np.random.randn(dim, dim) * 0.02
        self.wg = np.random.randn(dim, ffn_dim) * 0.02
        self.wu = np.random.randn(dim, ffn_dim) * 0.02
        self.wd = np.random.randn(ffn_dim, dim) * 0.02
        self.n_heads = n_heads; self.n_kv = n_kv; self.head_dim = dim // n_heads
    
    def forward(self, x):
        B, T, D = x.shape; r = x
        x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5)
        q = (x @ self.wq).reshape(B, T, self.n_heads, self.head_dim).transpose(0,2,1,3)
        k = (x @ self.wk).reshape(B, T, self.n_kv, self.head_dim).transpose(0,2,1,3)
        v = (x @ self.wv).reshape(B, T, self.n_kv, self.head_dim).transpose(0,2,1,3)
        rep = self.n_heads // self.n_kv
        k = np.repeat(k, rep, axis=1); v = np.repeat(v, rep, axis=1)
        score = (q @ k.transpose(0,1,3,2)) / np.sqrt(self.head_dim)
        mask = np.triu(np.ones((T,T)) * -1e9, k=1)
        attn = np.exp((score+mask) - (score+mask).max(axis=-1, keepdims=True))
        attn = attn / attn.sum(axis=-1, keepdims=True)
        x = r + (attn @ v).transpose(0,2,1,3).reshape(B,T,D) @ self.wo
        r = x
        x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5)
        gate = x @ self.wg; up = x @ self.wu
        return r + SwiGLU.forward(gate, up) @ self.wd

class SNNBlock:
    def __init__(self, dim=256, snn_dim=128, thresh=0.5, tau=0.8):
        self.w_in = np.random.randn(dim, snn_dim) * 0.05
        self.w_out = np.random.randn(snn_dim, dim) * 0.05
        self.thresh = thresh; self.tau = tau
        self.potential = np.zeros(snn_dim)
    
    def forward(self, x):
        currents = x @ self.w_in
        self.potential = self.potential * self.tau + currents
        spikes = (self.potential >= self.thresh).astype(np.float32)
        self.potential *= (1 - spikes)
        return x + (spikes @ self.w_out) * 0.05, np.mean(spikes)

class SSMBlock:
    def __init__(self, dim=256, state_dim=32):
        self.A = np.random.randn(state_dim, state_dim) * 0.01
        self.B = np.random.randn(state_dim, dim) * 0.01
        self.C = np.random.randn(dim, state_dim) * 0.01
        self.N = state_dim
    
    def forward(self, x):
        B, T, D = x.shape; out = np.zeros_like(x)
        for b in range(B):
            h = np.zeros(self.N)
            for t in range(T):
                h = self.A @ h + self.B @ x[b, t]
                out[b, t] = self.C @ h
        return x + out * 0.1

class JEPA:
    def __init__(self, dim=256, latent=128):
        self.enc = np.random.randn(dim, latent) * 0.02
        self.pred = np.random.randn(latent, latent) * 0.02
        self.dec = np.random.randn(latent, dim) * 0.02
    
    def forward(self, x):
        latent = x @ self.enc; pred = latent @ self.pred
        return latent, x + (pred @ self.dec) * 0.05

class MateriaUnified:
    def __init__(self, dim=256, vocab=1024, n_layers=2):
        self.tok_emb = np.random.randn(vocab, dim) * 0.02
        self.layers = [TransformerBlock(dim) for _ in range(n_layers)]
        self.snn = SNNBlock(dim); self.ssm = SSMBlock(dim); self.jepa = JEPA(dim)
        self.norm = np.ones(dim); self.head = np.random.randn(dim, vocab) * 0.02
        self._params = None
    
    def forward(self, ids):
        x = self.tok_emb[ids]
        for l in self.layers: x = l.forward(x)
        x_enh, self._rate = self.snn.forward(x[:, -1, :])
        x[:, -1, :] = x_enh
        x = self.ssm.forward(x)
        _, x = self.jepa.forward(x)
        x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-5) * self.norm
        return x @ self.head, self._rate
    
    def generate(self, ids, n=10, t=0.8):
        for _ in range(n):
            logits, _ = self.forward(ids[:, -64:])
            p = np.exp(logits[0,-1,:]/t); p = p/p.sum()
            ids = np.concatenate([ids, np.array([[np.random.choice(len(p), p=p)]])], axis=1)
        return ids
    
    def count_params(self):
        if self._params is None:
            self._params = self.tok_emb.size + self.head.size
            for l in self.layers:
                self._params += sum(p.size for p in [l.wq, l.wk, l.wv, l.wo, l.wg, l.wu, l.wd])
            self._params += self.snn.w_in.size + self.snn.w_out.size
            self._params += self.ssm.A.size + self.ssm.B.size + self.ssm.C.size
            self._params += self.jepa.enc.size + self.jepa.pred.size + self.jepa.dec.size
        return self._params
