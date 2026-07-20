"""
M.A.T.E.R.I.A. V3 - Full Model Assembly
Ensambla componentes modulares de core/ en el modelo completo.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import os, pickle, time
import numpy as np

from core import (
    RoPE, GQA, SwiGLU, TransformerBlock,
    LIFNeuron, SNNLayer, SSMBlock,
    JEPA, HSAQ, SynapsisMemory,
    AudioEncoder, AudioDecoder,
)


class Tokenizer:
    def __init__(self, model_path=None):
        if model_path is None:
            base = os.path.join(os.path.dirname(__file__), '..',
                                'data/multilingual/tokenizer')
            model_path = os.path.join(base, 'materia_multilingual_v2.model')
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

    def encode(self, text, add_bos=False):
        ids = self.sp.EncodeAsIds(text)
        if add_bos and self.sp.bos_id() >= 0:
            ids = [self.sp.bos_id()] + ids
        return ids

    def decode(self, ids):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self.sp.DecodeIds(ids)

    @property
    def vocab_size(self):
        return self.sp.GetPieceSize()

    @property
    def pad_id(self):
        return self.sp.pad_id() if self.sp.pad_id() >= 0 else 3

    @property
    def unk_id(self):
        return self.sp.unk_id()


class MateriaV3Full(nn.Module):
    def __init__(self, vocab_size=32000, dim=256, n_layers=4, n_heads=8, n_kv=4,
                 audio_dim=80, audio_latent=256, synapsis_slots=128, hsaq_sparsity=0.3,
                 use_synapsis=True):
        super().__init__()
        self.dim = dim
        self.use_synapsis = use_synapsis
        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv) for _ in range(n_layers)
        ])
        self.snn = SNNLayer(dim)
        self.ssm = SSMBlock(dim)
        self.jepa = JEPA(dim)
        self.synapsis = SynapsisMemory(dim, n_slots=synapsis_slots) if use_synapsis else None
        self.hsaq = HSAQ(sparsity=hsaq_sparsity)
        self.norm = nn.RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.audio_enc = AudioEncoder(input_dim=audio_dim, output_dim=audio_latent)
        self.audio_dec = AudioDecoder(input_dim=audio_latent, output_channels=audio_dim)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)

    def forward(self, x, mask=None, audio_input=None):
        if audio_input is not None:
            audio_feats = self.audio_enc(audio_input)
            audio_ctx = audio_feats.mean(dim=1, keepdim=True)
            x = self.tok_emb(x) + audio_ctx
        else:
            x = self.tok_emb(x)
        x = self.hsaq(x)
        for l in self.layers:
            x = l(x, mask)
        if self.synapsis is not None:
            x = self.synapsis(x)
        x_enh, rate = self.snn(x[:, -1:])
        x = torch.cat([x[:, :-1], x_enh], dim=1)
        x = self.ssm(x)
        _, x = self.jepa(x)
        logits = self.head(self.norm(x))
        return logits, rate

    def generate(self, idx, max_new=50, temp=0.8, top_p=0.9):
        self.eval()
        for _ in range(max_new):
            logits, _ = self.forward(idx[:, -128:])
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

    def encode_audio(self, audio):
        return self.audio_enc(audio)

    def decode_audio(self, embeddings, target_len=None):
        return self.audio_dec(embeddings)

    def forward_upscale(self, low_audio):
        feats = self.audio_enc(low_audio)
        return self.audio_dec(feats)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
