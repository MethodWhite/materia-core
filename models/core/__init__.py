"""
M.A.T.E.R.I.A. V3 - Core Components
Arquitectura modular escalable: GQA + RoPE + SwiGLU + LIF-SNN + SSM + JEPA + Synapsis + HSAQ
"""
from .blocks import RoPE, GQA, FlashGQA, SwiGLU, TransformerBlock
from .neuro import LIFNeuron, SNNLayer, SSMBlock
from .jepa import JEPA
from .hsaq import HSAQ
from .synapsis import SynapsisMemory
from .audio import AudioEncoder, AudioDecoder
