"""
MATERIA V4 — OpenAI-compatible Inference Server
================================================
Arquitectura: JEPA-First + Toroidal + GQA (16H/8KV) + SNN + SSM + HSAQ
Checkpoint: HuggingFace MethodWhite/materia-v4-1b-bpe/checkpoint_epoch2.pt
Tokenizer: SentencePiece BPE 32K + char-level fallback
"""

import os, sys, json, time, logging
from typing import Optional, List, AsyncGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ── Path setup ──────────────────────────────────────────────────────
MATERIA_HOME = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
MODELS_DIR = os.path.join(MATERIA_HOME, 'models')
sys.path.insert(0, MODELS_DIR)

from materia_v4 import MateriaV4

logger = logging.getLogger("materia-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── Architecture (auto-detected from checkpoint) ────────────────────
# Checkpoint analysis (MethodWhite/materia-v4-1b-bpe/checkpoint_epoch2.pt):
#   tok_emb.weight: [32000, 1792]  →  vocab=32000, dim=1792
#   wq.weight: [1792, 1792]        →  n_heads=1792/112=16
#   wk.weight: [896, 1792]         →  n_kv=896/112=8
#   rope.inv_freq: [56]            →  head_dim=112
#   ffn.gate.weight: [7168, 1792]  →  4x hidden
#   snn.w_in.weight: [896, 1792]   →  snn_dim=896
#   ssm.A: [64, 64]                →  ssm_state=64
#   layers.0—23 (24 layers)
#   synapsis: NOT in checkpoint    →  use_synapsis=False
#   MoE: NOT in checkpoint         →  use_moe=False
#   n_cycles: default 3 (no override in checkpoint)

ARCH = {
    "vocab_size": 32000,
    "dim": 1792,
    "n_layers": 24,
    "n_heads": 16,       # 1792 / 112
    "n_kv": 8,            # 896 / 112
    "latent_dim": 1792,
    "snn_dim": 896,
    "ssm_state": 64,
    "snn_threshold": 0.001,
    "snn_tau": 0.8,
    "synapsis_slots": 64,
    "hsaq_sparsity": 0.3,
    "jepa_weight": 2.781042,
    "n_cycles": 3,
    "use_synapsis": False,
    "use_flash": False,
    "use_moe": False,
    "rope_theta": 10000.0,
    "rope_scaling_factor": 1.0,
}

MAX_SEQ_LEN = 2048
TOP_K = 50

HF_REPO = "MethodWhite/materia-v4-1b-bpe"
CHECKPOINT_FILENAME = "checkpoint_epoch2.pt"

CHECKPOINT_DIR = os.path.join(MATERIA_HOME, "checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

TOKENIZER_MODEL = os.path.join(
    MATERIA_HOME, "data", "multilingual", "tokenizer", "materia_multilingual_v2.model"
)

CHAR_TOKENIZER_PATH = os.path.join(CHECKPOINT_DIR, "char_tokenizer.json")

# ── App globals ─────────────────────────────────────────────────────
app = FastAPI(title="MATERIA V4 Inference API", version="4.0.0")
model_global: Optional[MateriaV4] = None
tokenizer_global: Optional["MateriaTokenizer"] = None
char_stoi: Optional[dict] = None
char_itos: Optional[dict] = None
device_global: Optional[torch.device] = None


# ====================================================================
# Tokenizer
# ====================================================================

class MateriaTokenizer:
    """BPE SentencePiece tokenizer with char-level fallback."""

    def __init__(self, sp_model_path: str = TOKENIZER_MODEL):
        self.sp = None
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sp_model_path)
            logger.info(f"BPE tokenizer loaded: vocab_size={self.sp.GetPieceSize()}")
        except Exception as e:
            logger.warning(f"Cannot load SentencePiece model ({e}). Using char-level fallback.")

    def encode(self, text: str) -> list[int]:
        if self.sp is not None:
            try:
                return self.sp.EncodeAsIds(text)
            except Exception:
                pass
        return self._char_encode(text)

    def decode(self, ids: list[int]) -> str:
        if self.sp is not None:
            try:
                return self.sp.DecodeIds([i for i in ids if i < self.sp.GetPieceSize()])
            except Exception:
                pass
        return self._char_decode(ids)

    def _char_encode(self, text: str) -> list[int]:
        global char_stoi
        if char_stoi is None:
            return [ord(c) if ord(c) < ARCH["vocab_size"] else 3 for c in text]
        return [char_stoi.get(c, 3) for c in text]

    def _char_decode(self, ids: list[int]) -> str:
        global char_itos
        if char_itos is None:
            return ''.join(chr(i) if i < 0x110000 else '?' for i in ids)
        return ''.join(char_itos.get(i, '?') for i in ids)

    @property
    def vocab_size(self) -> int:
        if self.sp is not None:
            return self.sp.GetPieceSize()
        global char_stoi
        if char_stoi is not None:
            return len(char_stoi)
        return ARCH["vocab_size"]


# ====================================================================
# Checkpoint & model loading
# ====================================================================

def download_checkpoint() -> str:
    """Download checkpoint from HuggingFace if not cached locally."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(CHECKPOINT_PATH):
        size = os.path.getsize(CHECKPOINT_PATH)
        logger.info(f"Checkpoint cached: {CHECKPOINT_PATH} ({size / 1024**3:.2f}GB)")
        return CHECKPOINT_PATH

    logger.info(f"Downloading {HF_REPO}/{CHECKPOINT_FILENAME} from HuggingFace...")
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=HF_REPO, filename=CHECKPOINT_FILENAME,
                                local_dir=CHECKPOINT_DIR, local_dir_use_symlinks=False)
        logger.info(f"Downloaded to {path}")
        return path
    except Exception as e:
        logger.warning(f"HF download failed: {e}. Searching locally...")
        for root, _, files in os.walk(MATERIA_HOME):
            for f in files:
                if f.endswith(('.basemateria', '.pt', '.pth')):
                    path = os.path.join(root, f)
                    logger.info(f"Found local: {path}")
                    return path
        raise RuntimeError(
            f"Checkpoint not found. Tried: {HF_REPO}/{CHECKPOINT_FILENAME} "
            f"and searched {MATERIA_HOME} for .pt/.pth/.basemateria"
        )


def build_char_tokenizer(embedding_weight: torch.Tensor):
    """Build char-level tokenizer from checkpoint embedding shape."""
    global char_stoi, char_itos
    n = embedding_weight.shape[0]
    # Map printable codepoints to token IDs, keeping within vocab range
    chars = set()
    for cp in range(32, min(0x4E00, n - 10)):
        if 0xD800 <= cp < 0xE000:
            continue
        chars.add(chr(cp))
    for cp in range(0x4E00, min(0x9FFF, n - 10), 50):
        chars.add(chr(cp))
    chars = sorted(chars)[:n - 6]
    stoi = {ch: i + 6 for i, ch in enumerate(chars) if i + 6 < n}
    stoi.update({'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3, '<SEP>': 4, '<CLS>': 5})
    itos = {v: k for k, v in stoi.items()}
    char_stoi, char_itos = stoi, itos
    logger.info(f"Char-level tokenizer built: vocab={len(stoi)}")
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        with open(CHAR_TOKENIZER_PATH, 'w', encoding='utf-8') as f:
            json.dump({'stoi': stoi, 'itos': {str(k): v for k, v in itos.items()}}, f)
    except Exception:
        pass


def load_model() -> tuple[MateriaV4, MateriaTokenizer, torch.device]:
    """Load model + tokenizer, dispatch to GPU (half) with CPU fallback."""

    # ── Device selection ────────────────────────────────────────────
    if torch.cuda.is_available():
        free = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {torch.cuda.get_device_properties(0).name} ({free:.1f}GB VRAM)")
        if free >= 3.0:
            device, dtype = torch.device('cuda:0'), torch.float16
            logger.info("→ GPU float16")
        elif free >= 2.0:
            bf16 = hasattr(torch, 'bfloat16') and torch.cuda.is_bf16_supported()
            device, dtype = torch.device('cuda:0'), torch.bfloat16 if bf16 else torch.float16
            logger.info(f"→ GPU {dtype} (limited VRAM)")
        else:
            device, dtype = torch.device('cpu'), torch.float32
            logger.info("→ CPU float32 (insufficient GPU VRAM)")
    else:
        device, dtype = torch.device('cpu'), torch.float32
        logger.info("→ CPU float32 (no GPU)")

    # ── Load checkpoint ─────────────────────────────────────────────
    ckpt_path = download_checkpoint()
    logger.info(f"Loading checkpoint (this may take a minute)...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    fmt = "nested" if 'model_state_dict' in ckpt else "flat"
    logger.info(f"Checkpoint format: {fmt}, {len(state_dict)} keys")

    actual_vocab = state_dict['tok_emb.weight'].shape[0]
    logger.info(f"Vocab from checkpoint: {actual_vocab}")

    # ── Char-level tokenizer fallback ───────────────────────────────
    if os.path.exists(CHAR_TOKENIZER_PATH):
        try:
            with open(CHAR_TOKENIZER_PATH) as f:
                data = json.load(f)
            global char_stoi, char_itos
            char_stoi = data['stoi']
            char_itos = {int(k): v for k, v in data['itos'].items()}
            logger.info(f"Char tokenizer restored from disk ({len(char_stoi)} tokens)")
        except Exception:
            pass
    if char_stoi is None:
        build_char_tokenizer(state_dict['tok_emb.weight'])

    # ── Instantiate model ───────────────────────────────────────────
    a = ARCH
    logger.info(f"Building MateriaV4(vocab={actual_vocab}, dim={a['dim']}, "
                f"layers={a['n_layers']}, heads={a['n_heads']}, kv={a['n_kv']})")

    model = MateriaV4(
        vocab_size=actual_vocab,
        dim=a['dim'],
        n_layers=a['n_layers'],
        n_heads=a['n_heads'],
        n_kv=a['n_kv'],
        latent_dim=a['latent_dim'],
        snn_dim=a['snn_dim'],
        snn_threshold=a['snn_threshold'],
        snn_tau=a['snn_tau'],
        ssm_state=a['ssm_state'],
        synapsis_slots=a['synapsis_slots'],
        hsaq_sparsity=a['hsaq_sparsity'],
        jepa_weight=a['jepa_weight'],
        n_cycles=a['n_cycles'],
        use_synapsis=a['use_synapsis'],
        use_checkpointing=False,
        use_flash=torch.cuda.is_available(),
        use_moe=a['use_moe'],
        rope_theta=a['rope_theta'],
        rope_scaling_factor=a['rope_scaling_factor'],
    )

    # ── Load weights ────────────────────────────────────────────────
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:8]}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:8]}")
    logger.info(f"Weights loaded. Missing={len(missing)}, Unexpected={len(unexpected)}")
    model.eval()

    # ── Move to device ──────────────────────────────────────────────
    try:
        model = model.to(device=device, dtype=dtype)
        logger.info(f"Model on {device} ({dtype})")
    except RuntimeError as e:
        logger.error(f"GPU OOM: {e}. Falling back to CPU float32.")
        device, dtype = torch.device('cpu'), torch.float32
        model = model.to(device=device, dtype=dtype)

    params = sum(p.numel() for p in model.parameters())
    mem = params * 2 / 1024**3 if dtype == torch.float16 else params * 4 / 1024**3
    logger.info(f"Parameters: {params:,}  |  Memory: {mem:.2f}GB")

    # ── Tokenizer ───────────────────────────────────────────────────
    tokenizer = MateriaTokenizer()

    return model, tokenizer, device


# ====================================================================
# Generation
# ====================================================================

@torch.no_grad()
def generate_tokens(
    model: MateriaV4,
    tokenizer: MateriaTokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = TOP_K,
) -> tuple[str, list[int], float]:
    """Generate text. Returns (text, token_ids, tokens_per_sec)."""
    input_ids = tokenizer.encode(prompt) or [char_stoi.get(' ', 3)]
    device = next(model.parameters()).device
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = []
    t0 = time.time()

    for _ in range(max_tokens):
        ctx = x[:, -MAX_SEQ_LEN:]
        logits, _, _, _ = model.forward(ctx)
        logits = logits[:, -1, :] / max(temperature, 0.01)

        if top_k > 0:
            vals, idx = logits.topk(top_k, dim=-1)
            probs = F.softmax(vals, dim=-1)
            tid = idx.gather(1, torch.multinomial(probs, 1))
        else:
            if top_p < 1.0:
                sorted_l, sorted_i = logits.sort(descending=True)
                cum = sorted_l.softmax(dim=-1).cumsum(dim=-1)
                keep = cum <= top_p
                if keep.any():
                    sorted_l[~keep] = float('-inf')
                    logits = sorted_l.scatter(1, sorted_i, sorted_l)
            probs = F.softmax(logits, dim=-1)
            if probs.isnan().any() or (probs == 0).all():
                probs = torch.ones_like(probs) / probs.size(-1)
            tid = torch.multinomial(probs, 1)

        token_id = tid.item()
        generated.append(token_id)
        if token_id in (0, 2):  # <PAD>, <EOS>
            break
        x = torch.cat([x, tid], dim=1)
        if x.shape[1] > MAX_SEQ_LEN + max_tokens:
            x = x[:, -MAX_SEQ_LEN:]

    elapsed = time.time() - t0
    text = tokenizer.decode(generated)
    return text, generated, len(generated) / elapsed if elapsed > 0 else 0


async def stream_generate(
    model: MateriaV4,
    tokenizer: MateriaTokenizer,
    prompt: str,
    max_tokens: int = 128,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = TOP_K,
) -> AsyncGenerator[str, None]:
    """Stream tokens as Server-Sent Events."""
    input_ids = tokenizer.encode(prompt) or [char_stoi.get(' ', 3)]
    device = next(model.parameters()).device
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    generated = []
    finish = None

    # Initial chunk
    yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant'}, 'index': 0, 'finish_reason': None}]})}\n\n"

    for _ in range(max_tokens):
        ctx = x[:, -MAX_SEQ_LEN:]
        logits, _, _, _ = model.forward(ctx)
        logits = logits[:, -1, :] / max(temperature, 0.01)

        if top_k > 0:
            vals, idx = logits.topk(top_k, dim=-1)
            probs = F.softmax(vals, dim=-1)
            tid = idx.gather(1, torch.multinomial(probs, 1))
        else:
            if top_p < 1.0:
                sorted_l, sorted_i = logits.sort(descending=True)
                cum = sorted_l.softmax(dim=-1).cumsum(dim=-1)
                keep = cum <= top_p
                if keep.any():
                    sorted_l[~keep] = float('-inf')
                    logits = sorted_l.scatter(1, sorted_i, sorted_l)
            probs = F.softmax(logits, dim=-1)
            if probs.isnan().any() or (probs == 0).all():
                probs = torch.ones_like(probs) / probs.size(-1)
            tid = torch.multinomial(probs, 1)

        token_id = tid.item()
        token_text = tokenizer.decode([token_id]).replace('\n', '\\n')
        generated.append(token_id)

        if token_id in (0, 2):
            finish = "stop"
            yield f"data: {json.dumps({'choices': [{'delta': {'content': token_text}, 'index': 0, 'finish_reason': None}]})}\n\n"
            break
        else:
            yield f"data: {json.dumps({'choices': [{'delta': {'content': token_text}, 'index': 0, 'finish_reason': None}]})}\n\n"

        x = torch.cat([x, tid], dim=1)
        if x.shape[1] > MAX_SEQ_LEN + max_tokens:
            x = x[:, -MAX_SEQ_LEN:]

    usage = {
        "prompt_tokens": len(input_ids),
        "completion_tokens": len(generated),
        "total_tokens": len(input_ids) + len(generated),
    }
    yield f"data: {json.dumps({'choices': [{'delta': {}, 'index': 0, 'finish_reason': finish or 'length'}], 'usage': usage})}\n\n"
    yield "data: [DONE]\n\n"


# ====================================================================
# API request models
# ====================================================================

class CompletionRequest(BaseModel):
    model: str = "materia-v4-1b"
    prompt: str = Field(default="", description="Input text")
    suffix: Optional[str] = None
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=TOP_K, ge=0, le=100)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    role: str = Field(default="user", pattern="^(system|user|assistant)$")
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "materia-v4-1b"
    messages: List[ChatMessage] = Field(default_factory=list)
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=TOP_K, ge=0, le=100)
    n: int = Field(default=1, ge=1, le=1)
    stream: bool = False
    stop: Optional[List[str]] = None


# ====================================================================
# Endpoints
# ====================================================================

@app.on_event("startup")
async def startup_event():
    global model_global, tokenizer_global, device_global
    if model_global is not None:
        return
    logger.info("Loading model on startup...")
    try:
        model_global, tokenizer_global, device_global = load_model()
        logger.info("Ready!")
        logger.info(f"  Device: {device_global}")
        logger.info(f"  Params: {sum(p.numel() for p in model_global.parameters()):,}")
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise


@app.get("/health")
async def health():
    ready = model_global is not None
    return {
        "status": "ok" if ready else "loading",
        "model": "materia-v4-1b",
        "device": str(device_global) if device_global else "unknown",
        "vocab_size": tokenizer_global.vocab_size if tokenizer_global else "?",
        "parameters": f"{sum(p.numel() for p in model_global.parameters()):,}" if model_global else "?",
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{
            "id": "materia-v4-1b",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "MethodWhite",
            "permission": [],
        }],
    }


@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    if model_global is None:
        raise HTTPException(503, "Model still loading")
    if not req.prompt:
        raise HTTPException(400, "prompt is required")
    temp = req.temperature if req.temperature > 0 else 0.1

    if req.stream:
        return StreamingResponse(
            stream_generate(model_global, tokenizer_global, req.prompt,
                            max_tokens=req.max_tokens, temperature=temp,
                            top_p=req.top_p, top_k=req.top_k),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )

    text, ids, tps = generate_tokens(model_global, tokenizer_global, req.prompt,
                                     max_tokens=req.max_tokens, temperature=temp,
                                     top_p=req.top_p, top_k=req.top_k)
    prompt_ids = tokenizer_global.encode(req.prompt)
    return {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(ids),
            "total_tokens": len(prompt_ids) + len(ids),
        },
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model_global is None:
        raise HTTPException(503, "Model still loading")
    if not req.messages:
        raise HTTPException(400, "messages is required")

    parts = []
    for m in req.messages:
        tag = m.role
        parts.append(f"<|{tag}|>{m.content}</s>")
    parts.append("<|assistant|>")
    prompt = "".join(parts)

    temp = req.temperature if req.temperature > 0 else 0.1

    if req.stream:
        return StreamingResponse(
            stream_generate(model_global, tokenizer_global, prompt,
                            max_tokens=req.max_tokens, temperature=temp,
                            top_p=req.top_p, top_k=req.top_k),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive",
                     "X-Accel-Buffering": "no"},
        )

    text, ids, tps = generate_tokens(model_global, tokenizer_global, prompt,
                                     max_tokens=req.max_tokens, temperature=temp,
                                     top_p=req.top_p, top_k=req.top_k)
    pt = sum(len(tokenizer_global.encode(m.content)) for m in req.messages)
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                      "finish_reason": "stop"}],
        "usage": {"prompt_tokens": pt, "completion_tokens": len(ids),
                  "total_tokens": pt + len(ids)},
    }


# ====================================================================
# Main
# ====================================================================

if __name__ == "__main__":
    port = int(os.environ.get("MATERIA_PORT", 8080))
    host = os.environ.get("MATERIA_HOST", "0.0.0.0")
    a = ARCH

    logger.info("=" * 60)
    logger.info("MATERIA V4 Inference Server")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Model: materia-v4-1b ({a['dim']}d, {a['n_layers']}L, "
                f"{a['n_heads']}H, {a['n_kv']}KV)")
    logger.info(f"  Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"  Tokenizer: {TOKENIZER_MODEL}")
    logger.info("=" * 60)

    uvicorn.run(app, host=host, port=port, log_level="info", timeout_keep_alive=30)
