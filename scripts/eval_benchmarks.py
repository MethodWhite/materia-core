#!/usr/bin/env python3
"""
MATERIA — Evaluation Benchmarks
==================================
Perplexity on WikiText-2  |  Accuracy on HellaSwag

Usage:
    python scripts/eval_benchmarks.py --checkpoint <path>
                                      [--model v4] [--device cuda] [--samples N]

Supports:
  • V4 (char-level, vocab_size ~1024) — MateriaV4 from materia_v4.py
  • V3 (SentencePiece, vocab_size 32000) — MateriaV3Full from materia_v3_full.py
  • .pt / .pth / .materia checkpoint formats

Output: JSON object with keys {hellaswag_acc, wikitext_ppl, params, device, time}.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))


# ═════════════════════════════════════════════════════════════════════════════
#  Tokenizers
# ═════════════════════════════════════════════════════════════════════════════

def _build_char_stoi(texts, vocab_size=1024):
    """Build char-to-id mapping from a list of strings (matches V4 training)."""
    chars = set()
    for t in texts:
        chars.update(t)
    chars = sorted(chars)[:vocab_size - 4]
    stoi = {ch: i + 4 for i, ch in enumerate(chars)}
    stoi["<PAD>"] = 0
    stoi["<BOS>"] = 1
    stoi["<EOS>"] = 2
    stoi["<UNK>"] = 3
    return stoi


class CharTokenizer:
    """Char-level tokenizer matching MATERIA V4 training pipeline."""

    def __init__(self, stoi: dict):
        self.stoi = stoi
        self.itos = {i: ch for ch, i in stoi.items()}
        self.vocab_size = len(stoi)
        self.pad_id = stoi.get("<PAD>", 0)
        self.unk_id = stoi.get("<UNK>", 3)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.unk_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "�") for i in ids)


def _build_v4_tokenizer_from_datasets(vocab_size: int) -> CharTokenizer:
    """Build a char tokenizer with coverage over eval datasets."""
    from datasets import load_dataset

    # Sample a few hundred examples from each eval dataset to build the
    # character set so no character is unknown at evaluation time.
    texts: list[str] = []

    try:
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for ex in wiki.select(range(min(500, len(wiki)))):
            texts.append(ex["text"])
    except Exception:
        pass

    try:
        hs = load_dataset("hellaswag", split="validation")
        for ex in hs.select(range(min(500, len(hs)))):
            texts.append(ex["ctx"])
            for end in ex["endings"]:
                texts.append(end)
    except Exception:
        pass

    # Fallback: include all printable ASCII so we always have a reasonable set
    if not texts:
        import string
        texts = [string.printable]

    stoi = _build_char_stoi(texts, vocab_size)
    return CharTokenizer(stoi)


# ═════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═════════════════════════════════════════════════════════════════════════════

def _infer_vocab_size(state_dict: dict) -> int:
    for key in ("tok_emb.weight", "head.weight"):
        w = state_dict.get(key)
        if w is not None:
            return w.shape[0]
    return 1024


def load_model(checkpoint_path: str, device: str, model_version: str = "v4", config_path: str = ""):
    """Load model + tokenizer from a checkpoint file.

    Returns (model, tokenizer, params).
    """
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # Unwrap nested checkpoint dictionaries ──────────────────────────────
    state_dict = ckpt
    loaded_stoi = None  # may be populated from .materia files

    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            # .materia files sometimes carry a tokenizer mapping
            loaded_stoi = ckpt.get("tokenizer")

    vocab_size = _infer_vocab_size(state_dict)
    is_v3 = (model_version == "v3") or (vocab_size >= 30000)

    # Load SentencePiece tokenizer for V3 ───────────────────────────────
    if is_v3:
        from materia_v3_full import MateriaV3Full, Tokenizer

        # Determine model dim from state_dict
        tok_emb = state_dict["tok_emb.weight"]
        dim = tok_emb.shape[1]

        # Infer n_layers from state dict keys
        layer_keys = [k for k in state_dict if k.startswith("layers.") and k.endswith(".attn_norm.weight")]
        n_layers = len(layer_keys)

        tok = Tokenizer()
        model = MateriaV3Full(vocab_size=vocab_size, dim=dim, n_layers=n_layers)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        return model, tok, params

    # Load V4 (char-level) ───────────────────────────────────────────────
    from materia_v4 import MateriaV4

    # Detect hyper-parameters from state_dict keys (heuristic)
    dim = state_dict["tok_emb.weight"].shape[1]
    head_weight = state_dict["head.weight"]
    # n_layers: count layer norms
    layer_keys = [k for k in state_dict if k.startswith("layers.") and k.endswith(".attn_norm.weight")]
    n_layers = len(layer_keys) if layer_keys else 3

    # Detect toroidal architecture (HexagonalTorus projections)
    is_toroidal = any("emb_to_jepa" in k for k in state_dict)

    # Try to load from config file if provided
    config_args = {}
    if config_path and os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            mc = yaml.safe_load(f).get('model', {})
        config_args = {
            'n_layers': mc.get('n_layers', n_layers),
            'latent_dim': mc.get('latent_dim', dim),
            'snn_dim': mc.get('snn_dim', 128),
            'snn_threshold': mc.get('snn_threshold', 0.001),
            'ssm_state': mc.get('ssm_state', 32),
            'n_cycles': mc.get('n_cycles', 2),
            'use_flash': mc.get('use_flash', False),
            'use_checkpointing': False,
        }

    if config_args:
        model = MateriaV4(
            vocab_size=vocab_size, dim=dim,
            n_layers=config_args['n_layers'],
            latent_dim=config_args['latent_dim'],
            snn_dim=config_args['snn_dim'],
            snn_threshold=config_args['snn_threshold'],
            ssm_state=config_args['ssm_state'],
            n_cycles=config_args['n_cycles'],
            use_flash=config_args['use_flash'],
            use_checkpointing=False,
        )
    elif is_toroidal:
        # Toroidal always has emb_to_jepa, t_to_jepa, snn_to_jepa, ssm_to_jepa
        n_heads = state_dict.get("layers.0.attn.wq.weight", torch.zeros(0)).shape[0]
        n_kv = state_dict.get("layers.0.attn.wk.weight", torch.zeros(0)).shape[0]
        latent_dim = state_dict["emb_to_jepa.to_latent.weight"].shape[0]
        snn_dim = state_dict.get("snn.w_in.weight", torch.zeros(0)).shape[0]
        ssm_state = state_dict.get("ssm.A", torch.zeros(0)).shape[0]
        use_flash = any("flash" in k.lower() for k in state_dict)
        model = MateriaV4(
            vocab_size=vocab_size, dim=dim, n_layers=n_layers,
            latent_dim=latent_dim, snn_dim=snn_dim, ssm_state=ssm_state,
            n_cycles=n_cycles, use_flash=use_flash,
            use_checkpointing=False,
        )
    else:
        model = MateriaV4(vocab_size=vocab_size, dim=dim, n_layers=n_layers)
    # Some checkpoints may be for older MateriaV3Torch-based V4s; load
    # with strict=False to skip mismatched/extra keys gracefully
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"[WARN] Strict loading failed ({len(e.args)} errs), trying non-strict ...", file=sys.stderr)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[WARN] Missing keys: {missing[:10]} ...", file=sys.stderr)
        if unexpected:
            print(f"[WARN] Unexpected keys: {unexpected[:10]} ...", file=sys.stderr)

    model.to(device)
    model.eval()

    # Build tokenizer ───────────────────────────────────────────────────
    if loaded_stoi is not None:
        tokenizer = CharTokenizer(loaded_stoi)
    else:
        tokenizer = _build_v4_tokenizer_from_datasets(vocab_size)

    params = sum(p.numel() for p in model.parameters())
    return model, tokenizer, params


# ═════════════════════════════════════════════════════════════════════════════
#  WikiText-2 Perplexity
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_wikitext2(model, tokenizer, device, max_samples: int | None = None) -> float:
    """Compute perplexity on WikiText-2 test split.

    Result is ``exp(avg_cross_entropy)``, where the average is over
    non-padding tokens.
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    total_loss = 0.0
    total_tokens = 0
    n_examples = 0

    # Resolve pad_id for both CharTokenizer (dict-based) and SentencePiece
    if hasattr(tokenizer, "stoi"):
        pad_id = tokenizer.stoi.get("<PAD>", 0)
    else:
        pad_id = getattr(tokenizer, "pad_id", 0)
    # Resolve encode method
    encode = tokenizer.encode if hasattr(tokenizer, "encode") else tokenizer.EncodeAsIds

    for example in dataset:
        text = example["text"]
        if not text.strip():
            continue

        ids = encode(text)
        if isinstance(ids, list):
            pass  # already list[int]
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()

        # Skip single-token or empty sequences
        if len(ids) < 2:
            continue

        inp = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
        tgt = torch.tensor([ids[1:]], dtype=torch.long, device=device)

        if hasattr(model, "forward") and hasattr(model, "generate"):
            logits, *_ = model(inp)
        else:
            logits = model(inp)

        vocab_size = logits.size(-1)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            tgt.view(-1),
            ignore_index=pad_id,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += (tgt != pad_id).sum().item()
        n_examples += 1

        if n_examples % 100 == 0:
            print(f"  [wiki] {n_examples:>5d} examples  running ppl={math.exp(total_loss / max(total_tokens, 1)):.2f}",
                  file=sys.stderr)

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss)
    print(f"  [wiki] DONE  {n_examples} examples  ppl={ppl:.2f}", file=sys.stderr)
    return ppl


# ═════════════════════════════════════════════════════════════════════════════
#  HellaSwag Accuracy
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_hellaswag(model, tokenizer, device, max_samples: int | None = None) -> float:
    """Compute accuracy on HellaSwag validation set.

    For each example the log-likelihood of every ending (conditioned on the
    context) is computed and the most likely ending is selected.  Accuracy =
    fraction of correct selections.
    """
    from datasets import load_dataset

    dataset = load_dataset("hellaswag", split="validation")
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    correct = 0
    total = 0
    encode = tokenizer.encode if hasattr(tokenizer, "encode") else tokenizer.EncodeAsIds

    for idx, example in enumerate(dataset):
        ctx = example["ctx"]
        endings = example["endings"]
        label = int(example["label"])

        ctx_ids = encode(ctx)
        if isinstance(ctx_ids, torch.Tensor):
            ctx_ids = ctx_ids.tolist()

        scores: list[float] = []
        for ending in endings:
            ending_ids = encode(ending)
            if isinstance(ending_ids, torch.Tensor):
                ending_ids = ending_ids.tolist()

            # Full sequence = context + ending, truncated to avoid OOM
            full_ids = ctx_ids + ending_ids
            if len(full_ids) < 2:
                scores.append(float("-inf"))
                continue

            if len(full_ids) > 768:
                full_ids = full_ids[-768:]

            inp = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            tgt = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

            if hasattr(model, "forward") and hasattr(model, "generate"):
                logits, *_ = model(inp)
            else:
                logits = model(inp)

            vocab_size = logits.size(-1)
            log_probs = F.log_softmax(logits, dim=-1)           # (1, T, V)

            # Score only the *ending* portion (excluding context)
            ctx_offset = len(ctx_ids) - 1                       # first token predicted *after* ctx start
            ending_slice = slice(max(ctx_offset, 0), log_probs.size(1))

            if ending_slice.start >= ending_slice.stop:
                scores.append(float("-inf"))
                continue

            tok_log_probs = log_probs[0, ending_slice].gather(
                1, tgt[0, ending_slice].unsqueeze(-1)
            ).squeeze(-1)                                        # (ending_len,)

            # Average log-probability per token (length-normalised)
            avg_ll = tok_log_probs.mean().item()
            scores.append(avg_ll)

        # Pick the ending with the highest score
        pred = max(range(len(scores)), key=lambda i: scores[i])
        if pred == label:
            correct += 1
        total += 1

        if (idx + 1) % 100 == 0:
            interim_acc = correct / max(total, 1)
            print(f"  [hellaswag] {idx + 1:>5d}/{len(dataset)}  acc={interim_acc:.4f}",
                  file=sys.stderr)

    acc = correct / max(total, 1)
    print(f"  [hellaswag] DONE  {total} examples  acc={acc:.4f}", file=sys.stderr)
    return acc


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MATERIA evaluation benchmarks (WikiText-2 PPL, HellaSwag Acc)",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        required=True,
        help="Path to .pt, .pth, or .materia checkpoint",
    )
    parser.add_argument(
        "--model",
        choices=["v3", "v4", "auto"],
        default="auto",
        help="Model version (auto-detected from vocab size)",
    )
    parser.add_argument(
        "--device", "-d",
        default="auto",
        help="Device: 'cuda', 'cpu', or 'auto'",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=None,
        help="Limit number of eval samples per benchmark (for quick smoke-tests)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Save JSON results to this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/V4_210M_BPE.yaml",
        help="Path to YAML config (for toroidal model parameters)",
    )
    args = parser.parse_args()

    # ---- Device resolution ------------------------------------------------
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[CONFIG] device={device}  checkpoint={args.checkpoint}  model={args.model}  samples={args.samples}",
          file=sys.stderr)

    # ---- Load model -------------------------------------------------------
    model_version = args.model
    if model_version == "auto":
        # Heuristic: peek at vocab size
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            for key in ("model_state_dict", "state_dict"):
                sd = ckpt.get(key)
                if sd is not None:
                    break
            else:
                sd = ckpt  # raw state dict
        else:
            sd = ckpt  # raw
        vs = _infer_vocab_size(sd)
        model_version = "v3" if vs >= 30000 else "v4"
        print(f"[CONFIG] Auto-detected model_version={model_version} (vocab_size={vs})", file=sys.stderr)
        del ckpt

    print(f"[LOAD] Loading model from {args.checkpoint} ...", file=sys.stderr)
    t0 = time.time()
    model, tokenizer, n_params = load_model(args.checkpoint, device, model_version, args.config or "")
    load_time = time.time() - t0
    print(f"[LOAD] Done in {load_time:.1f}s — {n_params:,} params", file=sys.stderr)

    # ---- WikiText-2 -------------------------------------------------------
    print(f"\n[BENCH] WikiText-2 perplexity ...", file=sys.stderr)
    t0 = time.time()
    ppl = evaluate_wikitext2(model, tokenizer, device, args.samples)
    wiki_time = time.time() - t0
    print(f"[BENCH] WikiText-2 PPL = {ppl:.2f}  ({wiki_time:.1f}s)", file=sys.stderr)

    # ---- HellaSwag --------------------------------------------------------
    print(f"\n[BENCH] HellaSwag accuracy ...", file=sys.stderr)
    t0 = time.time()
    acc = evaluate_hellaswag(model, tokenizer, device, args.samples)
    hs_time = time.time() - t0
    print(f"[BENCH] HellaSwag Acc = {acc:.4f}  ({hs_time:.1f}s)", file=sys.stderr)

    # ---- Report -----------------------------------------------------------
    total_time = load_time + wiki_time + hs_time
    report = {
        "hellaswag_acc": round(acc, 6),
        "wikitext_ppl": round(ppl, 4),
        "params": n_params,
        "device": device,
        "time": round(total_time, 2),
    }
    json_out = json.dumps(report, indent=2, ensure_ascii=False)
    print(f"\n{'=' * 52}", file=sys.stderr)
    print(json_out)
    if args.output:
        Path(args.output).write_text(json_out)
        print(f"Saved to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
