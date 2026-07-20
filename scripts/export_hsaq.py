"""
HSAQ export utilities: ONNX, TorchScript, and serializable hooks.

Replaces ``torch.kthvalue`` (not in ONNX opset) with ``torch.topk``
for export, and provides a ``SerializableQuantHook`` that survives
``torch.jit.save`` / ``torch.jit.script`` (the closure-based hooks
from ``HSAQ._make_hook`` are not serializable by default).

Usage::

    from export_hsaq import (
        export_to_onnx,
        export_to_torchscript,
        add_hook_export_serializable,
    )

    model = MateriaV4(...)
    model.load_state_dict(torch.load('ckpt.pt')['model_state_dict'])
    model.eval()

    # ONNX
    export_to_onnx(model, 'model.onnx', example_input=torch.randint(0, 100, (1, 64)))

    # TorchScript
    export_to_torchscript(model, 'model.pt', method='trace',
                          example_input=torch.randint(0, 100, (1, 64)))
"""
import os
import sys
import types
from typing import Any, Callable, Optional

import torch
import torch.nn as nn

# ── Path setup ─────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.normpath(os.path.join(_SCRIPTS_DIR, ".."))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from models.core.hsaq import HSAQ, _ste_round


# ═══════════════════════════════════════════════════════════════════
#  1. ONNX export — kthvalue → topk replacement
# ═══════════════════════════════════════════════════════════════════

def _kthvalue_via_topk(x: torch.Tensor, k: int, dim: int = 1) -> torch.Tensor:
    """ONNX-compatible ``kthvalue`` using ``torch.topk`` (opset 11+).

    ``kthvalue`` returns the k-th smallest value along a dimension.
    ONNX does not define a ``KthValue`` operator, but it does define
    ``TopK``.  Negating the input turns "k-th smallest" into
    "k-th largest of the negated input".
    """
    topk_vals = torch.topk(-x, k, dim=dim).values  # [B, k]
    # Index the k-th element (last along dim)
    idx = [slice(None)] * topk_vals.dim()
    idx[dim] = slice(-1, None)
    return -topk_vals[tuple(idx)].squeeze(dim)


class HSAQExportWrapper(nn.Module):
    """Wrapper that makes ``HSAQ.forward`` exportable to ONNX.

    The only change vs the original ``HSAQ.forward`` is that
    ``torch.kthvalue`` is replaced by ``_kthvalue_via_topk``
    (backed by ``torch.topk``, which *is* in the ONNX operator set).

    After export, you can restore the original module::

        wrapper = HSAQExportWrapper(model.hsaq)
        model.hsaq = wrapper          # replace for export
        export_to_onnx(model, ...)
        model.hsaq = wrapper.hsaq     # restore
    """

    def __init__(self, hsaq: HSAQ):
        super().__init__()
        self.hsaq = hsaq

    @property
    def sparsity(self) -> float:
        return self.hsaq.sparsity

    @sparsity.setter
    def sparsity(self, val: float) -> None:
        self.hsaq.sparsity = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Same logic as ``HSAQ.forward`` but with ``topk`` instead of ``kthvalue``."""
        if self.hsaq.sparsity <= 0.0:
            return x

        flat = x.abs().view(x.size(0), -1)
        n = flat.size(1)
        k = max(1, min(n - 1, int(n * (1.0 - self.hsaq.sparsity))))

        # ONNX-safe: topk on negated values
        thresh = _kthvalue_via_topk(flat, k, dim=1)
        thresh = thresh.view(-1, *([1] * (x.dim() - 1)))
        return x * (x.abs() >= thresh)


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    example_input: Optional[torch.Tensor] = None,
    opset_version: int = 17,
    replace_kthvalue: bool = True,
    input_names: Optional[list[str]] = None,
    output_names: Optional[list[str]] = None,
    dynamic_axes: Optional[dict] = None,
    verbose: bool = False,
) -> str:
    """Export the model to ONNX.

    HSAQ-specific handling:

    * ``torch.kthvalue`` is replaced with ``torch.topk`` (opset 11+)
      when ``replace_kthvalue=True`` (the default).
    * The sparse mask becomes a simple element-wise multiply-by-mask
      in the exported graph.

    Args:
        model: PyTorch model in ``.eval()`` state.
        output_path: Destination ``.onnx`` path.
        example_input: Example input tensor (e.g. ``torch.randint(0, V, (1, T))``
            for token models).  Falls back to ``torch.randn(1, 64)``.
        opset_version: ONNX opset (default 17).
        replace_kthvalue: Automatically wrap ``HSAQ`` submodules in
            ``HSAQExportWrapper`` before exporting.
        input_names: ONNX graph input names.  Defaults to ``["input"]``.
        output_names: ONNX graph output names.  Defaults to ``["output"]``.
        dynamic_axes: Dict for variable-length axes
            (e.g. ``{"input": {0: "batch", 1: "seq"}}``).
        verbose: Print ONNX export diagnostics.

    Returns:
        Absolute path of the exported ``.onnx`` file.
    """
    model.eval()
    device = next(model.parameters()).device

    if example_input is None:
        example_input = torch.randn(1, 64, device=device)

    if input_names is None:
        input_names = ["input"]
    if output_names is None:
        output_names = ["output"]

    # ── Replace HSAQ modules with ONNX-compatible wrappers ──
    replacements: dict[HSAQ, HSAQExportWrapper] = {}
    if replace_kthvalue:
        for name, child in model.named_children():
            if isinstance(child, HSAQ):
                wrapper = HSAQExportWrapper(child)
                replacements[child] = wrapper
                setattr(model, name, wrapper)
                if verbose:
                    print(f"  [ONNX] Replaced {name}: HSAQ → HSAQExportWrapper")

    try:
        torch.onnx.export(
            model,
            example_input,
            output_path,
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            verbose=verbose,
        )
    finally:
        # ── Restore original modules ──
        for orig, wrapper in replacements.items():
            for name, child in model.named_children():
                if child is wrapper:
                    setattr(model, name, orig)
                    break

    abs_path = os.path.abspath(output_path)
    size_kb = os.path.getsize(abs_path) // 1024
    if verbose:
        print(f"  [ONNX] Exported: {abs_path} ({size_kb} KB)")
    return abs_path


# ═══════════════════════════════════════════════════════════════════
#  2. TorchScript export
# ═══════════════════════════════════════════════════════════════════

class SerializableQuantHook(nn.Module):
    """``forward_pre_hook`` serializable for TorchScript.

    Replaces the closure-based ``_make_hook`` from ``HSAQ``, which
    cannot be pickled / saved by ``torch.jit.save``.

    Stores quantization parameters as buffers so they appear in the
    module's state dict and survive serialization.

    Usage::

        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                mod._forward_pre_hooks.clear()
                mod.register_forward_pre_hook(SerializableQuantHook(hsaq))
    """

    def __init__(self, hsaq: HSAQ):
        super().__init__()
        self.weight_bits = hsaq.weight_bits
        self.weight_quant_mode = hsaq.weight_quant_mode
        self.weight_tying = hsaq.weight_tying

        # Buffers for tied-scale state (needed for TorchScript serialization)
        if hsaq._tied_scale is not None:
            self.register_buffer("_tied_scale", hsaq._tied_scale.clone())
        else:
            self.register_buffer("_tied_scale", torch.zeros(1))
        self.register_buffer("_tied_scale_bits", torch.tensor(hsaq._tied_scale_bits, dtype=torch.long))

    def forward(self, module: nn.Module, args: tuple) -> None:
        """Quantize ``module.weight`` in-place (forward_pre_hook signature)."""
        if self.weight_bits <= 0 or self.weight_bits >= 16:
            return

        bits = self.weight_bits
        qmax = 2 ** (bits - 1) - 1

        if self.weight_quant_mode == "per_tensor":
            if (
                self.weight_tying
                and self._tied_scale is not None
                and self._tied_scale_bits.item() == bits
            ):
                scale = self._tied_scale
            else:
                amax = module.weight.abs().max()
                scale = (
                    amax / qmax
                    if amax > 1e-10
                    else torch.tensor(1.0, device=module.weight.device)
                )
            scale = scale.clamp(min=1e-10)
            q = _ste_round(module.weight / scale)
            q = torch.clamp(q, -qmax, qmax)
            module.weight.data = q * scale
        elif self.weight_quant_mode == "per_channel":
            weight = module.weight
            shape = weight.shape
            x_flat = weight.view(shape[0], -1)
            amax = x_flat.abs().max(dim=1).values
            s = amax / qmax
            s = s.clamp(min=1e-10).view(-1, *([1] * (weight.dim() - 1)))
            q = _ste_round(weight / s)
            q = torch.clamp(q, -qmax, qmax)
            module.weight.data = q * s


def export_to_torchscript(
    model: nn.Module,
    output_path: str,
    method: str = "trace",
    example_input: Optional[torch.Tensor] = None,
    optimize: bool = True,
    make_hooks_serializable: bool = True,
) -> str:
    """Export the model to TorchScript.

    Args:
        model: PyTorch model in ``.eval()`` state.
        output_path: Destination ``.pt`` path.
        method: ``"trace"`` (default, recommended) or ``"script"``.
        example_input: Example input for tracing.  Required for
            ``method="trace"``.  Falls back to ``torch.randn(1, 64)``.
        optimize: Apply ``torch.jit.optimize_for_inference``.
        make_hooks_serializable: Replace closure-based hooks on
            ``nn.Linear`` with ``SerializableQuantHook`` instances.
            This is necessary for ``method="script"`` and recommended
            for ``method="trace"`` to produce a saveable graph.

    Returns:
        Absolute path of the exported ``.pt`` file.
    """
    model.eval()
    device = next(model.parameters()).device

    if example_input is None:
        example_input = torch.randn(1, 64, device=device)

    # ── Make hooks serializable ──
    if make_hooks_serializable:
        _replace_hooks_with_serializable(model)

    # ── Export ──
    if method == "trace":
        traced = torch.jit.trace(model, example_input, check_trace=False)
        scripted = traced
    elif method == "script":
        scripted = torch.jit.script(model)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'trace' or 'script'.")

    if optimize:
        scripted = torch.jit.optimize_for_inference(scripted)

    torch.jit.save(scripted, output_path)
    abs_path = os.path.abspath(output_path)
    size_kb = os.path.getsize(abs_path) // 1024
    print(f"  [TorchScript] Exported: {abs_path} ({size_kb} KB)")
    return abs_path


# ═══════════════════════════════════════════════════════════════════
#  3. Serializable hooks — modify HSAQ instance in-place
# ═══════════════════════════════════════════════════════════════════

def _replace_hooks_with_serializable(model: nn.Module) -> int:
    """Replace every closure-based HSAQ hook on ``nn.Linear`` with
    ``SerializableQuantHook``.

    Returns the number of hooks replaced.
    """
    hsaq = _find_hsaq(model)
    if hsaq is None:
        return 0

    serial_hook = SerializableQuantHook(hsaq)
    count = 0
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            mod._forward_pre_hooks.clear()
            mod.register_forward_pre_hook(serial_hook)
            count += 1
    return count


def add_hook_export_serializable(hsaq: HSAQ) -> None:
    """Modify an ``HSAQ`` instance so its hooks survive serialization.

    After calling this on an HSAQ instance, subsequent calls to
    ``hsaq.attach_to(model)`` will register ``SerializableQuantHook``
    modules instead of closure-based hooks.  This makes the parent
    model exportable via ``torch.jit.script`` / ``torch.jit.trace``
    and serializable with ``torch.jit.save``.

    Example::

        model = MateriaV4(...)
        add_hook_export_serializable(model.hsaq)
        model.hsaq.attach_to(model)     # registers serializable hooks
        traced = torch.jit.trace(model, example_input)
        torch.jit.save(traced, "model.pt")

    The modification is **in-place** and **non-destructive** — the
    original ``_make_hook`` and ``attach_to`` are preserved as
    ``_orig_make_hook`` / ``_orig_attach`` so they can be restored.
    """
    # ── Preserve originals ──
    hsaq._orig_make_hook = hsaq._make_hook  # type: ignore[attr-defined]
    hsaq._orig_attach = hsaq.attach_to  # type: ignore[attr-defined]

    def _serializable_make_hook(self) -> nn.Module:
        if self.weight_bits <= 0 or self.weight_bits >= 16:
            return nn.Identity()
        return SerializableQuantHook(self)

    def _serializable_attach(self, model: nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
        self.detach()
        hook_module = self._make_hook()  # type: ignore[return-value]
        if isinstance(hook_module, nn.Identity):
            return self._hooks
        for _, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                handle = mod.register_forward_pre_hook(hook_module)  # type: ignore[arg-type]
                self._hooks.append(handle)
        return self._hooks

    # Bind overrides to this instance
    hsaq._make_hook = types.MethodType(_serializable_make_hook, hsaq)  # type: ignore[method-assign]
    hsaq.attach_to = types.MethodType(_serializable_attach, hsaq)  # type: ignore[method-assign]


# ═══════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════

def _find_hsaq(model: nn.Module) -> Optional[HSAQ]:
    """Find the first HSAQ submodule."""
    for mod in model.modules():
        if isinstance(mod, HSAQ):
            return mod
    return None


def __getattr__(name: str) -> Any:
    """Allow ``from export_hsaq import HSAQ`` for convenience."""
    if name == "HSAQ":
        return HSAQ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ═══════════════════════════════════════════════════════════════════
#  4. CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export HSAQ-equipped model to ONNX / TorchScript"
    )
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True,
        help="Path to model checkpoint (.pt / .pth)",
    )
    parser.add_argument(
        "--model-class", type=str, default="materia_v4.MateriaV4",
        help="Dotted import path to the model class (default: materia_v4.MateriaV4)",
    )
    parser.add_argument(
        "--model-kwargs", type=str, default=None,
        help="JSON string of kwargs for the model constructor",
    )
    parser.add_argument(
        "--format", "-f", type=str, choices=["onnx", "torchscript", "both"],
        default="onnx",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output path (without extension)",
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset version (default 17)",
    )
    parser.add_argument(
        "--method", type=str, choices=["trace", "script"], default="trace",
        help="TorchScript export method (default trace)",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device for export (default cpu)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
    )
    args = parser.parse_args()

    # ── Import model class ──
    import importlib
    mod_path, cls_name = args.model_class.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    ModelCls: type[nn.Module] = getattr(mod, cls_name)

    # ── Load checkpoint & build model ──
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    model_kwargs: dict[str, Any] = {}
    if args.model_kwargs:
        import json
        model_kwargs = json.loads(args.model_kwargs)

    # If config is bundled in checkpoint, try to use it
    if "model_config" in ckpt and not model_kwargs:
        model_kwargs = ckpt["model_config"]

    model = ModelCls(**model_kwargs)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(sd, strict=False)
    model.eval()

    # ── Prepare output path ──
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]
    out_stem = args.output or os.path.join(os.path.dirname(args.checkpoint), ckpt_stem)

    # ── Build example input ──
    B, T = 1, 64
    vocab_size = getattr(model, "vocab_size", 1024) or model_kwargs.get("vocab_size", 1024)
    example_input = torch.randint(0, min(vocab_size, 100), (B, T), device=device)

    # ── Export ──
    if args.format in ("onnx", "both"):
        export_to_onnx(
            model,
            f"{out_stem}.onnx",
            example_input=example_input,
            opset_version=args.opset,
            verbose=args.verbose,
        )

    if args.format in ("torchscript", "both"):
        export_to_torchscript(
            model,
            f"{out_stem}.pt",
            method=args.method,
            example_input=example_input,
            make_hooks_serializable=True,
        )
