"""
MATERIA V4 — AWQ Calibration for HSAQ
======================================
CalibraciÃ³n Activation-Aware Weight Quantization para el mÃ³dulo HSAQ
(HyperSparse Adaptive Quantization).

Uso:
    python scripts/calibrate_hsaq.py --checkpoint outputs/experiment/checkpoint_epoch8.pt \\
                                     --output models/calibrated/materia-v4-awq.pt

Dependencias: torch, numpy
"""
import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from materia_v4 import MateriaV4

MATERIA_HOME = os.environ.get(
    'MATERIA_HOME',
    os.path.normpath(os.path.join(os.path.dirname(__file__), '..')),
)

log = lambda msg: print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers de datos (gemelos a train_v4.py / inference_v4.py)
# ─────────────────────────────────────────────────────────────────────────────


class CharTextDataset(Dataset):
    """Dataset de caracteres — idÃ©ntico al usado en train_v4.py."""

    def __init__(self, texts, stoi, seq_len=64):
        self.seq_len = seq_len
        self.data = []
        for text in texts:
            ids = [stoi.get(c, 3) for c in text]
            if len(ids) > seq_len + 1:
                for i in range(0, len(ids) - seq_len, seq_len // 2):
                    self.data.append(ids[i : i + seq_len + 1])

    def __len__(self):
        return max(1, len(self.data))

    def __getitem__(self, idx):
        ids = self.data[idx % len(self.data)][: self.seq_len + 1]
        ids = ids + [0] * (self.seq_len + 1 - len(ids))
        ids = torch.tensor(ids[: self.seq_len + 1], dtype=torch.long)
        return ids[:-1], ids[1:]


def load_text_data(filepaths, max_lines=80000):
    texts = []
    for fp in filepaths:
        if not os.path.exists(fp):
            continue
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if len(line) > 50:
                    texts.append(line)
                    if len(texts) >= max_lines:
                        break
    return texts


def build_char_tokenizer(texts, vocab_size=1024):
    chars = set()
    for t in texts:
        chars.update(t)
    chars = sorted(chars)[: vocab_size - 4]
    stoi = {ch: i + 4 for i, ch in enumerate(chars)}
    stoi['<PAD>'] = 0
    stoi['<BOS>'] = 1
    stoi['<EOS>'] = 2
    stoi['<UNK>'] = 3
    return stoi, {i: ch for ch, i in stoi.items()}


def build_calib_loader(
    data_dir: str,
    max_lines: int = 5000,
    seq_len: int = 64,
    batch_size: int = 8,
    num_workers: int = 0,
):
    """Construye un DataLoader de calibraciÃ³n desde archivos de texto.

    Reconstruye el tokenizer de la misma forma que training, para que
    los ids coincidan con el vocabulario del modelo.
    """
    data_files = [
        os.path.join(data_dir, 'c4_en.txt'),
        os.path.join(data_dir, 'combined_for_spm.txt'),
    ]
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.startswith('wiki_') and f.endswith('.txt'):
                data_files.append(os.path.join(data_dir, f))

    texts = load_text_data(data_files, max_lines)
    texts_for_vocab = texts[:10000]

    for vocab_size in range(1024, 50, -1):
        stoi, _ = build_char_tokenizer(texts_for_vocab, vocab_size)
        if len(stoi) >= 256:
            break

    ds = CharTextDataset(texts, stoi, seq_len)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
    )
    return loader, stoi


# ─────────────────────────────────────────────────────────────────────────────
#  AWQ Calibration (pÃºblica)
# ─────────────────────────────────────────────────────────────────────────────


def calibrate_hsaq_model(
    model: MateriaV4,
    calib_loader,
    alpha: float = 0.5,
    device: str | torch.device = 'cuda',
    output_path: str | None = None,
) -> dict[str, torch.Tensor]:
    """CalibraciÃ³n AWQ completa para un modelo MateriaV4 entrenado.

    La funciÃ³n delega la recolecciÃ³n de estadÃ­sticas y el re-escalado de
    pesos al mÃ©todo ``HSAQ.calibrate_awq()`` ya implementado en
    ``models/core/hsaq.py``.

    Pasos
    -----
    1. Forward passes sobre el lote de calibraciÃ³n para recolectar
       la magnitud promedio de activaciones por canal de entrada de cada
       ``nn.Linear``.
    2. Re-escalado de pesos:  W' = W / (act_scale ** alpha)
       Esto amplifica los canales con alta activaciÃ³n antes de la
       cuantizaciÃ³n, reduciendo el error en los canales importantes.
    3. Las escalas AWQ se registran como **buffers persistentes** en el
       mÃ³dulo ``HSAQ``, lo que las incluye automÃ¡ticamente en
       ``model.state_dict()`` y se preservan al guardar/recargar.
    4. Opcionalmente guarda un checkpoint calibrado separado.

    ParÃ¡metros
    ----------
    model:
        Modelo ``MateriaV4`` entrenado (debe estar en modo ``eval``).
    calib_loader:
        ``DataLoader`` que produce batches de entrada (tensor o tupla
        ``(input, target)``).  Se usa solo el input para la calibraciÃ³n.
    alpha:
        Exponente de escalado AWQ (tÃ­pico 0.3–1.0).  ``0.5`` es el
        valor estÃ¡ndar en la literatura (SmoothQuant / AWQ).
    device:
        Dispositivo donde ejecutar los forward passes de calibraciÃ³n.
    output_path:
        Si se especifica, guarda aquÃ­ un checkpoint que incluye el
        state_dict calibrado y los metadatos de calibraciÃ³n.

    Retorna
    -------
    ``dict {nombre_mÃ³dulo: escala_1D}`` con las escalas de activaciÃ³n
    por canal de entrada para cada ``nn.Linear`` calibrado.
    """
    model.eval()
    model.to(device)

    n_batches = len(calib_loader)
    log(f"Iniciando calibraciÃ³n AWQ (alpha={alpha}, {n_batches} batches)...")

    # ── 1. Recolectar escalas y re-escalar pesos ─────────────────────────
    #     Delega en HSAQ.calibrate_awq() que:
    #       a) Registra hooks en todos los nn.Linear
    #       b) Ejecuta forward passes
    #       c) W' = W / (act_scale ** alpha)
    act_scales = model.hsaq.calibrate_awq(model, calib_loader, alpha=alpha, device=device)

    n_scales = len(act_scales)
    log(f"  {n_scales} mÃ³dulos lineales calibrados")

    # ── 2. Registrar escalas en el state_dict (buffers persistentes) ────
    #      AsÃ­ sobreviven a torch.save(model.state_dict()).
    _register_awq_scales(model, act_scales)

    # ── 3. Guardar checkpoint calibrado ──────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        calib_meta = {
            'alpha': alpha,
            'n_scales': n_scales,
            'awq_scale_keys': list(act_scales.keys()),
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }
        torch.save(
            {'model_state_dict': model.state_dict(), 'calibration': calib_meta},
            output_path,
        )
        log(f"  Checkpoint calibrado guardado:  {output_path}")
        mb = os.path.getsize(output_path) / 1024**2
        log(f"  TamaÃ±o del archivo:             {mb:.1f} MB")

    return act_scales


# ─────────────────────────────────────────────────────────────────────────────
#  Registro de escalas como buffers persistentes
# ─────────────────────────────────────────────────────────────────────────────


def _register_awq_scales(model: MateriaV4, act_scales: dict[str, torch.Tensor]) -> None:
    """Registra las escalas AWQ como ``persistent_buffer`` en el mÃ³dulo HSAQ.

    Cada entrada ``(ruta_mÃ³dulo, tensor_escala)`` se registra como un
    buffer con nombre ``awq_scale_<ruta_sanitizada>``.  El sanitizado
    convierte ``'.'`` → ``'_'`` para obtener identificadores vÃ¡lidos.

    Los buffers persistentes aparecen en ``model.state_dict()`` con clave
    ``hsaq.awq_scale_<ruta>``, por lo que se preservan al guardar con
    ``torch.save()`` y se restauran automÃ¡ticamente al hacer
    ``load_state_dict()``.
    """
    # Limpiar registros previos
    stale = [k for k in list(model.hsaq._buffers) if k.startswith('awq_scale_')]
    for k in stale:
        del model.hsaq._buffers[k]

    for module_path, scale_tensor in act_scales.items():
        safe_name = f"awq_scale_{module_path.replace('.', '_')}"
        model.hsaq.register_buffer(safe_name, scale_tensor.detach().cpu(), persistent=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Carga de modelo calibrado
# ─────────────────────────────────────────────────────────────────────────────


def load_calibrated_model(
    checkpoint_path: str,
    calib_loader=None,
    alpha: float = 0.5,
    device: str | torch.device = 'cpu',
    output_path: str | None = None,
) -> tuple[MateriaV4, dict | None]:
    """Carga un checkpoint y restaura las escalas AWQ si estÃ¡n presentes.

    Si el checkpoint contiene buffers ``hsaq.awq_scale_*``, estos se
    restauran automÃ¡ticamente al hacer ``load_state_dict()`` y las
    escalas ya estÃ¡n disponibles en ``model.hsaq.awq_scales``.

    Si el checkpoint **no** tiene escalas pero se provee un
    ``calib_loader``, se ejecuta calibraciÃ³n en caliente como fallback.

    Retorna
    -------
    ``(modelo, metadatos_calibraciÃ³n | None)``
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    vocab_size = ckpt['model_state_dict']['tok_emb.weight'].shape[0]
    model = MateriaV4(vocab_size=vocab_size)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.eval()

    calib_meta = ckpt.get('calibration')

    # Detectar si el state_dict trae escalas AWQ
    has_scales = any(k.startswith('hsaq.awq_scale_') for k in ckpt['model_state_dict'])

    if not has_scales and calib_loader is not None:
        log("  Escalas AWQ no encontradas en checkpoint. Calibrando en caliente...")
        calibrate_hsaq_model(model, calib_loader, alpha=alpha, device=device, output_path=output_path)
    elif has_scales:
        scale_count = sum(1 for k in ckpt['model_state_dict'] if 'awq_scale_' in k)
        log(f"  Escalas AWQ restauradas desde checkpoint ({scale_count} mÃ³dulos)")
    else:
        log("  AVISO: Checkpoint sin escalas AWQ y sin calib_loader. Ejecute calibrate_hsaq_model().")

    model.to(device)
    return model, calib_meta


# ─────────────────────────────────────────────────────────────────────────────
#  EvaluaciÃ³n de error de cuantizaciÃ³n
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _perplexity(model: MateriaV4, loader, device: str | torch.device = 'cuda') -> float:
    """Calcula perplejidad (exp(loss)) del modelo sobre el DataLoader completo."""
    model.eval()
    total_loss = 0.0
    n_tokens = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=0, reduction='sum'
        )
        total_loss += loss.item()
        n_tokens += (y != 0).sum().item()
        del logits, loss, x, y
    avg_loss = total_loss / max(1, n_tokens)
    return float(torch.exp(torch.tensor(avg_loss)).item())


def evaluate_quantization_error(
    model: MateriaV4,
    calib_loader,
    device: str | torch.device = 'cuda',
    alpha: float = 0.5,
) -> dict:
    """Mide el error de cuantizaciÃ³n introducido por AWQ.

    1. Calcula la perplejidad del modelo **original** (sin calibrar).
    2. Aplica calibraciÃ³n AWQ (re-escala pesos *in-place*).
    3. Calcula la perplejidad del modelo **calibrado**.
    4. Restaura los pesos originales.
    5. Retorna un reporte JSON con las diferencias.

    Retorna
    -------
    ``dict`` con las llaves:
        ``original_ppl``, ``calibrated_ppl``, ``error_increase``,
        ``relative_error_pct``, ``alpha``.
    """
    log("Evaluando error de cuantizaciÃ³n AWQ ...")

    # 1. Clonar state_dict del modelo original
    orig_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 2. Perplejidad original
    orig_ppl = _perplexity(model, calib_loader, device)
    log(f"  Perplejidad original:            {orig_ppl:.4f}")

    # 3. Aplicar AWQ
    calibrate_hsaq_model(model, calib_loader, alpha=alpha, device=device, output_path=None)

    # 4. Perplejidad calibrada
    calib_ppl = _perplexity(model, calib_loader, device)
    log(f"  Perplejidad calibrada (Î±={alpha}): {calib_ppl:.4f}")

    # 5. Restaurar pesos originales
    model.load_state_dict(orig_state, strict=False)

    error_inc = calib_ppl - orig_ppl
    rel_err = (error_inc / max(1e-10, orig_ppl)) * 100.0

    results = {
        'original_ppl': round(orig_ppl, 4),
        'calibrated_ppl': round(calib_ppl, 4),
        'error_increase': round(error_inc, 4),
        'relative_error_pct': round(rel_err, 4),
        'alpha': alpha,
    }
    log(f"  Reporte:\n{json.dumps(results, indent=2)}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description='MATERIA V4 â€” AWQ Calibration for HSAQ')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                        help='Ruta al checkpoint entrenado (.pt)')
    parser.add_argument('--output', '-o', type=str,
                        default=os.path.join(MATERIA_HOME, 'models', 'calibrated', 'materia-v4-awq.pt'),
                        help='Ruta de salida para el checkpoint calibrado')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Exponente AWQ (0.3â€“1.0, default: 0.5)')
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=64)
    parser.add_argument('--max-lines', type=int, default=5000,
                        help='MÃ¡ximo de lÃ­neas de texto para calibraciÃ³n')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Directorio con datos de texto')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluar error de cuantizaciÃ³n antes de guardar')
    parser.add_argument('--eval-report', type=str, default=None,
                        help='Ruta para guardar el reporte JSON de evaluaciÃ³n')
    args = parser.parse_args()

    device = torch.device(args.device)
    log(f"Dispositivo:                     {device}")
    log(f"Checkpoint:                      {args.checkpoint}")

    # ── Cargar datos ────────────────────────────────────────────────────
    data_dir = args.data_dir or os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
    calib_loader, stoi = build_calib_loader(
        data_dir,
        max_lines=args.max_lines,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    log(f"DataLoader:                      {len(calib_loader)} batches, vocab={len(stoi)}")

    # ── Cargar modelo (con fallback a calibraciÃ³n si no tiene escalas) ──
    model, calib_meta = load_calibrated_model(
        args.checkpoint,
        calib_loader=calib_loader,
        alpha=args.alpha,
        device=device,
        output_path=args.output,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Modelo:                          MateriaV4 ({n_params:,} params)")

    # ── EvaluaciÃ³n opcional ─────────────────────────────────────────────
    if args.eval:
        results = evaluate_quantization_error(model, calib_loader, device=device, alpha=args.alpha)
        if args.eval_report:
            os.makedirs(os.path.dirname(args.eval_report) or '.', exist_ok=True)
            with open(args.eval_report, 'w') as f:
                json.dump(results, f, indent=2)
            log(f"Reporte JSON guardado en:        {args.eval_report}")
    else:
        calibrate_hsaq_model(model, calib_loader, alpha=args.alpha, device=device, output_path=args.output)

    log("CalibraciÃ³n AWQ completada.")


if __name__ == '__main__':
    main()
