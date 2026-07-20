"""
Export MATERIA V4 .materia weights from checkpoint
"""
import os, sys, yaml, pickle, json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from materia_v4 import MateriaV4

MATERIA_HOME = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))

def load_text_data(filepaths, max_lines=5000):
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
    chars = sorted(chars)[:vocab_size - 4]
    stoi = {ch: i + 4 for i, ch in enumerate(chars)}
    stoi['<PAD>'] = 0; stoi['<BOS>'] = 1; stoi['<EOS>'] = 2; stoi['<UNK>'] = 3
    return stoi, {i: ch for ch, i in stoi.items()}

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--output', '-o', type=str, default=None)
    parser.add_argument('--max-lines', type=int, default=5000)
    args = parser.parse_args()

    ckpt_path = args.checkpoint
    output_dir = os.path.dirname(ckpt_path)
    config_path = os.path.join(output_dir, 'config.yaml')

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    max_lines = args.max_lines
    data_dir = os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
    data_files = [os.path.join(data_dir, 'c4_en.txt'),
                  os.path.join(data_dir, 'combined_for_spm.txt')]
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.startswith('wiki_') and f.endswith('.txt'):
                data_files.append(os.path.join(data_dir, f))

    texts = load_text_data(data_files, max_lines)
    vocab_size = cfg['model'].get('vocab_size', 1024)
    stoi, itos = build_char_tokenizer(texts[:10000], vocab_size)
    vocab_size = len(stoi)

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    mc = cfg['model']

    model = MateriaV4(
        vocab_size=vocab_size,
        dim=mc.get('dim', 256),
        n_layers=mc.get('n_layers', 3),
        n_heads=mc.get('n_heads', 8),
        n_kv=mc.get('n_kv', 4),
        latent_dim=mc.get('latent_dim', 256),
        snn_dim=mc.get('snn_dim', 128),
        snn_threshold=mc.get('snn_threshold', 0.005),
        snn_tau=mc.get('snn_tau', 0.8),
        ssm_state=mc.get('ssm_state', 32),
        synapsis_slots=mc.get('synapsis_slots', 128),
        hsaq_sparsity=mc.get('hsaq_sparsity', 0.3),
        jepa_weight=mc.get('jepa_weight', 2.781042),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    total = sum(p.numel() for p in model.parameters())

    materia_path = args.output or os.path.join(output_dir, 'materia-v4.materia')
    weight_data = {
        'config': {'vocab_size': vocab_size, 'dim': model.dim,
                    'version': 'V4', 'latent_dim': model.latent_dim,
                    'K': 2.781042, 'params': total},
        'state_dict': {k: v.numpy() for k, v in model.state_dict().items()},
        'tokenizer': stoi,
        'stats': {
            'train_loss': ckpt['stats']['train_loss'],
            'val_loss': ckpt['stats']['val_loss'],
            'val_acc': ckpt['stats']['val_acc'],
        },
    }
    with open(materia_path, 'wb') as f:
        pickle.dump(weight_data, f)
    print(f"Exported: {materia_path}")
    print(f"  Vocab: {vocab_size} | Params: {total:,} ({total*4/1024**2:.1f}MB)")
    print(f"  Size: {os.path.getsize(materia_path)//1024}KB")
    print(f"  Epoch: {ckpt.get('epoch', '?')}")
