"""
MATERIA V4 - Inference from checkpoint
"""
import os, sys, yaml, pickle, json
import torch

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
    stoi['<PAD>'] = 0
    stoi['<BOS>'] = 1
    stoi['<EOS>'] = 2
    stoi['<UNK>'] = 3
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def save_tokenizer(stoi, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stoi, f, ensure_ascii=False)

def load_tokenizer(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def reconstruct_tokenizer(checkpoint_path, data_dir, max_lines):
    """Rebuild tokenizer exactly as training did."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    ckpt_vocab = ckpt['model_state_dict']['tok_emb.weight'].shape[0]

    data_files = [os.path.join(data_dir, 'c4_en.txt'),
                  os.path.join(data_dir, 'combined_for_spm.txt')]
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            if f.startswith('wiki_') and f.endswith('.txt'):
                data_files.append(os.path.join(data_dir, f))

    texts = load_text_data(data_files, max_lines)
    texts_for_vocab = texts[:10000]

    for vocab_size in range(1024, 50, -1):
        stoi, itos = build_char_tokenizer(texts_for_vocab, vocab_size)
        if len(stoi) == ckpt_vocab:
            return stoi, itos

    stoi, itos = build_char_tokenizer(texts_for_vocab, 1024)
    print(f"Warning: could not match vocab size exactly. Using {len(stoi)} (ckpt has {ckpt_vocab})")
    return stoi, itos

def load_model(checkpoint_path, data_dir, max_lines=5000):
    model = MateriaV4(vocab_size=256, dim=256, n_layers=3, n_heads=8, n_kv=4,
                      latent_dim=256, snn_dim=128, snn_threshold=0.005,
                      snn_tau=0.8, ssm_state=32, synapsis_slots=128,
                      hsaq_sparsity=0.3, jepa_weight=2.781042)
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)

    stoi, itos = reconstruct_tokenizer(checkpoint_path, data_dir, max_lines)
    ckpt_vocab = ckpt['model_state_dict']['tok_emb.weight'].shape[0]

    model = MateriaV4(vocab_size=len(stoi), dim=256, n_layers=3, n_heads=8, n_kv=4,
                      latent_dim=256, snn_dim=128, snn_threshold=0.005,
                      snn_tau=0.8, ssm_state=32, synapsis_slots=128,
                      hsaq_sparsity=0.3, jepa_weight=2.781042)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, stoi, itos

def generate(model, stoi, itos, prompt, max_new=100, temp=0.7, top_p=0.9):
    ids = [stoi.get(c, 3) for c in prompt]
    if not ids:
        ids = [stoi.get(' ', 3)]
    x = torch.tensor([ids], dtype=torch.long)
    device = next(model.parameters()).device
    x = x.to(device)
    with torch.no_grad():
        out = model.generate(x, max_new=max_new, temp=temp, top_p=top_p)
    gen_ids = out[0].tolist()
    result = ''.join(itos.get(i, '?') for i in gen_ids[len(ids):])
    return result

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-c', type=str, required=True)
    parser.add_argument('--prompt', '-p', type=str, default='Hola')
    parser.add_argument('--max-new', '-n', type=int, default=100)
    parser.add_argument('--temp', '-t', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--data-dir', type=str, default=None)
    parser.add_argument('--max-lines', type=int, default=5000)
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(MATERIA_HOME, 'data', 'multilingual', 'tokenizer')
    model, stoi, itos = load_model(args.checkpoint, data_dir, args.max_lines)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Vocab: {len(stoi)} tokens | Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")
    result = generate(model, stoi, itos, args.prompt, max_new=args.max_new, temp=args.temp, top_p=args.top_p)
    print(f"Output: {result}")
