"""
MATERIA V3 - Master Pipeline
Multilingual text + Audio upscaling + AGI training
"""
import os, sys, time, json, gzip, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import requests, subprocess, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from materia_v3_full import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = '/home/methodwhite/MATERIA'

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# =========== DATASET ACQUISITION ===========

def download_wikipedia(langs=None, max_docs=10000):
    """Download Wikipedia via API for all target languages"""
    if langs is None:
        langs = ['en', 'es', 'fr', 'de', 'pt', 'it', 'ja', 'ru', 'ar', 'zh', 'hi', 'ko']
    out_dir = os.path.join(BASE, 'data/multilingual/datasets')
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'MateriaV3/1.0'})

    results = {}
    for lang in langs:
        out = os.path.join(out_dir, f'wiki_{lang}.txt')
        if os.path.exists(out) and os.path.getsize(out) > 500000:
            results[lang] = os.path.getsize(out)
            log(f"  ✓ {lang} already cached ({results[lang]//1024}KB)")
            continue

        log(f"  → {lang}: downloading...")
        api = f'https://{lang}.wikipedia.org/w/api.php'
        total = 0; done = 0; seen = set()
        with open(out, 'w', encoding='utf-8') as f:
            while done < max_docs:
                try:
                    r = session.get(api, params={
                        'action': 'query', 'format': 'json', 'list': 'random',
                        'rnlimit': 50, 'rnnamespace': 0
                    }, timeout=15).json()
                    titles = [p['title'] for p in r['query']['random'] if p['title'] not in seen]
                    if not titles: break
                    seen.update(titles)
                    for batch_start in range(0, len(titles), 20):
                        batch = titles[batch_start:batch_start+20]
                        r2 = session.get(api, params={
                            'action': 'query', 'format': 'json',
                            'titles': '|'.join(batch), 'prop': 'extracts',
                            'explaintext': True, 'exlimit': 20
                        }, timeout=15).json()
                        for pid, page in r2.get('query', {}).get('pages', {}).items():
                            text = page.get('extract', '')
                            if len(text) > 500:
                                f.write(text.strip() + '\n')
                                total += len(text)
                                done += 1
                                if done >= max_docs: break
                except Exception as e:
                    log(f"    ! {lang}: {e}")
                    break
            results[lang] = total
            log(f"    OK: {done} docs, {total:,} chars")
    return results

def download_oscar_sample(langs=None):
    """Download OSCAR corpus samples (multilingual web text)"""
    if langs is None:
        langs = ['es', 'fr', 'de', 'pt', 'it', 'ru', 'ar']
    out_dir = os.path.join(BASE, 'data/multilingual/datasets')
    os.makedirs(out_dir, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': 'MateriaV3/1.0'})

    for lang in langs:
        out = os.path.join(out_dir, f'oscar_{lang}.txt')
        if os.path.exists(out) and os.path.getsize(out) > 100000:
            log(f"  ✓ oscar_{lang} exists ({os.path.getsize(out)//1024}KB)")
            continue
        log(f"  ✗ oscar_{lang}: need manual download from huggingface")
    return True

def download_audio_datasets():
    """Download audio datasets for upscaling training"""
    adir = os.path.join(BASE, 'data/audio/datasets')
    os.makedirs(adir, exist_ok=True)
    
    # LibriSpeech sample (clean speech, ~300MB)
    ls_dir = os.path.join(adir, 'LibriSpeech')
    if not os.path.exists(ls_dir):
        url = 'https://www.openslr.org/resources/12/train-clean-100.tar.gz'
        log(f"LibriSpeech: downloading {url}")
        # Use wget for resume support
        subprocess.run(['wget', '-c', url, '-O', f'{adir}/train-clean-100.tar.gz'], capture_output=True)
        subprocess.run(['tar', '-xzf', f'{adir}/train-clean-100.tar.gz', '-C', adir], capture_output=True)
        log("LibriSpeech: done")
    else:
        log("  ✓ LibriSpeech exists")

    # High-quality WAV samples for testing
    hq_dir = os.path.join(adir, 'hq_samples')
    os.makedirs(hq_dir, exist_ok=True)
    sample_urls = [
        ('https://file-examples.com/storage/fe5f9a8f7c7e0b0e6e8a9e9/2017/11/file_example_WAV_1MG.wav',
         f'{hq_dir}/sample1.wav'),
    ]
    for url, path in sample_urls:
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
            log(f"  → sample: {os.path.getsize(path)//1024}KB")

    return True

# =========== DATA PREPARATION ===========

def prepare_multilingual_dataset(tokenizer, max_per_lang=50000):
    """Load all multilingual text into pre-tokenized chunks"""
    data_dir = os.path.join(BASE, 'data/multilingual/datasets')
    tok_dir = os.path.join(BASE, 'data/multilingual/tokenizer')

    all_texts = []

    # Load Wikipedia
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith('wiki_') and fname.endswith('.txt'):
            lang = fname.replace('wiki_', '').replace('.txt', '')
            fpath = os.path.join(data_dir, fname)
            with open(fpath, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if len(l.strip()) > 100]
            random.shuffle(lines)
            all_texts.extend(lines[:max_per_lang])
            log(f"  {fname}: {min(len(lines), max_per_lang):,} docs")

    # Load C4 English
    c4_path = os.path.join(tok_dir, 'c4_en.txt')
    if os.path.exists(c4_path):
        with open(c4_path, 'r', encoding='utf-8') as f:
            c4_lines = [l.strip() for l in f if len(l.strip()) > 100]
        random.shuffle(c4_lines)
        all_texts.extend(c4_lines[:max_per_lang * 3])
        log(f"  c4_en: {min(len(c4_lines), max_per_lang*3):,} docs")

    log(f"Total texts: {len(all_texts):,}")

    # Tokenize all
    log("Tokenizing...")
    t0 = time.time()
    all_chunks = []
    for text in all_texts:
        ids = tokenizer.encode(text)
        if len(ids) > 65:
            for i in range(0, len(ids) - 65, 32):
                all_chunks.append(ids[i:i+65])
    t1 = time.time()
    log(f"  {len(all_chunks):,} chunks in {t1-t0:.1f}s")
    return all_chunks

def prepare_audio_dataset(max_pairs=100, audio_dim=80, max_frames=300):
    """Load audio pairs (low-quality → high-quality)"""
    import librosa
    pairs = []

    # From compressed kino audio
    original_dir = os.path.join(BASE, 'data/kino/audio')
    compressed_dir = os.path.join(BASE, 'data/audio/compressed')

    if os.path.exists(original_dir):
        wavs = [f for f in os.listdir(original_dir) if f.endswith('.wav')]
        for wf in wavs[:max_pairs]:
            wav_path = os.path.join(original_dir, wf)
            mp3_path = os.path.join(compressed_dir, wf.replace('.wav', '.mp3'))
            if not os.path.exists(mp3_path): continue
            try:
                y_hi, sr = librosa.load(wav_path, sr=16000)
                y_lo, _ = librosa.load(mp3_path, sr=16000)
                spec_hi = librosa.feature.melspectrogram(y=y_hi, sr=sr, n_mels=audio_dim,
                                                         n_fft=400, hop_length=160).T
                spec_lo = librosa.feature.melspectrogram(y=y_lo, sr=sr, n_mels=audio_dim,
                                                         n_fft=400, hop_length=160).T
                spec_hi = np.log1p(spec_hi).astype(np.float32)
                spec_lo = np.log1p(spec_lo).astype(np.float32)
                min_len = min(len(spec_lo), len(spec_hi), max_frames)
                pairs.append((spec_lo[:min_len], spec_hi[:min_len]))
            except:
                continue

    log(f"Audio pairs: {len(pairs)}")
    return pairs

# =========== TRAINING ===========

def train_multilingual(model, tokenizer, chunks, epochs=5, batch_size=4, lr=3e-4):
    """Train on multilingual text data"""
    log(f"Training multilingual: {len(chunks):,} chunks, {epochs} epochs")

    def collate(batch):
        x, y = [], []
        for item in batch:
            x.append(item[:-1])
            y.append(item[1:])
        max_len = max(len(i) for i in x)
        x_pad = torch.full((len(x), max_len), tokenizer.pad_id, dtype=torch.long)
        y_pad = torch.full((len(y), max_len), -100, dtype=torch.long)
        for i, (xi, yi) in enumerate(zip(x, y)):
            x_pad[i, :len(xi)] = torch.tensor(xi, dtype=torch.long)
            y_pad[i, :len(yi)] = torch.tensor(yi, dtype=torch.long)
        return x_pad, y_pad

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    total_steps = epochs * (len(chunks) // batch_size)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    for epoch in range(epochs):
        random.shuffle(chunks)
        total_loss = 0.0
        steps = len(chunks) // batch_size

        for i in range(steps):
            batch = chunks[i*batch_size:(i+1)*batch_size]
            x, y = collate(batch)
            opt.zero_grad()
            logits, rate = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            total_loss += loss.item()
            if (i+1) % 200 == 0:
                sp = (logits.detach() == 0).float().mean().item()
                log(f"  E{epoch+1}/{epochs} [{i+1}/{steps}] loss={loss.item():.4f} "
                    f"spike={rate:.3f} sparse={sp:.3f} lr={sch.get_last_lr()[0]:.2e}")
        avg = total_loss / steps
        log(f"→ Epoch {epoch+1}/{epochs}: avg_loss={avg:.4f}")
    return avg

def train_audio_upscale(model, pairs, epochs=30, batch_size=2, lr=3e-4, audio_dim=80):
    """Train audio upscaling"""
    log(f"Training audio upscale: {len(pairs)} pairs, {epochs} epochs")

    class PairDataset(Dataset):
        def __init__(self, pairs, max_frames=300):
            self.lo = [torch.from_numpy(p[0]) for p in pairs]
            self.hi = [torch.from_numpy(p[1]) for p in pairs]

        def __len__(self):
            return max(1, len(self.lo))

        def __getitem__(self, idx):
            lo_pad = torch.zeros(300, audio_dim)
            hi_pad = torch.zeros(300, audio_dim)
            lo = self.lo[idx % len(self.lo)]
            hi = self.hi[idx % len(self.lo)]
            n = min(lo.shape[0], 300)
            lo_pad[:n] = lo[:n]
            hi_pad[:n] = hi[:n]
            return lo_pad, hi_pad

    dataset = PairDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    mse = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for lo, hi in loader:
            opt.zero_grad()
            pred = model.forward_upscale(lo)
            loss = mse(pred, hi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            total_loss += loss.item()
        avg = total_loss / max(1, len(loader))
        if (epoch+1) % 5 == 0 or epoch == 0:
            log(f"  Audio E{epoch+1}/{epochs}: loss={avg:.6f}")
    return avg

# =========== EVALUATION ===========

def evaluate_generation(model, tokenizer, prompts, temp=0.7):
    """Evaluate text generation quality"""
    model.eval()
    log("=== Generation Eval ===")
    results = []
    for prompt in prompts:
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        gen = model.generate(ids, max_new=30, temp=temp, top_p=0.9)
        text = tokenizer.decode(gen[0].tolist())
        log(f"  [{prompt}] → {text[:120]}")
        results.append((prompt, text))
    return results

def eval_audio_upscale(model, pairs, n=3, audio_dim=80):
    """Evaluate audio upscaling quality"""
    import librosa, soundfile as sf
    model.eval()
    log("=== Audio Upscale Eval ===")
    eval_dir = os.path.join(BASE, 'data/audio/eval')
    os.makedirs(eval_dir, exist_ok=True)

    for i in range(min(n, len(pairs))):
        lo_spec, hi_spec = pairs[i]
        with torch.no_grad():
            inp = torch.from_numpy(lo_spec).float().unsqueeze(0)
            pred = model.forward_upscale(inp).squeeze(0).numpy()

        # Convert back to audio
        lo_audio = librosa.feature.inverse.mel_to_audio(
            np.expm1(lo_spec.T), sr=16000, n_fft=400, hop_length=160)
        hi_audio = librosa.feature.inverse.mel_to_audio(
            np.expm1(hi_spec.T), sr=16000, n_fft=400, hop_length=160)
        pred_audio = librosa.feature.inverse.mel_to_audio(
            np.expm1(pred.T), sr=16000, n_fft=400, hop_length=160)

        sf.write(f'{eval_dir}/eval_{i}_low.wav', lo_audio, 16000)
        sf.write(f'{eval_dir}/eval_{i}_original.wav', hi_audio, 16000)
        sf.write(f'{eval_dir}/eval_{i}_upscaled.wav', pred_audio, 16000)
        log(f"  → eval_{i}: saved to {eval_dir}")

# =========== MAIN ===========

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['data', 'train', 'all'], default='all')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--no-audio', action='store_true')
    parser.add_argument('--load', type=str, default=None)
    args = parser.parse_args()

    log(f"Device: {DEVICE}")
    log(f"Base: {BASE}")

    tok = Tokenizer()
    log(f"Tokenizer: {tok.vocab_size} vocab")

    if args.mode in ('data', 'all'):
        log("\n=== PHASE 1: Dataset Acquisition ===")
        download_wikipedia()
        download_audio_datasets()

    if args.mode in ('train', 'all'):
        log("\n=== PHASE 2: Data Preparation ===")
        chunks = prepare_multilingual_dataset(tok)
        audio_pairs = prepare_audio_dataset()

        log("\n=== PHASE 3: Model Initialization ===")
        model = MateriaV3Full(vocab_size=tok.vocab_size, dim=256, n_layers=4)
        if args.load and os.path.exists(args.load):
            model.load_state_dict(torch.load(args.load, map_location=DEVICE, weights_only=True))
            log(f"Loaded: {args.load}")
        if DEVICE.type == 'cuda':
            model.to(DEVICE)
        total = sum(p.numel() for p in model.parameters())
        log(f"Total params: {total:,}")

        log("\n=== PHASE 4: Multilingual Training ===")
        train_multilingual(model, tok, chunks, epochs=args.epochs)

        if not args.no_audio and len(audio_pairs) >= 2:
            log("\n=== PHASE 5: Audio Upscale Training ===")
            # Freeze text layers, train only audio
            for p in model.audio_enc.parameters(): p.requires_grad = True
            for p in model.audio_dec.parameters(): p.requires_grad = True
            for p in model.layers.parameters(): p.requires_grad = False
            for p in model.snn.parameters(): p.requires_grad = False
            train_audio_upscale(model, audio_pairs, epochs=30)

        log("\n=== PHASE 6: Saving ===")
        model_path = os.path.join(BASE, 'models/materia-v3-master.pth')
        torch.save(model.state_dict(), model_path)
        log(f"Model saved: {model_path}")

        log("\n=== PHASE 7: Evaluation ===")
        prompts = [
            "Hello, this is a test of", "Hola, esto es una prueba de",
            "Bonjour, ceci est un test", "你好，这是一个测试",
            "Guten Tag, dies ist ein", "Ciao, questo è un",
        ]
        evaluate_generation(model, tok, prompts)

        if len(audio_pairs) >= 2:
            eval_audio_upscale(model, audio_pairs)

    log("\n✓ Master pipeline complete!")
