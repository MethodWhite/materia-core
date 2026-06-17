"""
MATERIA V3 - Audio Upscaling Pipeline
MP3 → FLAC/WAV reconstruction via spectrogram prediction
"""
import os, sys, json, time, subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from materia_v3_full import MateriaV3Full, AudioEncoder, AudioDecoder, count_params

class AudioDataset(Dataset):
    def __init__(self, wav_dir, mp3_dir=None, audio_dim=80, max_frames=500):
        self.pairs = []
        wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
        if mp3_dir is None:
            mp3_dir = wav_dir.replace('original', 'compressed')
        for wf in wav_files:
            wav_path = os.path.join(wav_dir, wf)
            mp3_path = os.path.join(mp3_dir, wf.replace('.wav', '.mp3'))
            if not os.path.exists(mp3_path):
                mp3_path = wav_path.replace('.wav', '_lo.mp3')
            if os.path.exists(mp3_path):
                self.pairs.append((wav_path, mp3_path))
        self.audio_dim = audio_dim
        self.max_frames = max_frames

    def __len__(self):
        return max(1, len(self.pairs))

    def __getitem__(self, idx):
        wav_path, mp3_path = self.pairs[idx % len(self.pairs)]
        try:
            y_hi, sr = librosa.load(wav_path, sr=16000)
            y_lo, _ = librosa.load(mp3_path, sr=16000)
        except:
            return torch.zeros(self.max_frames, self.audio_dim), \
                   torch.zeros(self.max_frames, self.audio_dim)

        spec_hi = librosa.feature.melspectrogram(
            y=y_hi, sr=sr, n_mels=self.audio_dim, n_fft=400, hop_length=160).T
        spec_lo = librosa.feature.melspectrogram(
            y=y_lo, sr=sr, n_mels=self.audio_dim, n_fft=400, hop_length=160).T

        spec_hi = np.log1p(spec_hi).astype(np.float32)
        spec_lo = np.log1p(spec_lo).astype(np.float32)

        min_len = min(len(spec_lo), len(spec_hi), self.max_frames)
        lo = torch.from_numpy(spec_lo[:min_len])
        hi = torch.from_numpy(spec_hi[:min_len])

        if lo.shape[0] < self.max_frames:
            pad = torch.zeros(self.max_frames - lo.shape[0], self.audio_dim)
            lo = torch.cat([lo, pad])
            hi = torch.cat([hi, pad])
        return lo, hi

def prepare_audio(wav_path, mp3_path=None):
    """Create MP3 from WAV at various bitrates for paired data"""
    if mp3_path is None:
        mp3_path = wav_path.replace('.wav', '_64k.mp3')
    if not os.path.exists(mp3_path):
        subprocess.run([
            'ffmpeg', '-i', wav_path, '-codec:a', 'libmp3lame',
            '-b:a', '64k', '-y', mp3_path
        ], capture_output=True)
    return mp3_path

def upscale_file(model, input_mp3, output_wav, audio_dim=80, device='cpu'):
    """Upscale a single MP3 file to WAV"""
    import soundfile as sf
    model.eval()
    y, sr = librosa.load(input_mp3, sr=16000)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=audio_dim, n_fft=400, hop_length=160)
    spec = np.log1p(spec).T.astype(np.float32)

    with torch.no_grad():
        inp = torch.from_numpy(spec).unsqueeze(0).to(device)
        pred = model.forward_upscale(inp)
        pred = pred[0].cpu().numpy()

    pred = np.expm1(pred)
    y_hi = librosa.feature.inverse.mel_to_audio(pred.T, sr=sr, n_fft=400, hop_length=160)
    sf.write(output_wav, y_hi, sr)
    return output_wav

def train_audio_upscale(model, dataset, epochs=20, lr=3e-4, batch_size=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    mse = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (lo, hi) in enumerate(loader):
            opt.zero_grad()
            pred = model.forward_upscale(lo)
            loss = mse(pred, hi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            total_loss += loss.item()
            if (i+1) % 10 == 0:
                print(f"  E{epoch+1}/{epochs} [{i+1}/{len(loader)}] loss={loss.item():.6f}", flush=True)
        avg = total_loss / len(loader)
        print(f"  → Epoch {epoch+1}: avg_loss={avg:.6f}", flush=True)
    return avg

def batch_prepare(wav_dir, output_dir):
    """Batch create compressed MP3 from WAV files"""
    os.makedirs(output_dir, exist_ok=True)
    for f in os.listdir(wav_dir):
        if f.endswith('.wav'):
            wav_path = os.path.join(wav_dir, f)
            mp3_path = os.path.join(output_dir, f.replace('.wav', '.mp3'))
            if not os.path.exists(mp3_path):
                print(f"  Compressing {f}...", end=' ', flush=True)
                prepare_audio(wav_path, mp3_path)
                print("OK", flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['prepare', 'train', 'upscale'], default='train')
    parser.add_argument('--input', default='/home/methodwhite/MATERIA/data/audio/original')
    parser.add_argument('--compressed', default='/home/methodwhite/MATERIA/data/audio/compressed')
    parser.add_argument('--output', default='/home/methodwhite/MATERIA/data/audio/upscaled')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--load', default=None)
    args = parser.parse_args()

    if args.mode == 'prepare':
        print("=== Preparing audio pairs ===")
        batch_prepare(args.input, args.compressed)

    elif args.mode == 'train':
        print("=== Training audio upscaler ===")
        dataset = AudioDataset(args.input, args.compressed)
        print(f"Dataset: {len(dataset)} pairs")

        model = MateriaV3Full(vocab_size=100, dim=256, n_layers=2)
        if args.load and os.path.exists(args.load):
            model.load_state_dict(torch.load(args.load, map_location='cpu', weights_only=True))
            print(f"Loaded pretrained: {args.load}")

        # Only train audio parts
        for p in model.audio_enc.parameters(): p.requires_grad = True
        for p in model.audio_dec.parameters(): p.requires_grad = True
        for p in model.layers.parameters(): p.requires_grad = False
        for p in model.snn.parameters(): p.requires_grad = False
        for p in model.ssm.parameters(): p.requires_grad = False

        print(f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        train_audio_upscale(model, dataset, epochs=args.epochs)

        torch.save(model.state_dict(),
                   '/home/methodwhite/MATERIA/models/materia-v3-audio-upscaler.pth')
        print("✓ Model saved!")

    elif args.mode == 'upscale':
        print(f"=== Upscaling {args.input} ===")
        model = MateriaV3Full(vocab_size=100, dim=256, n_layers=2)
        ckpt = '/home/methodwhite/MATERIA/models/materia-v3-audio-upscaler.pth'
        if os.path.exists(ckpt):
            model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=True))
        else:
            print("No trained model found, using untrained")
        os.makedirs(args.output, exist_ok=True)
        for f in os.listdir(args.input):
            if f.endswith('.mp3'):
                out = os.path.join(args.output, f.replace('.mp3', '_upscaled.wav'))
                upscale_file(model, os.path.join(args.input, f), out)
                print(f"  {f} -> {out}")
