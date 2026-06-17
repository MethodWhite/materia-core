\"\"\"
MATERIA V3 - Standalone Audio Upscaler (EXPERIMENTAL)
MP3 → WAV/FLAC reconstruction via spectrogram super-resolution

NOTA: Este módulo es experimental. Cuando no hay pares reales (WAV+MP3),
genera datos sintéticos degradando espectrogramas, lo que NO equivale
a un upscaling real. Los resultados con datos sintéticos no son válidos
para reconstrucción audio real.
\"\"\"
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, time

class ConvBlock(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        self.conv = nn.Conv1d(c, c, k, padding=k//2)
        self.norm = nn.LayerNorm(c)
    def forward(self, x):
        return F.gelu(self.conv(x))

class AudioUpscaler(nn.Module):
    """Efficient standalone audio upscaler: 6 conv layers, 1.5M params"""
    def __init__(self, n_mels=80, hidden=128, scale_factor=3):
        super().__init__()
        self.scale = scale_factor
        
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv1d(n_mels, hidden, 7, padding=3),
            nn.GELU(),
            ConvBlock(hidden),
            nn.Conv1d(hidden, hidden, 3, stride=2, padding=1),  # 2x downsample
            nn.GELU(),
            ConvBlock(hidden),
            nn.Conv1d(hidden, hidden * 2, 3, stride=2, padding=1),  # 2x downsample
            nn.GELU(),
            ConvBlock(hidden * 2),
            ConvBlock(hidden * 2),
        )

        # Decoder (upsampler)
        self.dec = nn.Sequential(
            ConvBlock(hidden * 2),
            nn.ConvTranspose1d(hidden * 2, hidden, 4, stride=2, padding=1),  # 2x up
            nn.GELU(),
            ConvBlock(hidden),
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1),  # 2x up
            nn.GELU(),
            ConvBlock(hidden),
            nn.Conv1d(hidden, n_mels, 7, padding=3),
        )

    def forward(self, x):
        # x: [B, T, n_mels]
        x = x.transpose(1, 2)  # [B, n_mels, T] for Conv1d
        z = self.enc(x)
        y = self.dec(z)
        # Trim to match scaled length
        target_t = x.shape[-1] * self.scale
        if y.shape[-1] > target_t:
            y = y[..., :target_t]
        return y.transpose(1, 2)  # [B, T', n_mels]

class AudioUpscaleDataset(Dataset):
    def __init__(self, low_specs, high_specs, max_frames=300):
        self.low = [torch.from_numpy(s.astype(np.float32)) for s in low_specs]
        self.high = [torch.from_numpy(s.astype(np.float32)) for s in high_specs]

    def __len__(self):
        return max(1, len(self.low))

    def __getitem__(self, idx):
        i = idx % len(self.low)
        lo = self.low[i]
        hi = self.high[i]
        t = min(lo.shape[0], hi.shape[0] // 3, 300)
        return lo[:t], hi[:t*3]


def prepare_pairs(wav_dir, mp3_dir, sr=16000, n_mels=80, n_fft=400, hop=160):
    import librosa
    lo_list, hi_list = [], []
    wavs = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    for wf in wavs:
        wav_path = os.path.join(wav_dir, wf)
        mp3_path = os.path.join(mp3_dir, wf.replace('.wav', '.mp3'))
        if not os.path.exists(mp3_path):
            continue
        try:
            y_hi, _ = librosa.load(wav_path, sr=sr)
            y_lo, _ = librosa.load(mp3_path, sr=sr)
            spec_hi = librosa.feature.melspectrogram(y=y_hi, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop).T
            spec_lo = librosa.feature.melspectrogram(y=y_lo, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop).T
            spec_hi = np.log1p(spec_hi)
            spec_lo = np.log1p(spec_lo)
            lo_list.append(spec_lo)
            hi_list.append(spec_hi)
        except:
            continue
    return lo_list, hi_list

def train(model, loader, epochs=50, lr=3e-4):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(loader))
    mse = nn.MSELoss()
    model.train()
    history = []

    for epoch in range(epochs):
        total_loss = 0.0
        for lo, hi in loader:
            opt.zero_grad()
            pred = model(lo)
            # Pad hi to match pred length
            if hi.shape[1] < pred.shape[1]:
                pad = torch.zeros(hi.shape[0], pred.shape[1] - hi.shape[1], hi.shape[2])
                hi = torch.cat([hi, pad], dim=1)
            elif hi.shape[1] > pred.shape[1]:
                hi = hi[:, :pred.shape[1]]
            loss = mse(pred, hi)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            total_loss += loss.item()

        avg = total_loss / max(1, len(loader))
        history.append(avg)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  E{epoch+1}/{epochs}: loss={avg:.6f}", flush=True)
    return history

def upscale_file(model, mp3_path, output_path, sr=16000, n_mels=80, device='cpu'):
    import librosa, soundfile as sf
    model.eval()
    y, _ = librosa.load(mp3_path, sr=sr)
    spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=400, hop_length=160)
    spec = np.log1p(spec).T.astype(np.float32)
    inp = torch.from_numpy(spec).unsqueeze(0)
    with torch.no_grad():
        pred = model(inp).squeeze(0).numpy()
    pred = np.expm1(pred)
    audio = librosa.feature.inverse.mel_to_audio(pred.T, sr=sr, n_fft=400, hop_length=160)
    sf.write(output_path, audio, sr)
    return output_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'upscale'], default='train')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    BASE = '/home/methodwhite/MATERIA'

    if args.mode == 'train':
        wav_dir = os.path.join(BASE, 'data/kino/audio')
        mp3_dir = os.path.join(BASE, 'data/audio/compressed')
        print("Preparing audio pairs...", flush=True)
        lo_list, hi_list = prepare_pairs(wav_dir, mp3_dir)
        print(f"  {len(lo_list)} pairs loaded", flush=True)

        if len(lo_list) < 2:
            # Generate synthetic data from any WAV
            print("  Too few real pairs, generating synthetic...", flush=True)
            import librosa
            wavs = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
            for wf in wavs[:10]:
                wav_path = os.path.join(wav_dir, wf)
                y, sr = librosa.load(wav_path, sr=16000)
                spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, n_fft=400, hop_length=160).T
                spec = np.log1p(spec).astype(np.float32)
                # Create degraded version
                degraded = spec.copy()
                degraded[::2] *= 0.3  # Remove some frequencies
                degraded += np.random.randn(*degraded.shape) * 0.1  # Add noise
                lo_list.append(degraded)
                hi_list.append(spec)
            print(f"  {len(lo_list)} total pairs", flush=True)

        dataset = AudioUpscaleDataset(lo_list, hi_list)
        loader = DataLoader(dataset, batch_size=2, shuffle=True)

        model = AudioUpscaler()
        if args.load and os.path.exists(args.load):
            model.load_state_dict(torch.load(args.load, map_location='cpu', weights_only=True))
            print(f"Loaded: {args.load}", flush=True)

        total = sum(p.numel() for p in model.parameters())
        print(f"Model: {total:,} params", flush=True)

        print("Training...", flush=True)
        train(model, loader, epochs=args.epochs)

        save_path = os.path.join(BASE, 'models/audio_upscaler.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Saved: {save_path}", flush=True)

    elif args.mode == 'upscale':
        model = AudioUpscaler()
        load_path = args.load or os.path.join(BASE, 'models/audio_upscaler.pth')
        if os.path.exists(load_path):
            model.load_state_dict(torch.load(load_path, map_location='cpu', weights_only=True))
            print(f"Loaded: {load_path}")
        else:
            print(f"No model found at {load_path}, using untrained")

        inp = args.input
        out = args.output or inp.replace('.mp3', '_upscaled.wav')
        upscale_file(model, inp, out)
        print(f"Upscaled: {inp} → {out}")
