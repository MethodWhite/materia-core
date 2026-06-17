"""
Audio encoder/decoder (EXPERIMENTAL)
Requiere datos de audio reales (pares WAV/MP3) para entrenar.
Actualmente sin datos - no usado en entrenamientos de produccion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden=128, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, output_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.proj(x)
        return x


class AudioDecoder(nn.Module):
    def __init__(self, input_dim=256, hidden=128, output_channels=80):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden * 8)
        self.convt1 = nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1)
        self.convt2 = nn.ConvTranspose1d(hidden, hidden, kernel_size=4, stride=2, padding=1)
        self.convt3 = nn.ConvTranspose1d(hidden, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.proj(x)
        B, T, D = x.shape
        x = x.view(B, T, -1, 8).permute(0, 2, 1, 3).reshape(B, -1, T * 8)
        x = F.gelu(self.convt1(x))
        x = F.gelu(self.convt2(x))
        x = self.convt3(x)
        x = x.transpose(1, 2)
        return x
