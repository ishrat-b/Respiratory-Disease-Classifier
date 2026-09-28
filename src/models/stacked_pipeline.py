"""Reusable two-stage respiratory classification components.

This module keeps the final architecture in importable Python code. It does not
load project-specific checkpoints at import time; callers provide the models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import librosa
import torchaudio.transforms as T


CLASS_NAMES = {
    0: "healthy",
    1: "asthma",
    2: "copd",
    3: "covid",
}
SUBTYPE_NAMES = {
    0: "asthma",
    1: "copd",
    2: "covid",
}


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RespiratoryNet(nn.Module):
    """CNN used by Stage 2 for cough and vowel Mel spectrograms."""

    def __init__(self, num_classes: int = 3, n_mels: int = 128) -> None:
        super().__init__()
        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 32, dropout_p=0.2),
            ConvBlock(32, 64, dropout_p=0.3),
            ConvBlock(64, 128, dropout_p=0.3),
            ConvBlock(128, 256, dropout_p=0.4),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_mel_transform(
    sample_rate: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
    f_min: int = 20,
    f_max: int = 8000,
) -> nn.Sequential:
    """Create the Mel-spectrogram transform used by the CNN path."""
    return nn.Sequential(
        T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        ),
        T.AmplitudeToDB(stype="power", top_db=80),
    )


def load_audio_spectrogram(
    path: str,
    transform: nn.Module,
    sample_rate: int = 16_000,
    duration: float = 5.0,
) -> torch.Tensor:
    """Load one WAV file and return a normalized Mel spectrogram."""
    try:
        waveform, source_rate = torchaudio.load(path)
    except Exception:
        audio, source_rate = librosa.load(path, sr=None, mono=True)
        waveform = torch.tensor(audio).unsqueeze(0)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if source_rate != sample_rate:
        waveform = T.Resample(orig_freq=source_rate, new_freq=sample_rate)(waveform)

    target_length = int(sample_rate * duration)
    waveform = torch.nn.functional.pad(
        waveform[:, :target_length],
        (0, max(0, target_length - waveform.shape[1])),
    )
    spectrogram = transform(waveform)
    return (spectrogram - spectrogram.mean()) / (spectrogram.std() + 1e-8)


@dataclass
class Prediction:
    stage1_label: str
    diseased_probability: float
    subtype_label: str | None
    final_label: str


class TwoStageClassifier:
    """Combine a binary tabular model with the subtype CNN."""

    def __init__(
        self,
        stage1_model: Any,
        stage2_model: nn.Module,
        mel_transform: nn.Module,
        device: torch.device,
        threshold: float = 0.5,
    ) -> None:
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model.eval()
        self.mel_transform = mel_transform
        self.device = device
        self.threshold = threshold

    def predict(
        self,
        features: Any,
        cough_path: str,
        vowel_path: str,
    ) -> Prediction:
        """Run the staged prediction for one patient."""
        row = np.asarray(features).reshape(1, -1)
        probability = float(self.stage1_model.predict_proba(row)[0, 1])
        if probability < self.threshold:
            return Prediction("healthy", probability, None, "healthy")

        cough = load_audio_spectrogram(cough_path, self.mel_transform)
        vowel = load_audio_spectrogram(vowel_path, self.mel_transform)
        combined = torch.cat([cough, vowel], dim=2).unsqueeze(0).to(self.device)
        with torch.no_grad():
            subtype_index = int(self.stage2_model(combined).argmax(dim=1).item())

        subtype = SUBTYPE_NAMES[subtype_index]
        return Prediction("diseased", probability, subtype, subtype)
