"""
syncnet.py — Wav2Lip SyncNet Model (Pretrained, Frozen)
========================================================
Self-contained SyncNet_color model from the Wav2Lip project.
Used as a frozen discriminator for lip-sync loss during Ditto training.

Architecture:
    face_encoder:  (B, 15, 96, 96) → (B, 512)   [5 lip frames × 3 channels]
    audio_encoder: (B, 1, 80, 16)  → (B, 512)   [mel spectrogram]

Both outputs are L2-normalized.

Reference: https://github.com/Rudrabha/Wav2Lip
License:   Research / academic / personal use only (LRS2 restriction)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conv2d building block (from Wav2Lip conv.py)
# ---------------------------------------------------------------------------
class Conv2d(nn.Module):
    """Conv + BatchNorm + ReLU, with optional residual."""

    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


# ---------------------------------------------------------------------------
# SyncNet color model (from Wav2Lip syncnet.py)
# ---------------------------------------------------------------------------
class SyncNet_color(nn.Module):
    """
    Wav2Lip SyncNet — pretrained audio-visual synchrony discriminator.

    Inputs:
        audio_sequences : (B, 1, 80, 16)   mel spectrogram
        face_sequences  : (B, 15, 96, 96)  5 consecutive lip crops, channels stacked

    Outputs:
        audio_embedding : (B, 512)  L2-normalized
        face_embedding  : (B, 512)  L2-normalized
    """

    def __init__(self):
        super(SyncNet_color, self).__init__()

        self.face_encoder = nn.Sequential(
            Conv2d(15, 32, kernel_size=(7, 7), stride=1, padding=3),
            Conv2d(32, 64, kernel_size=5, stride=(1, 2), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, audio_sequences, face_sequences):
        face_embedding = self.face_encoder(face_sequences)
        audio_embedding = self.audio_encoder(audio_sequences)

        audio_embedding = audio_embedding.view(audio_embedding.size(0), -1)
        face_embedding = face_embedding.view(face_embedding.size(0), -1)

        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        face_embedding = F.normalize(face_embedding, p=2, dim=1)

        return audio_embedding, face_embedding


# ---------------------------------------------------------------------------
# Helper: load pretrained SyncNet
# ---------------------------------------------------------------------------
def load_syncnet(ckpt_path: str, device: str = "cuda") -> SyncNet_color:
    """
    Load pretrained Wav2Lip SyncNet checkpoint.

    The checkpoint is the "Expert Discriminator" from the Wav2Lip repo
    (typically named lipsync_expert.pth).

    Returns a frozen SyncNet in eval mode — no gradient updates.
    """
    model = SyncNet_color()

    # Wav2Lip checkpoints store state_dict under 'state_dict' key
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    model.load_state_dict(sd)
    model = model.to(device)

    # Freeze: set eval mode and disable all gradients
    model.eval()
    model.requires_grad_(False)

    print(f"[SyncNet] Loaded pretrained checkpoint: {ckpt_path}")
    print(f"[SyncNet] Parameters: {sum(p.numel() for p in model.parameters()):,} (all frozen)")
    return model
