"""
audio_pipeline.py — Frozen Audio Feature Extraction Pipeline
=============================================================
Implements: Raw Audio → Mimi Encoder → Bridge Module → Features (T, 1024)

This module replaces HuBERT feature extraction in the Ditto training pipeline.
Both Mimi and Bridge are frozen (no gradients) — only used for feature extraction.

Output: .npy files with shape (T, 1024) at 25 Hz — drop-in replacement for
        the original HuBERT .npy files produced by extract_audio_feat_by_Hubert.py.

Usage:
    pipeline = FrozenAudioPipeline(
        bridge_ckpt="checkpoints/bridge_best.pt",
        bridge_config="bridge_module/config.yaml",
        mimi_model="kyutai/moshiko-pytorch-bf16",
        device="cuda",
    )
    features = pipeline.extract("audio.wav")          # (T, 1024) numpy
    pipeline.extract_and_save("audio.wav", "out.npy") # saves to disk
"""

import os
import sys
import logging
import hashlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ensure bridge_module is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_BRIDGE_DIR = os.path.join(_PROJECT_ROOT, "bridge_module")
if _BRIDGE_DIR not in sys.path:
    sys.path.insert(0, _BRIDGE_DIR)


class FrozenAudioPipeline:
    """
    Frozen Mimi → Bridge feature extraction pipeline.

    Both models are loaded in eval mode with requires_grad=False.
    No gradients flow through this pipeline — it is used only for
    preprocessing audio into features compatible with Ditto training.

    Output: (T, 1024) float32 numpy array at 25 Hz
            — identical format to HuBERT .npy files.
    """

    def __init__(
        self,
        bridge_ckpt: str = "checkpoints/bridge_best.pt",
        bridge_config: str = "bridge_module/config.yaml",
        mimi_model: str = "kyutai/moshiko-pytorch-bf16",
        device: str = "cuda",
        cache_dir: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        bridge_ckpt   : Path to trained bridge model checkpoint (.pt)
        bridge_config : Path to bridge config.yaml
        mimi_model    : HuggingFace repo or local path for Mimi weights
        device        : "cuda" or "cpu"
        cache_dir     : Optional directory to cache extracted features
        """
        import yaml
        from model import MimiHuBERTBridge
        from dataset import MimiExtractor

        self.device = device

        # ── Load bridge config ────────────────────────────────────────────
        bridge_config = os.path.abspath(bridge_config)
        with open(bridge_config) as f:
            self.cfg = yaml.safe_load(f)

        self.output_dim = self.cfg["model"]["output_dim"]       # 1024
        self.num_codebooks = self.cfg["model"]["num_codebooks"]  # 8
        self.upsample_factor = self.cfg["model"]["upsample_factor"]  # 2

        # ── Load Mimi encoder (FROZEN) ────────────────────────────────────
        logger.info(f"[FrozenAudioPipeline] Loading Mimi encoder: {mimi_model}")
        self.mimi = MimiExtractor(mimi_model, device=device)
        logger.info("[FrozenAudioPipeline] Mimi encoder loaded (frozen).")

        # ── Load Bridge model (FROZEN) ────────────────────────────────────
        bridge_ckpt = os.path.abspath(bridge_ckpt)
        logger.info(f"[FrozenAudioPipeline] Loading bridge model: {bridge_ckpt}")

        self.bridge = MimiHuBERTBridge(self.cfg).to(device)

        # Load checkpoint (supports both full trainer ckpts and bare state_dicts)
        try:
            ckpt = torch.load(bridge_ckpt, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(bridge_ckpt, map_location=device)

        sd = ckpt.get("bridge", ckpt)
        missing, unexpected = self.bridge.load_state_dict(sd, strict=False)
        if missing:
            logger.warning(f"Missing keys in bridge checkpoint: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys in bridge checkpoint: {unexpected}")

        # ── Freeze both models ────────────────────────────────────────────
        self.bridge.eval()
        self.bridge.requires_grad_(False)
        # Mimi is already frozen inside MimiExtractor (eval mode)

        logger.info(
            f"[FrozenAudioPipeline] Ready. "
            f"Output: (T, {self.output_dim}) @ 25 Hz. Device: {device}"
        )

        # ── Optional caching ─────────────────────────────────────────────
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, audio_path: str) -> Optional[Path]:
        """Generate a deterministic cache path for an audio file."""
        if self.cache_dir is None:
            return None
        h = hashlib.md5(os.path.abspath(audio_path).encode()).hexdigest()
        return self.cache_dir / f"{h}_bridge.npy"

    @torch.no_grad()
    def extract(self, audio_path: str) -> np.ndarray:
        """
        Extract bridge features from an audio file.

        Parameters
        ----------
        audio_path : Path to audio file (.wav, .flac, etc.)

        Returns
        -------
        numpy.ndarray : (T, 1024) float32 at 25 Hz
                        Drop-in replacement for HuBERT .npy features.
        """
        # Check cache first
        cp = self._cache_path(audio_path)
        if cp is not None and cp.exists():
            return np.load(cp)

        import torchaudio

        # ── 1. Load audio at native sample rate ──────────────────────────
        waveform, native_sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(0, keepdim=True)  # mono

        # ── 2. Mimi encode: audio → tokens (T_m, 8) @ 12.5 Hz ───────────
        tokens = self.mimi.extract(waveform, native_sr)  # (T_m, 8) int64 CPU

        # ── 3. Bridge: tokens → features (2*T_m, 1024) @ 25 Hz ──────────
        tokens_batch = tokens.unsqueeze(0).to(self.device)  # (1, T_m, 8)
        features, _ = self.bridge(tokens_batch)              # (1, 2*T_m, 1024)
        features = features.squeeze(0).float().cpu().numpy() # (2*T_m, 1024)

        # ── 4. Cache if enabled ──────────────────────────────────────────
        if cp is not None:
            np.save(cp, features)

        return features

    def extract_and_save(self, audio_path: str, output_npy: str) -> np.ndarray:
        """
        Extract features and save directly to .npy file.

        Parameters
        ----------
        audio_path : Path to audio file
        output_npy : Path to save .npy output

        Returns
        -------
        numpy.ndarray : (T, 1024) float32
        """
        features = self.extract(audio_path)
        os.makedirs(os.path.dirname(output_npy) or ".", exist_ok=True)
        np.save(output_npy, features)
        return features

    def get_feature_rate(self) -> float:
        """Returns the output feature rate in Hz (should be 25.0)."""
        mimi_rate = self.cfg["data"].get("mimi_rate", 12.5)
        return mimi_rate * self.upsample_factor  # 12.5 * 2 = 25.0


def create_pipeline(
    bridge_ckpt: str = None,
    bridge_config: str = None,
    mimi_model: str = "kyutai/moshiko-pytorch-bf16",
    device: str = "cuda",
    cache_dir: Optional[str] = None,
) -> FrozenAudioPipeline:
    """
    Factory function with sensible defaults for RunPod.

    If paths are not provided, they are resolved relative to the project root.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if bridge_ckpt is None:
        bridge_ckpt = os.path.join(project_root, "checkpoints", "bridge_best.pt")
    if bridge_config is None:
        bridge_config = os.path.join(project_root, "bridge_module", "config.yaml")

    return FrozenAudioPipeline(
        bridge_ckpt=bridge_ckpt,
        bridge_config=bridge_config,
        mimi_model=mimi_model,
        device=device,
        cache_dir=cache_dir,
    )
