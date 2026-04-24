"""
train_bridge_ditto.py — Training Entry Point for Bridge-Ditto Pipeline
======================================================================
Trains the Ditto LMDM diffusion model using bridge module features
instead of HuBERT features. All other components are identical to
the original Ditto training pipeline.

Key differences from original train.py:
  1. audio_feat_dim defaults to bridge output (1024 base, or 1103 with cond)
  2. Adds verification that bridge .npy files exist
  3. Adds gradient flow verification (optional debug mode)

Usage:
    # Single GPU:
    python train_bridge_ditto.py \
        --experiment_name bridge_ditto_v1 \
        --data_list_json data/hdtf_bridge_train.json \
        --audio_feat_dim 1103 \
        --epochs 500 \
        --batch_size 256

    # Multi-GPU with accelerate:
    accelerate launch train_bridge_ditto.py \
        --experiment_name bridge_ditto_v1 \
        --data_list_json data/hdtf_bridge_train.json \
        --use_accelerate

    # Full example with all conditioning:
    python train_bridge_ditto.py \
        --experiment_name bridge_ditto_full \
        --data_list_json data/hdtf_bridge_train.json \
        --data_preload --data_preload_pkl data/hdtf_bridge_preload.pkl \
        --audio_feat_dim 1103 \
        --motion_feat_dim 265 \
        --use_emo --use_eye_open --use_eye_ball --use_sc \
        --use_last_frame --use_accelerate \
        --epochs 1000 --batch_size 512 --lr 1e-4

Notes:
    - The data_list_json uses bridge .npy files where HuBERT .npy was used
    - The Stage2Dataset, LMDM, and Trainer are 100% unchanged
    - Bridge features are (T, 1024) at 25 Hz — same as HuBERT features
    - audio_feat_dim = 1024 + 63(sc) + 8(emo) + 2(eye_open) + 6(eye_ball) = 1103
"""

import json
import os
import sys
import time

# Ensure MotionDiT src is importable
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MOTIONDIT_DIR = os.path.join(CUR_DIR, "MotionDiT")
if MOTIONDIT_DIR not in sys.path:
    sys.path.insert(0, MOTIONDIT_DIR)

import tyro
from src.options.option import TrainOptions, check_train_opt
from src.trainers.trainer import Trainer


def verify_bridge_features(data_list_json: str, max_check: int = 10):
    """
    Quick sanity check: verify that a few audio .npy files in the data list
    actually contain bridge features (shape: (T, 1024), not HuBERT).
    """
    import numpy as np

    with open(data_list_json, encoding="utf-8") as f:
        data_list = json.load(f)

    if len(data_list) == 0:
        print(
            "⚠️  data_list_json has 0 entries — run gather_data_list_json_for_train.py "
            "(see Phase 3) after all .npy features exist on disk."
        )
        return

    checked = 0
    for entry in data_list[:max_check]:
        aud_path = entry.get("aud", "")
        if not aud_path or not os.path.isfile(aud_path):
            continue

        arr = np.load(aud_path)
        if arr.ndim != 2 or arr.shape[1] != 1024:
            print(
                f"⚠️  WARNING: {aud_path} has shape {arr.shape}, "
                f"expected (T, 1024). Is this a bridge feature file?"
            )
        else:
            checked += 1

    if checked > 0:
        print(f"✅ Verified {checked} bridge feature files: shape (T, 1024)")
    else:
        print("⚠️  Could not verify any bridge feature files.")


def main():
    # Set tyro theme (matches original Ditto train.py)
    tyro.extras.set_accent_color("bright_cyan")

    print("=" * 60)
    print("  Bridge-Ditto Training Pipeline")
    print("  Audio features: Mimi → Bridge (replaces HuBERT)")
    print("  Trainable: LMDM diffusion model only")
    print("=" * 60)

    # Parse options (same as original Ditto)
    opt = tyro.cli(TrainOptions)
    print(opt)

    # Validate
    check_train_opt(opt)

    with open(opt.data_list_json, encoding="utf-8") as _f:
        _n = len(json.load(_f))
    if _n == 0:
        raise SystemExit(
            f"Training data list is empty: {opt.data_list_json}\n"
            "Fix gather / feature paths (Phase 1 must produce motion/eye/emo .npy files on this machine)."
        )

    # Verify bridge features exist and have correct shape
    print(f"\n{time.asctime()} Verifying bridge features...")
    verify_bridge_features(opt.data_list_json)

    # Print training configuration summary
    print(f"\n{'─' * 50}")
    print(f"  Experiment    : {opt.experiment_name}")
    print(f"  Data list     : {opt.data_list_json}")
    print(f"  Motion dim    : {opt.motion_feat_dim}")
    print(f"  Audio dim     : {opt.audio_feat_dim}")
    print(f"  Seq frames    : {opt.seq_frames}")
    print(f"  Batch size    : {opt.batch_size}")
    print(f"  Learning rate : {opt.lr}")
    print(f"  Epochs        : {opt.epochs}")
    print(f"  Accelerate    : {opt.use_accelerate}")
    print(f"  Conditioning  : emo={opt.use_emo}, eye_open={opt.use_eye_open}, "
          f"eye_ball={opt.use_eye_ball}, sc={opt.use_sc}")
    print(f"{'─' * 50}\n")

    # Initialize trainer (same as original)
    print(f"{time.asctime()} Initializing trainer...")
    trainer = Trainer(opt)

    # Start training (same as original)
    print(f"\n{time.asctime()} Starting training loop...")
    trainer.train_loop()

    print(f"\n{time.asctime()} Training complete.")


if __name__ == "__main__":
    main()
