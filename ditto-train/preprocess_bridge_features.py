"""
preprocess_bridge_features.py — Batch Bridge Feature Extraction
================================================================
Replaces: prepare_data/scripts/extract_audio_feat_by_Hubert.py

Takes the same data_info.json input format and produces .npy feature files
using the frozen Mimi → Bridge pipeline instead of HuBERT ONNX.

Output .npy shape: (T, 1024) @ 25 Hz — identical to HuBERT output.

Usage:
    # Single GPU:
    python preprocess_bridge_features.py \
        -i data_info.json \
        --bridge_ckpt checkpoints/bridge_best.pt \
        --bridge_config bridge_module/config.yaml

    # Multi-GPU (processes shards in parallel):
    python preprocess_bridge_features.py \
        -i data_info.json \
        --bridge_ckpt checkpoints/bridge_best.pt \
        --num_gpus 4 --gpu_id 0

    # Using prepare_data.sh style paths:
    python preprocess_bridge_features.py \
        -i data_info.json \
        --bridge_ckpt checkpoints/bridge_best.pt \
        --output_key bridge_aud_npy_list
"""

import os
import sys
import json
import argparse
import traceback
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project paths
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CUR_DIR)
sys.path.insert(0, CUR_DIR)
sys.path.insert(0, os.path.join(CUR_DIR, "prepare_data"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_output_npy_path(wav_path: str, output_dir: Optional[str] = None) -> str:
    """
    Generate output .npy path from wav path.
    If output_dir is given, place files there preserving basename.
    Otherwise, place alongside the wav file with _bridge_aud suffix.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        basename = os.path.splitext(os.path.basename(wav_path))[0]
        return os.path.join(output_dir, f"{basename}_bridge_aud.npy")
    else:
        return wav_path.replace(".wav", "_bridge_aud.npy").replace(
            ".flac", "_bridge_aud.npy"
        )


def process_data_list(
    wav_list: List[str],
    npy_list: List[str],
    bridge_ckpt: str,
    bridge_config: str,
    mimi_model: str = "kyutai/moshiko-pytorch-bf16",
    device: str = "cuda",
    skip_existing: bool = True,
    cache_dir: Optional[str] = None,
):
    """
    Process a list of wav files through the Mimi → Bridge pipeline.

    Parameters
    ----------
    wav_list      : List of audio file paths
    npy_list      : List of output .npy paths (1:1 with wav_list)
    bridge_ckpt   : Path to bridge checkpoint
    bridge_config : Path to bridge config.yaml
    mimi_model    : Mimi HF repo or local path
    device        : CUDA device string
    skip_existing : Skip files that already have .npy output
    cache_dir     : Optional cache directory for pipeline
    """
    from audio_pipeline import FrozenAudioPipeline

    # Initialize pipeline
    logger.info(f"Initializing Mimi → Bridge pipeline on {device}...")
    pipeline = FrozenAudioPipeline(
        bridge_ckpt=bridge_ckpt,
        bridge_config=bridge_config,
        mimi_model=mimi_model,
        device=device,
        cache_dir=cache_dir,
    )
    logger.info(f"Pipeline ready. Feature rate: {pipeline.get_feature_rate()} Hz")

    # Process files
    success = 0
    skipped = 0
    failed = 0

    for wav_path, npy_path in tqdm(
        zip(wav_list, npy_list),
        total=len(wav_list),
        desc="Extracting bridge features",
    ):
        try:
            # Skip if already exists
            if skip_existing and os.path.isfile(npy_path):
                skipped += 1
                continue

            # Extract and save
            features = pipeline.extract_and_save(wav_path, npy_path)
            success += 1

            if success % 100 == 0:
                logger.info(
                    f"Progress: {success} extracted, {skipped} skipped, "
                    f"{failed} failed. Last shape: {features.shape}"
                )

        except Exception:
            traceback.print_exc()
            failed += 1

    logger.info(
        f"\nDone! success={success}, skipped={skipped}, failed={failed}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract bridge audio features (replaces HuBERT extraction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard usage with data_info.json:
  python preprocess_bridge_features.py -i data_info.json

  # Multi-GPU sharding (run one per GPU):
  python preprocess_bridge_features.py -i data_info.json --num_gpus 4 --gpu_id 0
  python preprocess_bridge_features.py -i data_info.json --num_gpus 4 --gpu_id 1
  python preprocess_bridge_features.py -i data_info.json --num_gpus 4 --gpu_id 2
  python preprocess_bridge_features.py -i data_info.json --num_gpus 4 --gpu_id 3

  # Direct wav list mode:
  python preprocess_bridge_features.py --wav_dir /data/hdtf/wavs --output_dir /data/hdtf/bridge_feats
        """,
    )

    # Input options
    parser.add_argument(
        "-i", "--input_data_json", default="",
        help="data_info.json (same format as prepare_data.sh)",
    )
    parser.add_argument(
        "--wav_dir", default="",
        help="Alternative: directory containing wav files (instead of data_info.json)",
    )
    parser.add_argument(
        "--output_dir", default="",
        help="Output directory for .npy files (used with --wav_dir)",
    )

    # Model paths
    parser.add_argument(
        "--bridge_ckpt",
        default=os.path.join(PROJECT_ROOT, "checkpoints", "bridge_best.pt"),
        help="Path to bridge model checkpoint",
    )
    parser.add_argument(
        "--bridge_config",
        default=os.path.join(PROJECT_ROOT, "bridge_module", "config.yaml"),
        help="Path to bridge config.yaml",
    )
    parser.add_argument(
        "--mimi_model",
        default="kyutai/moshiko-pytorch-bf16",
        help="Mimi model HF repo or local path",
    )

    # Processing options
    parser.add_argument(
        "--device", default="cuda",
        help="Device (cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip files that already have .npy output",
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Force re-extraction even if .npy exists",
    )
    parser.add_argument(
        "--cache_dir", default="",
        help="Cache directory for the pipeline",
    )

    # Multi-GPU sharding
    parser.add_argument(
        "--num_gpus", type=int, default=1,
        help="Total number of GPUs for sharding",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=0,
        help="This GPU's shard index (0-indexed)",
    )

    # data_info.json key names
    parser.add_argument(
        "--wav_key", default="wav_list",
        help="Key in data_info.json for wav paths",
    )
    parser.add_argument(
        "--output_key", default="bridge_aud_npy_list",
        help="Key in data_info.json for output npy paths. "
             "If this key doesn't exist, paths are generated from hubert_aud_npy_list "
             "by replacing 'hubert' with 'bridge'.",
    )

    args = parser.parse_args()

    skip_existing = args.skip_existing and not args.no_skip

    # ── Build wav_list and npy_list ──────────────────────────────────────
    if args.input_data_json:
        data_info = load_json(args.input_data_json)
        wav_list = data_info[args.wav_key]

        # Generate output npy paths
        if args.output_key in data_info:
            npy_list = data_info[args.output_key]
        elif "hubert_aud_npy_list" in data_info:
            # Auto-generate by replacing hubert → bridge in paths
            npy_list = [
                p.replace("hubert_aud", "bridge_aud").replace("hubert_feat", "bridge_feat")
                for p in data_info["hubert_aud_npy_list"]
            ]
            logger.info(
                f"Generated bridge npy paths from hubert paths "
                f"(replacing 'hubert' → 'bridge')"
            )
        else:
            # Generate from wav paths
            npy_list = [get_output_npy_path(w, args.output_dir or None) for w in wav_list]

    elif args.wav_dir:
        import glob
        wav_list = sorted(
            glob.glob(os.path.join(args.wav_dir, "*.wav"))
            + glob.glob(os.path.join(args.wav_dir, "*.flac"))
        )
        output_dir = args.output_dir or os.path.join(args.wav_dir, "bridge_feats")
        npy_list = [get_output_npy_path(w, output_dir) for w in wav_list]
    else:
        parser.error("Provide either -i (data_info.json) or --wav_dir")

    assert len(wav_list) == len(npy_list), (
        f"wav_list ({len(wav_list)}) and npy_list ({len(npy_list)}) must match"
    )

    logger.info(f"Total files: {len(wav_list)}")

    # ── Multi-GPU sharding ────────────────────────────────────────────────
    if args.num_gpus > 1:
        total = len(wav_list)
        shard_size = (total + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_id * shard_size
        end = min(start + shard_size, total)
        wav_list = wav_list[start:end]
        npy_list = npy_list[start:end]
        # Use cuda:0 — CUDA_VISIBLE_DEVICES already restricts to the right GPU
        device = "cuda:0"
        logger.info(
            f"GPU {args.gpu_id}/{args.num_gpus}: "
            f"processing files [{start}:{end}] ({len(wav_list)} files) on {device}"
        )
    else:
        device = args.device

    # ── Run extraction ────────────────────────────────────────────────────
    process_data_list(
        wav_list=wav_list,
        npy_list=npy_list,
        bridge_ckpt=args.bridge_ckpt,
        bridge_config=args.bridge_config,
        mimi_model=args.mimi_model,
        device=device,
        skip_existing=skip_existing,
        cache_dir=args.cache_dir or None,
    )

    # ── Update data_info.json with bridge paths ───────────────────────────
    if args.input_data_json and args.gpu_id == 0:
        data_info = load_json(args.input_data_json)
        if args.output_key not in data_info:
            if "hubert_aud_npy_list" in data_info:
                data_info[args.output_key] = [
                    p.replace("hubert_aud", "bridge_aud").replace("hubert_feat", "bridge_feat")
                    for p in data_info["hubert_aud_npy_list"]
                ]
            else:
                data_info[args.output_key] = npy_list
            save_json(data_info, args.input_data_json)
            logger.info(
                f"Updated {args.input_data_json} with '{args.output_key}' key"
            )


if __name__ == "__main__":
    main()
