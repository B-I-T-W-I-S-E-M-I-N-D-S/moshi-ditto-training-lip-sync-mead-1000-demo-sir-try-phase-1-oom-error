import json
import random
import os
import numpy as np


def pipeline_log_video_every() -> int:
    """Log every N completed videos/items in prepare_data batch loops (default 25)."""
    return max(1, int(os.environ.get("DITTO_PIPELINE_LOG_VIDEO_EVERY", "25")))


def pipeline_log_frame_every() -> int:
    """tqdm miniters for per-frame bars — fewer console lines (default 25)."""
    return max(1, int(os.environ.get("DITTO_PIPELINE_LOG_FRAME_EVERY", "25")))


def log_batch_progress(
    done_1based: int,
    total: int,
    tag: str,
    *,
    unit: str = "videos",
) -> None:
    """Print one line every `pipeline_log_video_every()` steps and on the last item."""
    every = pipeline_log_video_every()
    if done_1based % every == 0 or done_1based == total:
        print(f"[{tag}] {done_1based}/{total} {unit} done", flush=True)
import torch
import time
import uuid
import pickle


def load_json(p):
    with open(p, "r") as f:
        return json.load(f)


def dump_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def dump_pkl(obj, pkl):
    with open(pkl, "wb") as fw:
        pickle.dump(obj, fw)
        