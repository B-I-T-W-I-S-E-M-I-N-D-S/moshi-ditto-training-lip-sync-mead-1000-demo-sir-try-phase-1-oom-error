"""
preprocess_lipsync_features.py — Precompute Lip-Sync Features for Training
============================================================================
Precomputes per-video features needed for lip-sync loss during Ditto training:
    1. f_s        — source appearance features  (1, 32, 16, 64, 64)
    2. x_s        — source keypoints            (1, 21, 3)
    3. kp_canon   — canonical keypoints         (1, 63)
    4. syncnet_A  — SyncNet audio embeddings    (N_windows, 512)
    5. sim_gt     — GT similarity scores        (N_windows,)

These are saved alongside existing features and referenced in data_list_json.

Usage:
    python preprocess_lipsync_features.py \\
        -i /workspace/HDTF/data_info.json \\
        --syncnet_ckpt checkpoints/lipsync_expert.pth \\
        --ditto_pytorch_path checkpoints/ditto_pytorch \\
        --device cuda

Multi-GPU:
    CUDA_VISIBLE_DEVICES=0 python preprocess_lipsync_features.py ... --num_gpus 4 --gpu_id 0 &
    CUDA_VISIBLE_DEVICES=1 python preprocess_lipsync_features.py ... --num_gpus 4 --gpu_id 1 &
    ...
"""

import os
import sys
import json
import types
import argparse
import traceback

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project modules are importable
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CUR_DIR)
MOTIONDIT_DIR = os.path.join(CUR_DIR, "MotionDiT")
INFERENCE_DIR = os.path.join(PROJECT_ROOT, "ditto-inference")
MODULES_DIR = os.path.join(INFERENCE_DIR, "core", "models", "modules")


def _setup_liveportrait_imports():
    """
    Set up imports for LivePortrait modules (AppearanceFeatureExtractor, etc.).

    These modules use relative imports (from .util import ...), so we must
    create a synthetic package with the correct __path__ so Python can
    resolve them.
    """
    if 'lp_modules' in sys.modules:
        return  # already set up

    # Create a synthetic package called 'lp_modules'
    pkg = types.ModuleType('lp_modules')
    pkg.__path__ = [MODULES_DIR]
    pkg.__package__ = 'lp_modules'
    pkg.__file__ = os.path.join(MODULES_DIR, '__init__.py')
    sys.modules['lp_modules'] = pkg

    # Also register it under the name the relative imports expect
    # When we do `from lp_modules.appearance_feature_extractor import ...`
    # the file has `from .util import ...` which resolves to `lp_modules.util`


# ── Mel spectrogram (matching Wav2Lip) ────────────────────────────────────

def compute_mel_spectrogram(wav_path, fps=25, n_mels=80, sr=16000,
                            n_fft=800, hop_length=200, win_length=800):
    """
    Compute mel spectrogram from a wav file, matching Wav2Lip conventions.

    Returns:
        mel: (n_mels, T_mel) float32 numpy array
    """
    import torchaudio

    waveform, native_sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    # Resample to 16kHz if needed
    if native_sr != sr:
        resampler = torchaudio.transforms.Resample(native_sr, sr)
        waveform = resampler(waveform)

    # Compute mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=1.0,
    )
    mel = mel_transform(waveform).squeeze(0).numpy()  # (n_mels, T_mel)

    # Log scale
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    return mel


def mel_to_syncnet_windows(mel, fps=25, sr=16000, hop_length=200, num_frames=5):
    """
    Slice mel spectrogram into SyncNet-compatible windows.

    Each window corresponds to 5 video frames (~0.2s).
    SyncNet audio input: (1, 80, 16) per window.

    Returns:
        windows: list of (1, 80, mel_steps_per_window) arrays
        mel_steps_per_window is approximately 16
    """
    # How many mel frames per video frame
    mel_per_video_frame = sr / (fps * hop_length)  # 16000 / (25 * 200) = 3.2

    # Mel frames per window (5 video frames)
    mel_per_window = int(mel_per_video_frame * num_frames)  # ~16

    # Total video frames
    total_mel_frames = mel.shape[1]
    total_video_frames = int(total_mel_frames / mel_per_video_frame)

    windows = []
    for i in range(total_video_frames - num_frames + 1):
        mel_start = int(i * mel_per_video_frame)
        mel_end = mel_start + mel_per_window
        if mel_end > total_mel_frames:
            mel_end = total_mel_frames
        win = mel[:, mel_start:mel_end]  # (80, ~16)

        # Pad/truncate to exactly mel_per_window
        if win.shape[1] < mel_per_window:
            win = np.pad(win, ((0, 0), (0, mel_per_window - win.shape[1])))
        elif win.shape[1] > mel_per_window:
            win = win[:, :mel_per_window]

        windows.append(win[np.newaxis, :, :])  # (1, 80, 16)

    return windows


# ── Lip region extraction ─────────────────────────────────────────────────

def extract_lip_crop(frame_256, lip_h=48, lip_w=96):
    """
    Extract lip region from a 256×256 face crop.
    SyncNet expects the LOWER HALF of a 96×96 face = 48×96.

    Args:
        frame_256: (H, W, 3) uint8 RGB, 256×256

    Returns:
        lip_crop: (lip_h, lip_w, 3) uint8  [default 48×96]
    """
    H, W = frame_256.shape[:2]
    # Crop lower-center mouth region
    y_start = int(H * 0.55)
    y_end = int(H * 0.95)
    x_start = int(W * 0.1)
    x_end = int(W * 0.9)

    lip = frame_256[y_start:y_end, x_start:x_end]
    lip = cv2.resize(lip, (lip_w, lip_h), interpolation=cv2.INTER_AREA)
    return lip


# ── Keypoint transform (numpy, for precomputation) ─────────────────────────

def transform_keypoint_np(kp_info):
    """Same as motion_stitch.transform_keypoint but standalone numpy."""
    from scipy.special import softmax

    kp = kp_info['kp'].reshape(-1, 21, 3)     # (1, 21, 3)
    pitch = kp_info['pitch']
    yaw = kp_info['yaw']
    roll = kp_info['roll']
    t = kp_info['t']
    exp = kp_info['exp'].reshape(-1, 21, 3)
    scale = kp_info['scale']

    def bin66_to_deg(pred):
        if pred.ndim > 1 and pred.shape[1] == 66:
            idx = np.arange(66).astype(np.float32)
            pred = softmax(pred, axis=1)
            return np.sum(pred * idx, axis=1) * 3 - 97.5
        return pred

    pitch_deg = bin66_to_deg(pitch)
    yaw_deg = bin66_to_deg(yaw)
    roll_deg = bin66_to_deg(roll)

    def get_rot(p, y, r):
        p, y, r = p / 180 * np.pi, y / 180 * np.pi, r / 180 * np.pi
        bs = 1
        ones = np.ones((bs, 1), dtype=np.float32)
        zeros = np.zeros((bs, 1), dtype=np.float32)
        x_, y_, z_ = p.reshape(bs, 1), y.reshape(bs, 1), r.reshape(bs, 1)

        rx = np.concatenate([ones, zeros, zeros, zeros, np.cos(x_), -np.sin(x_),
                              zeros, np.sin(x_), np.cos(x_)], axis=1).reshape(bs, 3, 3)
        ry = np.concatenate([np.cos(y_), zeros, np.sin(y_), zeros, ones, zeros,
                              -np.sin(y_), zeros, np.cos(y_)], axis=1).reshape(bs, 3, 3)
        rz = np.concatenate([np.cos(z_), -np.sin(z_), zeros, np.sin(z_), np.cos(z_),
                              zeros, zeros, zeros, ones], axis=1).reshape(bs, 3, 3)
        return np.matmul(np.matmul(rz, ry), rx).transpose(0, 2, 1)

    R = get_rot(pitch_deg, yaw_deg, roll_deg)
    kp_t = np.matmul(kp, R) + exp
    kp_t *= scale[..., None]
    kp_t[:, :, 0:2] += t[:, None, 0:2]

    return kp_t  # (1, 21, 3)


# ── Main processing ───────────────────────────────────────────────────────

def process_one_video(
    video_path, wav_path, mtn_npy_path,
    output_dir,
    appearance_extractor, motion_extractor,
    syncnet, device,
    fps=25, lip_size=96, num_frames=5,
):
    """Process a single video — extract all lip-sync features."""

    os.makedirs(output_dir, exist_ok=True)

    # Output paths
    f_s_path = os.path.join(output_dir, "lipsync_f_s.npy")
    x_s_path = os.path.join(output_dir, "lipsync_x_s.npy")
    kp_c_path = os.path.join(output_dir, "lipsync_kp_canonical.npy")
    A_path = os.path.join(output_dir, "lipsync_syncnet_A.npy")
    sim_gt_path = os.path.join(output_dir, "lipsync_sim_gt.npy")

    # Skip if all exist
    if all(os.path.isfile(p) for p in [f_s_path, x_s_path, kp_c_path, A_path, sim_gt_path]):
        return {
            'lipsync_f_s': f_s_path,
            'lipsync_x_s': x_s_path,
            'lipsync_kp_canonical': kp_c_path,
            'lipsync_A': A_path,
            'lipsync_sim_gt': sim_gt_path,
        }

    # ── Read source frame ─────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        raise RuntimeError(f"Cannot read first frame: {video_path}")

    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = first_frame_rgb.shape[:2]

    # Resize to 256×256 for LivePortrait
    frame_256 = cv2.resize(first_frame_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    frame_bchw = (frame_256.astype(np.float32) / 255.0)[np.newaxis].transpose(0, 3, 1, 2)
    # (1, 3, 256, 256)

    # ── Extract appearance features (f_s) ─────────────────────────────
    with torch.no_grad():
        f_s = appearance_extractor(
            torch.from_numpy(frame_bchw).to(device)
        ).float().cpu().numpy()  # (1, 32, 16, 64, 64)
    np.save(f_s_path, f_s)

    # ── Extract source keypoints ──────────────────────────────────────
    with torch.no_grad():
        me_out = motion_extractor(torch.from_numpy(frame_bchw).to(device))
        kp_info = {}
        output_names = ["pitch", "yaw", "roll", "t", "exp", "scale", "kp"]
        for i, name in enumerate(output_names):
            kp_info[name] = me_out[i].float().cpu().numpy()
        kp_info["exp"] = kp_info["exp"].reshape(1, -1)
        kp_info["kp"] = kp_info["kp"].reshape(1, -1)

    # Transform source keypoints
    x_s = transform_keypoint_np(kp_info)  # (1, 21, 3)
    np.save(x_s_path, x_s.astype(np.float32))
    np.save(kp_c_path, kp_info['kp'].astype(np.float32))  # (1, 63) canonical kp

    # ── Compute mel spectrogram ───────────────────────────────────────
    mel = compute_mel_spectrogram(wav_path, fps=fps)
    mel_windows = mel_to_syncnet_windows(mel, fps=fps, num_frames=num_frames)

    # ── Read all video frames for GT lip crops ────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    all_frames_256 = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        f256 = cv2.resize(frame_rgb, (256, 256), interpolation=cv2.INTER_AREA)
        all_frames_256.append(f256)
    cap.release()

    N_frames = len(all_frames_256)
    N_windows = min(len(mel_windows), N_frames - num_frames + 1)

    if N_windows <= 0:
        print(f"  ⚠ Video too short ({N_frames} frames), skipping: {video_path}")
        # Save empty arrays
        np.save(A_path, np.zeros((1, 512), dtype=np.float32))
        np.save(sim_gt_path, np.zeros((1,), dtype=np.float32))
        return {
            'lipsync_f_s': f_s_path,
            'lipsync_x_s': x_s_path,
            'lipsync_kp_canonical': kp_c_path,
            'lipsync_A': A_path,
            'lipsync_sim_gt': sim_gt_path,
        }

    # ── Process windows ───────────────────────────────────────────────
    all_A = []
    all_sim_gt = []

    batch_size_proc = 16  # Process in batches for efficiency

    for start_idx in range(0, N_windows, batch_size_proc):
        end_idx = min(start_idx + batch_size_proc, N_windows)
        batch_mel = []
        batch_lips = []

        for w_idx in range(start_idx, end_idx):
            # Audio: mel window
            mel_win = mel_windows[w_idx]  # (1, 80, ~16)
            batch_mel.append(mel_win)

            # Visual: 5 consecutive lip crops, stacked
            lip_frames = []
            for f_offset in range(num_frames):
                f_idx = w_idx + f_offset
                if f_idx < N_frames:
                    lip = extract_lip_crop(all_frames_256[f_idx])
                else:
                    lip = extract_lip_crop(all_frames_256[-1])
                # Normalize to [0, 1]
                lip_norm = lip.astype(np.float32) / 255.0
                lip_frames.append(lip_norm.transpose(2, 0, 1))  # (3, 48, 96)

            # Stack 5 frames → (15, 48, 96)
            lips_stacked = np.concatenate(lip_frames, axis=0)  # (15, 48, 96)
            batch_lips.append(lips_stacked)

        # Convert to tensors
        mel_tensor = torch.from_numpy(np.array(batch_mel)).float().to(device)  # (bs, 1, 80, 16)
        lips_tensor = torch.from_numpy(np.array(batch_lips)).float().to(device)  # (bs, 15, 48, 96)

        with torch.no_grad():
            audio_emb, face_emb = syncnet(mel_tensor, lips_tensor)
            # Both: (bs, 512) L2-normalized

            # Cosine similarity
            sim = F.cosine_similarity(audio_emb, face_emb, dim=1)  # (bs,)

        all_A.append(audio_emb.cpu().numpy())
        all_sim_gt.append(sim.cpu().numpy())

    # Concatenate all windows
    A_array = np.concatenate(all_A, axis=0)         # (N_windows, 512)
    sim_gt_array = np.concatenate(all_sim_gt, axis=0)  # (N_windows,)

    np.save(A_path, A_array.astype(np.float32))
    np.save(sim_gt_path, sim_gt_array.astype(np.float32))

    return {
        'lipsync_f_s': f_s_path,
        'lipsync_x_s': x_s_path,
        'lipsync_kp_canonical': kp_c_path,
        'lipsync_A': A_path,
        'lipsync_sim_gt': sim_gt_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute lip-sync features for Ditto training")
    parser.add_argument("-i", "--data_info_json", required=True, help="Path to data_info.json")
    parser.add_argument("--syncnet_ckpt", required=True, help="Path to Wav2Lip SyncNet checkpoint")
    parser.add_argument("--ditto_pytorch_path", required=True, help="Path to ditto_pytorch/")
    parser.add_argument("--device", default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total number of GPUs")
    parser.add_argument("--gpu_id", type=int, default=0, help="This GPU's ID (for sharding)")
    parser.add_argument("--output_key", default="lipsync", help="Key prefix in data_info.json")
    args = parser.parse_args()

    device = args.device
    print(f"\n{'='*60}")
    print(f"  Lip-Sync Feature Preprocessing")
    print(f"  GPU {args.gpu_id + 1}/{args.num_gpus}  |  Device: {device}")
    print(f"{'='*60}\n")

    # ── Load models ───────────────────────────────────────────────────
    print("Loading models...")

    # SyncNet
    if MOTIONDIT_DIR not in sys.path:
        sys.path.insert(0, MOTIONDIT_DIR)
    from src.models.syncnet import load_syncnet
    syncnet = load_syncnet(args.syncnet_ckpt, device)

    # Set up LivePortrait imports (handles relative imports like `from .util import ...`)
    _setup_liveportrait_imports()

    # LivePortrait AppearanceFeatureExtractor
    from lp_modules.appearance_feature_extractor import AppearanceFeatureExtractor
    app_ext = AppearanceFeatureExtractor()
    app_ext_ckpt = os.path.join(args.ditto_pytorch_path, "models", "appearance_extractor.pth")
    app_ext.load_state_dict(torch.load(app_ext_ckpt, map_location="cpu", weights_only=True))
    app_ext = app_ext.to(device).eval()
    app_ext.requires_grad_(False)
    print(f"  AppearanceExtractor: {app_ext_ckpt}")

    # LivePortrait MotionExtractor
    from lp_modules.motion_extractor import MotionExtractor
    mot_ext = MotionExtractor()
    mot_ext_ckpt = os.path.join(args.ditto_pytorch_path, "models", "motion_extractor.pth")
    mot_ext.load_state_dict(torch.load(mot_ext_ckpt, map_location="cpu", weights_only=True))
    mot_ext = mot_ext.to(device).eval()
    mot_ext.requires_grad_(False)
    print(f"  MotionExtractor:     {mot_ext_ckpt}")

    print("All models loaded.\n")

    # ── Load data info ────────────────────────────────────────────────
    with open(args.data_info_json) as f:
        data_info = json.load(f)

    video_list = data_info.get("fps25_video_list", data_info.get("video_list", []))
    wav_list = data_info.get("wav_list", [])
    mtn_list = data_info.get("LP_npy_list", [])

    N = len(video_list)
    assert N > 0, f"No videos found in {args.data_info_json}"
    assert len(wav_list) == N, f"Mismatch: {N} videos but {len(wav_list)} wavs"

    # Shard across GPUs
    indices = list(range(args.gpu_id, N, args.num_gpus))
    print(f"Processing {len(indices)} / {N} videos on this GPU\n")

    # ── Initialize output lists in data_info ──────────────────────────
    for key in ['lipsync_f_s_list', 'lipsync_x_s_list', 'lipsync_kp_canonical_list',
                'lipsync_A_list', 'lipsync_sim_gt_list']:
        if key not in data_info:
            data_info[key] = [""] * N

    # ── Process videos ────────────────────────────────────────────────
    from tqdm import tqdm

    success = 0
    errors = 0

    for idx in tqdm(indices, desc=f"GPU{args.gpu_id}"):
        try:
            video_path = video_list[idx]
            wav_path = wav_list[idx]
            mtn_npy_path = mtn_list[idx] if idx < len(mtn_list) else ""

            # Output directory: same directory as the video
            output_dir = os.path.dirname(video_path)

            paths = process_one_video(
                video_path, wav_path, mtn_npy_path,
                output_dir,
                app_ext, mot_ext, syncnet, device,
            )

            # Update data_info
            data_info['lipsync_f_s_list'][idx] = paths['lipsync_f_s']
            data_info['lipsync_x_s_list'][idx] = paths['lipsync_x_s']
            data_info['lipsync_kp_canonical_list'][idx] = paths['lipsync_kp_canonical']
            data_info['lipsync_A_list'][idx] = paths['lipsync_A']
            data_info['lipsync_sim_gt_list'][idx] = paths['lipsync_sim_gt']

            success += 1
        except Exception:
            errors += 1
            traceback.print_exc()
            print(f"  ⚠ Error processing video {idx}: {video_list[idx]}")

    # ── Save results ──────────────────────────────────────────────────
    # In multi-GPU mode, each GPU writes a shard file to avoid race conditions.
    # The shell script merges shards after all GPUs finish.
    if args.num_gpus > 1:
        shard_file = args.data_info_json.replace('.json', f'_lipsync_shard_{args.gpu_id}.json')
        shard_data = {}
        for idx in indices:
            shard_data[str(idx)] = {
                'lipsync_f_s': data_info.get('lipsync_f_s_list', [''] * N)[idx],
                'lipsync_x_s': data_info.get('lipsync_x_s_list', [''] * N)[idx],
                'lipsync_kp_canonical': data_info.get('lipsync_kp_canonical_list', [''] * N)[idx],
                'lipsync_A': data_info.get('lipsync_A_list', [''] * N)[idx],
                'lipsync_sim_gt': data_info.get('lipsync_sim_gt_list', [''] * N)[idx],
            }
        with open(shard_file, 'w') as f:
            json.dump(shard_data, f, indent=2)
        print(f"  Shard saved: {shard_file}")
    else:
        # Single GPU: safe to write directly
        with open(args.data_info_json, 'w') as f:
            json.dump(data_info, f, indent=2)
        print(f"  Updated: {args.data_info_json}")

    print(f"\n{'='*60}")
    print(f"  Done! GPU {args.gpu_id}: {success} succeeded, {errors} errors")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
