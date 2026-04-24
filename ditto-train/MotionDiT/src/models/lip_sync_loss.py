"""
lip_sync_loss.py — Lip-Sync Loss Module for Ditto Training
============================================================
Adds SyncNet-based audio-visual synchronisation loss to the Ditto
motion-diffusion training pipeline.

Components:
    1. Differentiable keypoint transform (PyTorch reimplementation)
    2. Frozen LivePortrait renderer (WarpingNetwork + SPADEDecoder)
    3. Lip region extraction (fixed crop, no face detection needed)
    4. LipSyncLoss — end-to-end loss module

New features (v2):
    A) Soft hard-sample weighting — never fully detaches; clamp/huber modes
    B) Delay-aware timing penalty — renders ±max_shift offsets on-the-fly
    C) Lip landmark supervision — reuses MediaPipe 478-pt lmk array

SyncNet architecture constraint:
    The pretrained SyncNet face encoder's first conv weight is [32, 15, 7, 7],
    meaning it ALWAYS expects exactly 5 frames × 3 channels = 15 channels.
    This is fixed by the checkpoint and cannot be changed.
    SYNCNET_FRAMES = 5 is therefore a hard constant throughout this module.
    'num_frames' (default 16) is the RENDER WINDOW for supervision —
    we render more frames, then slice a 5-frame sub-window for SyncNet.

Gradient flow:
    loss → SyncNet face_encoder → lip crop → SPADEDecoder → WarpNetwork
         → keypoint transform → predicted motion → diffusion model
    (All frozen modules pass gradients through their computation graphs;
     only diffusion model weights are updated.)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# SyncNet architecture hard constant — DO NOT CHANGE
# The pretrained face encoder conv1 weight is [32, 15, 7, 7]:
#   15 channels = SYNCNET_FRAMES (5) × 3 RGB channels
# ---------------------------------------------------------------------------
SYNCNET_FRAMES = 5

# ---------------------------------------------------------------------------
# MediaPipe FaceMesh lip indices (478-point model)
# These 40 indices cover the full lip outline (inner + outer contours)
# ---------------------------------------------------------------------------
_LIP_INDICES = [
    0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95,
    146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314,
    317, 318, 321, 324, 375, 402, 405, 409, 415,
]

# LivePortrait 21-kp indices that correspond to the lower-face / mouth region
_LP_LIP_KP_INDICES = [13, 14, 15, 16, 17, 18, 19, 20]  # 8 keypoints


# ===========================================================================
# 1. Differentiable keypoint transform  (PyTorch version of motion_stitch.py)
# ===========================================================================

def bin66_to_degree_torch(pred):
    idx = torch.arange(66, device=pred.device, dtype=pred.dtype)
    pred_softmax = F.softmax(pred, dim=-1)
    degree = (pred_softmax * idx).sum(dim=-1) * 3.0 - 97.5
    return degree


def get_rotation_matrix_torch(pitch, yaw, roll):
    pitch = pitch / 180.0 * math.pi
    yaw   = yaw   / 180.0 * math.pi
    roll  = roll  / 180.0 * math.pi

    bs    = pitch.shape[0]
    ones  = torch.ones(bs, 1, device=pitch.device, dtype=pitch.dtype)
    zeros = torch.zeros(bs, 1, device=pitch.device, dtype=pitch.dtype)

    x = pitch.unsqueeze(1)
    y = yaw.unsqueeze(1)
    z = roll.unsqueeze(1)

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x),  torch.cos(x),
    ], dim=1).reshape(bs, 3, 3)

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y),
    ], dim=1).reshape(bs, 3, 3)

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z),  torch.cos(z), zeros,
        zeros, zeros, ones,
    ], dim=1).reshape(bs, 3, 3)

    rot = torch.bmm(torch.bmm(rot_z, rot_y), rot_x)
    rot = rot.transpose(1, 2)
    return rot


def transform_keypoint_torch(kp, scale, pitch_bin66, yaw_bin66, roll_bin66, t, exp):
    bs     = kp.shape[0]
    num_kp = 21
    kp_3d  = kp.reshape(bs, num_kp, 3)

    pitch = bin66_to_degree_torch(pitch_bin66)
    yaw   = bin66_to_degree_torch(yaw_bin66)
    roll  = bin66_to_degree_torch(roll_bin66)

    rot    = get_rotation_matrix_torch(pitch, yaw, roll)
    exp_3d = exp.reshape(bs, num_kp, 3)

    kp_rot         = torch.bmm(kp_3d, rot)
    kp_transformed = kp_rot + exp_3d
    kp_transformed = kp_transformed * scale.unsqueeze(-1)
    kp_transformed[:, :, 0:2] = kp_transformed[:, :, 0:2] + t[:, None, 0:2]
    return kp_transformed  # (B, 21, 3)


def motion_vec_to_keypoints(motion_265, kp_canonical):
    scale = motion_265[:, 0:1]
    pitch = motion_265[:, 1:67]
    yaw   = motion_265[:, 67:133]
    roll  = motion_265[:, 133:199]
    t     = motion_265[:, 199:202]
    exp   = motion_265[:, 202:265]
    x_d   = transform_keypoint_torch(kp_canonical, scale, pitch, yaw, roll, t, exp)
    return x_d  # (B, 21, 3)


# ===========================================================================
# 2. Lip region extraction
# ===========================================================================

def extract_lip_region(frames, lip_h=48, lip_w=96):
    _, _, H, W = frames.shape
    y_start = int(H * 0.55)
    y_end   = int(H * 0.95)
    x_start = int(W * 0.1)
    x_end   = int(W * 0.9)
    lips    = frames[:, :, y_start:y_end, x_start:x_end]
    if lips.shape[2] != lip_h or lips.shape[3] != lip_w:
        lips = F.interpolate(lips, size=(lip_h, lip_w),
                             mode='bilinear', align_corners=False)
    return lips


# ===========================================================================
# 3. Frozen renderer wrapper
# ===========================================================================

class FrozenRenderer(nn.Module):
    def __init__(self, warp_ckpt: str, decoder_ckpt: str, device: str = "cuda"):
        super().__init__()
        import sys, os, types

        cur_dir      = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(cur_dir)
        )))
        modules_dir  = os.path.join(
            project_root, "ditto-inference", "core", "models", "modules"
        )

        if 'lp_modules' not in sys.modules:
            pkg             = types.ModuleType('lp_modules')
            pkg.__path__    = [modules_dir]
            pkg.__package__ = 'lp_modules'
            pkg.__file__    = os.path.join(modules_dir, '__init__.py')
            sys.modules['lp_modules'] = pkg

        from lp_modules.warping_network import WarpingNetwork
        self.warp_net = WarpingNetwork()
        self.warp_net.load_state_dict(torch.load(warp_ckpt, map_location="cpu", weights_only=True))
        self.warp_net = self.warp_net.to(device).eval()
        self.warp_net.requires_grad_(False)

        from lp_modules.spade_generator import SPADEDecoder
        self.decoder = SPADEDecoder()
        self.decoder.load_state_dict(torch.load(decoder_ckpt, map_location="cpu", weights_only=True))
        self.decoder = self.decoder.to(device).eval()
        self.decoder.requires_grad_(False)

        self.device = device
        print(f"[FrozenRenderer] Loaded WarpingNetwork: {warp_ckpt}")
        print(f"[FrozenRenderer] Loaded SPADEDecoder:   {decoder_ckpt}")
        print(f"[FrozenRenderer] Total frozen params:   "
              f"{sum(p.numel() for p in self.parameters()):,}")

    def forward(self, f_s, x_s, x_d):
        warped   = self.warp_net(f_s, x_s, x_d)
        rendered = self.decoder(warped)
        return rendered  # (B, 3, H, W), values in [0, 1]


# ===========================================================================
# 4. Helper functions
# ===========================================================================

def compute_hard_weight(l_sync_val: float,
                        mode: str,
                        cap: float,
                        min_weight: float) -> float:
    """
    Compute a scalar sample weight from l_sync magnitude.

    Args:
        l_sync_val:  float value of l_sync (detached)
        mode:        'none' | 'clamp' | 'huber'
        cap:         loss value at which weight reaches min_weight
        min_weight:  floor — never returns 0

    Returns:
        weight: float in [min_weight, 1.0]
    """
    if mode == "none":
        return 1.0

    # Normalised excess above 1.0 (sync loss at random = 1.0)
    excess = max(0.0, l_sync_val - 1.0) / max(1e-6, cap - 1.0)
    excess = min(excess, 1.0)  # clamp to [0, 1]

    if mode == "clamp":
        weight = 1.0 - excess * (1.0 - min_weight)
    elif mode == "huber":
        # Smooth quadratic-to-linear transition
        if excess < 0.5:
            weight = 1.0 - 2.0 * excess ** 2 * (1.0 - min_weight)
        else:
            weight = min_weight + 2.0 * (1.0 - excess) ** 2 * (1.0 - min_weight)
        weight = max(min_weight, min(1.0, weight))
    else:
        weight = 1.0

    return max(min_weight, float(weight))


def _visual_embedding(syncnet_face_encoder, rendered_window,
                      B: int, lip_h: int, lip_w: int,
                      syncnet_A):
    """
    Shared helper: rendered 5-frame window → normalised visual embedding + sim.

    Args:
        rendered_window: (B, SYNCNET_FRAMES, 3, H, W)  — MUST be exactly 5 frames
    """
    T_in = rendered_window.shape[1]
    assert T_in == SYNCNET_FRAMES, (
        f"SyncNet face encoder requires exactly {SYNCNET_FRAMES} frames "
        f"({SYNCNET_FRAMES * 3} channels) but got {T_in} frames ({T_in * 3} ch). "
        f"This is a pretrained architecture constraint."
    )
    rendered_flat = rendered_window.reshape(B * T_in, 3,
                                            rendered_window.shape[-2],
                                            rendered_window.shape[-1])
    lips         = extract_lip_region(rendered_flat, lip_h, lip_w)
    lips         = lips.reshape(B, T_in, 3, lip_h, lip_w)
    lips_stacked = lips.reshape(B, T_in * 3, lip_h, lip_w)  # (B, 15, 48, 96)

    v   = syncnet_face_encoder(lips_stacked)
    v   = v.view(v.size(0), -1)
    v   = F.normalize(v, p=2, dim=1)           # (B, 512)

    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    sim = cos(syncnet_A, v)                     # (B,)
    return v, sim


# ===========================================================================
# 5. LipSyncLoss — main module
# ===========================================================================

class LipSyncLoss(nn.Module):
    """
    SyncNet-based lip synchronisation loss for Ditto training (v2).

    New vs v1:
        - Soft hard-sample weighting (never fully detaches)
        - Delay-aware timing penalty (on-the-fly ±N shift rendering)
        - Lip landmark supervision scaffold (reuses MediaPipe 478-pt lmk)
        - Richer diagnostics dict

    Gradient flow:
        loss → SyncNet → lip crop → SPADEDecoder → WarpNetwork
             → keypoint transform → predicted motion → diffusion model
    """

    def __init__(
        self,
        syncnet_ckpt: str,
        warp_ckpt: str,
        decoder_ckpt: str,
        device: str = "cuda",
        lip_h: int = 48,
        lip_w: int = 96,
        num_frames: int = 16,
        max_shift: int = 3,
    ):
        super().__init__()

        self.device         = device
        self.lip_h          = lip_h
        self.lip_w          = lip_w
        self.num_frames     = num_frames     # render window (training supervision)
        self.max_shift      = max_shift
        self.syncnet_frames = SYNCNET_FRAMES # always 5 — SyncNet architecture constraint

        # Center offset: where the 5-frame SyncNet window sits inside the render window
        # e.g. num_frames=16 → center_offset=5  (frames 5..9 of the 16-frame render)
        self.center_offset  = max(0, (num_frames - self.syncnet_frames) // 2)

        from .syncnet import load_syncnet
        # Both SyncNet and FrozenRenderer run entirely on CPU.
        # The whole lipsync pipeline (render → lip crop → SyncNet → loss)
        # runs on CPU. Only the final scalar loss moves to GPU.
        # This uses ZERO GPU VRAM for lipsync, solving OOM permanently.
        # Gradient flows: GPU x_recon → .cpu() → CPU pipeline → CPU loss scalar
        #                 → .to(device) → added to GPU main loss → .backward()
        self.syncnet  = load_syncnet(syncnet_ckpt, 'cpu')
        self.renderer = FrozenRenderer(warp_ckpt, decoder_ckpt, 'cpu')
        self._cpu = torch.device('cpu')

        self.cos_sim  = nn.CosineSimilarity(dim=1, eps=1e-6)

        print(f"[LipSyncLoss] render_window={num_frames}  syncnet_frames={SYNCNET_FRAMES}  "
              f"center_offset={self.center_offset}  max_shift={max_shift}")

        # Warn once if lmk is unavailable
        self._lmk_warned = False

    # -----------------------------------------------------------------------
    # Rendering helpers
    # -----------------------------------------------------------------------

    def render_frames(self, pred_motion_window, kp_canonical, f_s, x_s,
                      grad_frame_range=None):
        """
        Render on CPU. pred_motion_window arrives as CPU tensor.
        Returns tensor on CPU. Final loss scalar is moved to GPU in forward().
        """
        B, T, _ = pred_motion_window.shape
        f_s_cpu  = f_s.cpu()
        x_s_cpu  = x_s.cpu()

        rendered_list = []
        for t in range(T):
            motion_t = pred_motion_window[:, t, :]
            x_d      = motion_vec_to_keypoints(motion_t, kp_canonical.cpu())

            needs_grad = (grad_frame_range is None or
                          grad_frame_range[0] <= t <= grad_frame_range[1])
            if needs_grad:
                frame = self.renderer(f_s_cpu, x_s_cpu, x_d)
            else:
                with torch.no_grad():
                    frame = self.renderer(f_s_cpu, x_s_cpu, x_d)
            rendered_list.append(frame)

        return torch.stack(rendered_list, dim=1)   # (B, T, 3, H, W) on CPU

    # -----------------------------------------------------------------------
    # Delay-aware penalty (B) — on-the-fly rendering at ±max_shift
    # -----------------------------------------------------------------------

    def _delay_aware_penalty(self, rendered_ext, syncnet_A,
                             sim_pred, B):
        """
        Compute delay-aware alignment penalty.

        We shift the 5-frame SyncNet evaluation window (center_offset + shift)
        within the extended render. Zero-shift is already computed differentiably
        in forward() and passed in as sim_pred.

        Args:
            rendered_ext: (B, T + 2*max_shift, 3, H, W)
            syncnet_A:    (B, 512)
            sim_pred:     (B,)  zero-shift sim, differentiable
            B:            int   batch size

        Returns:
            delay_penalty, best_shift (float), sim_best (float)
        """
        s_range = list(range(-self.max_shift, self.max_shift + 1))

        # Zero-shift already computed — reuse (detached copy for argmax)
        all_sims = {0: sim_pred.detach()}

        for s in s_range:
            if s == 0:
                continue
            # Shift the centered 5-frame window by s frames
            s_start = self.max_shift + self.center_offset + s
            s_end   = s_start + self.syncnet_frames          # always 5

            with torch.no_grad():
                rendered_s = rendered_ext[:, s_start:s_end, :, :, :]  # (B,5,3,H,W)
                _, sim_s   = _visual_embedding(
                    self.syncnet.face_encoder,
                    rendered_s, B,
                    self.lip_h, self.lip_w,
                    syncnet_A,
                )
                all_sims[s] = sim_s

        sims_list              = torch.stack([all_sims[s] for s in s_range], dim=1)
        sim_best_per, best_idx = sims_list.detach().max(dim=1)       # (B,)
        best_shift             = (best_idx.float() - self.max_shift).mean().item()

        # Push zero-shift sim toward sim_best (gradient through sim_pred only)
        delay_penalty = (sim_best_per - sim_pred).clamp(min=0.0).mean()

        return delay_penalty, best_shift, sim_best_per.mean().item()

    # -----------------------------------------------------------------------
    # Lip landmark loss (D) — MediaPipe 478-pt reuse
    # -----------------------------------------------------------------------

    def _landmark_loss(self, pred_kp_window, lmk_window):
        """
        Compute L1 loss between predicted driving-kp XY and GT lip landmarks.

        Args:
            pred_kp_window: (B, T, 21, 3)  predicted driving keypoints
            lmk_window:     (B, T, 478*3)  raw MediaPipe 478-pt landmarks

        Returns:
            lmk_loss: scalar (or 0 if shapes incompatible)
        """
        try:
            B, T, _ = lmk_window.shape
            # Reshape to (B, T, 478, 3) and take XY
            lmk_3d  = lmk_window.reshape(B, T, 478, 3)
            lmk_lip = lmk_3d[:, :, _LIP_INDICES, :2]  # (B, T, 40, 2)

            # Normalize lip landmarks to [0, 1] using per-sample bounding box
            lmk_min = lmk_lip.amin(dim=(1, 2, 3), keepdim=True)
            lmk_max = lmk_lip.amax(dim=(1, 2, 3), keepdim=True)
            lmk_norm = (lmk_lip - lmk_min) / (lmk_max - lmk_min + 1e-6)  # (B,T,40,2)

            # Predicted lip keypoints: subset of LP 21-kp
            pred_lip = pred_kp_window[:, :, _LP_LIP_KP_INDICES, :2]  # (B, T, 8, 2)

            # Align predicted kp to same normalised range
            pred_min  = pred_lip.amin(dim=(1, 2, 3), keepdim=True)
            pred_max  = pred_lip.amax(dim=(1, 2, 3), keepdim=True)
            pred_norm = (pred_lip - pred_min) / (pred_max - pred_min + 1e-6)  # (B,T,8,2)

            # Compare centroid as proxy (avoids index-count mismatch)
            gt_centroid   = lmk_norm.mean(dim=2)    # (B, T, 2)
            pred_centroid = pred_norm.mean(dim=2)    # (B, T, 2)

            loss = F.l1_loss(pred_centroid, gt_centroid.detach())
            return loss

        except Exception as e:
            if not self._lmk_warned:
                print(f"[LipSyncLoss] Landmark loss skipped: {e}")
                self._lmk_warned = True
            return torch.tensor(0.0, device=self.device)

    # -----------------------------------------------------------------------
    # Main forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        pred_motion_window,   # (B, T_ext, 265) where T_ext = T + 2*max_shift
        kp_canonical,         # (B, 63)
        f_s,                  # (B, 32, 16, 64, 64)
        x_s,                  # (B, 21, 3)
        syncnet_A,            # (B, 512)
        sim_gt,               # (B,)
        # — new params with safe defaults (backward compatible) —
        hard_mode: str = "clamp",
        hard_cap: float = 2.0,
        hard_min_weight: float = 0.2,
        use_delay_aware: bool = False,
        delay_mode: str = "best_shift_penalty",
        lmk_window=None,      # (B, T, 478*3) optional
        debug: bool = False,
        debug_dir=None,
        debug_step: int = 0,
    ):
        """
        Compute lip-sync loss (v2).

        Returns:
            l_sync:        scalar — 1 - cos(A, V_pred)  [differentiable]
            l_stable:      scalar — |sim_gt - sim_pred|  [differentiable]
            delay_penalty: scalar — timing alignment penalty
            sim_pred:      float  — mean predicted similarity (detached, logging)
            best_shift:    float  — mean best shift in frames (detached, logging)
            hard_weight:   float  — computed sample weight (logging)
            lmk_loss:      scalar — landmark L1 loss (0 if disabled/unavailable)
        """
        B       = pred_motion_window.shape[0]
        T       = self.num_frames                      # render window (e.g. 8)
        SF      = self.syncnet_frames                  # always 5 — SyncNet constraint
        T_ext   = T + 2 * self.max_shift               # extended length for delay-aware

        # ── Move ALL inputs to CPU — entire lipsync pipeline runs on CPU ──
        # This uses ZERO GPU VRAM. Only scalar loss is moved to GPU at the end.
        pred_motion_window = pred_motion_window.cpu()
        kp_canonical       = kp_canonical.cpu()
        f_s                = f_s.cpu()
        x_s                = x_s.cpu()
        syncnet_A          = syncnet_A.cpu()
        sim_gt             = sim_gt.cpu()

        # Validate / pad window length
        if pred_motion_window.shape[1] != T_ext:
            actual = pred_motion_window.shape[1]
            if actual < T_ext:
                pad = pred_motion_window[:, -1:, :].expand(B, T_ext - actual, -1)
                pred_motion_window = torch.cat([pred_motion_window, pad], dim=1)
            else:
                pred_motion_window = pred_motion_window[:, :T_ext, :]

        # ── z_start / z_end must be computed BEFORE render_frames call ──
        z_start = self.max_shift + self.center_offset
        z_end   = z_start + SF                         # always SF=5 frames

        # 1. Render FULL extended window on CPU (no GPU VRAM used).
        #    Only the SyncNet sub-window (z_start..z_end) needs gradients.
        grad_range = (z_start, z_end - 1)   # inclusive frame indices with grads
        rendered_ext = self.render_frames(
            pred_motion_window, kp_canonical, f_s, x_s,
            grad_frame_range=grad_range,
        )  # (B, T_ext, 3, H, W)  on CPU


        rendered_sync = rendered_ext[:, z_start:z_end, :, :, :]  # (B, 5, 3, H, W)

        if debug:
            print(f"  [RENDER] ext={rendered_ext.shape}  sync_slice=[{z_start}:{z_end}]")
            print(f"  [RENDER] range: [{rendered_ext.min():.4f}, {rendered_ext.max():.4f}]")
            diff = (rendered_sync[:, 1:] - rendered_sync[:, :-1]).abs().mean()
            print(f"  [RENDER] inter-frame diff (sync window): {diff:.6f}")

        # 3. Visual embedding — always exactly 5 frames → 15 channels (SyncNet constraint)
        v_pred, sim_per_sample = _visual_embedding(
            self.syncnet.face_encoder,
            rendered_sync, B,          # NOTE: no T arg — inferred from shape (must be 5)
            self.lip_h, self.lip_w,
            syncnet_A,
        )

        if debug:
            print(f"  [SYNCNET] v_pred norm: {v_pred.norm(dim=1).mean():.4f}")
            print(f"  [SIM] sim_pred mean={sim_per_sample.mean():.4f}")
            if debug_dir is not None:
                self._save_debug_images(rendered_sync, debug_dir, debug_step)

        # 4. Primary losses
        l_sync   = (1.0 - sim_per_sample).mean().clamp(0.0, 2.0)
        l_stable = torch.abs(sim_gt - sim_per_sample).mean().clamp(0.0, 2.0)

        # 5. Hard-sample soft weighting
        hard_weight = compute_hard_weight(
            float(l_sync.detach()), hard_mode, hard_cap, hard_min_weight
        )

        # 6. Delay-aware penalty (on CPU)
        if use_delay_aware and self.max_shift > 0:
            delay_penalty, best_shift, sim_best = self._delay_aware_penalty(
                rendered_ext, syncnet_A, sim_per_sample, B
            )
        else:
            delay_penalty = torch.tensor(0.0)   # CPU scalar
            best_shift    = 0.0
            sim_best      = float(sim_per_sample.mean().detach())

        # 7. Lip landmark loss — all on CPU, consistent with rest of pipeline
        if lmk_window is not None:
            pred_kp_list = []
            for t in range(T):
                mt  = pred_motion_window[:, self.max_shift + t, :]  # (B, 265) on CPU
                x_d = motion_vec_to_keypoints(mt, kp_canonical)      # (B, 21, 3)
                pred_kp_list.append(x_d)
            pred_kp_window = torch.stack(pred_kp_list, dim=1)        # (B, T, 21, 3)
            lmk_loss = self._landmark_loss(pred_kp_window, lmk_window.cpu())
        else:
            lmk_loss = torch.tensor(0.0)  # CPU scalar

        # ── Move loss scalars to GPU for addition with main training loss ──
        dev = self.device
        return (
            l_sync.to(dev),
            l_stable.to(dev),
            delay_penalty.to(dev),
            float(sim_per_sample.mean().detach()),
            best_shift,
            hard_weight,
            lmk_loss.to(dev),
        )

    # -----------------------------------------------------------------------
    # Debug image saving
    # -----------------------------------------------------------------------

    def _save_debug_images(self, rendered_zero, debug_dir, step):
        import os, cv2
        import numpy as np
        os.makedirs(debug_dir, exist_ok=True)
        try:
            T = rendered_zero.shape[1]
            for t in range(min(5, T)):
                frame    = rendered_zero[0, t].detach().cpu().clamp(0, 1)
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(debug_dir, f"step{step}_rendered_t{t}.png"),
                    cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                )
            print(f"  [DEBUG] Saved images to {debug_dir}")
        except Exception as e:
            print(f"  [DEBUG] Failed to save images: {e}")
