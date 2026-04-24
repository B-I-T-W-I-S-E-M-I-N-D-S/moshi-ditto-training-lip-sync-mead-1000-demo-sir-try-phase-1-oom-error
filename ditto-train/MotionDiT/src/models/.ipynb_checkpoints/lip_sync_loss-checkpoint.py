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


# ===========================================================================
# 1. Differentiable keypoint transform  (PyTorch version of motion_stitch.py)
# ===========================================================================

def bin66_to_degree_torch(pred):
    """
    Convert bin66 classification to degree (differentiable).

    Args:
        pred: (B, 66) raw logits

    Returns:
        degree: (B,) in range roughly [-97.5, 100.5]
    """
    idx = torch.arange(66, device=pred.device, dtype=pred.dtype)
    pred_softmax = F.softmax(pred, dim=-1)
    degree = (pred_softmax * idx).sum(dim=-1) * 3.0 - 97.5
    return degree


def get_rotation_matrix_torch(pitch, yaw, roll):
    """
    Compute rotation matrix from Euler angles (differentiable).

    Args:
        pitch, yaw, roll: (B,) in degrees

    Returns:
        rot: (B, 3, 3) rotation matrix (transposed, matching LivePortrait convention)
    """
    # degrees → radians
    pitch = pitch / 180.0 * math.pi
    yaw = yaw / 180.0 * math.pi
    roll = roll / 180.0 * math.pi

    bs = pitch.shape[0]
    ones = torch.ones(bs, 1, device=pitch.device, dtype=pitch.dtype)
    zeros = torch.zeros(bs, 1, device=pitch.device, dtype=pitch.dtype)

    x = pitch.unsqueeze(1)  # (B, 1)
    y = yaw.unsqueeze(1)
    z = roll.unsqueeze(1)

    # Rotation around X axis (pitch)
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x),
    ], dim=1).reshape(bs, 3, 3)

    # Rotation around Y axis (yaw)
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y),
    ], dim=1).reshape(bs, 3, 3)

    # Rotation around Z axis (roll)
    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones,
    ], dim=1).reshape(bs, 3, 3)

    # Combined rotation: R = Rz @ Ry @ Rx, then transpose
    rot = torch.bmm(torch.bmm(rot_z, rot_y), rot_x)
    rot = rot.transpose(1, 2)  # LivePortrait convention
    return rot


def transform_keypoint_torch(kp, scale, pitch_bin66, yaw_bin66, roll_bin66, t, exp):
    """
    Transform canonical keypoints with motion parameters (differentiable).

    Equation: x_transformed = (kp @ R + exp) * scale + t

    Args:
        kp:           (B, 63)  canonical keypoints (flattened 21×3)
        scale:        (B, 1)   scale factor
        pitch_bin66:  (B, 66)  pitch in bin66 format
        yaw_bin66:    (B, 66)  yaw in bin66 format
        roll_bin66:   (B, 66)  roll in bin66 format
        t:            (B, 3)   translation
        exp:          (B, 63)  expression deformation (flattened 21×3)

    Returns:
        kp_transformed: (B, 21, 3)
    """
    bs = kp.shape[0]
    num_kp = 21

    kp_3d = kp.reshape(bs, num_kp, 3)

    pitch = bin66_to_degree_torch(pitch_bin66)
    yaw = bin66_to_degree_torch(yaw_bin66)
    roll = bin66_to_degree_torch(roll_bin66)

    rot = get_rotation_matrix_torch(pitch, yaw, roll)  # (B, 3, 3)

    exp_3d = exp.reshape(bs, num_kp, 3)

    # Eqn: s * (kp @ R + exp) + t
    kp_rot = torch.bmm(kp_3d, rot)  # (B, 21, 3) @ (B, 3, 3) = (B, 21, 3)
    kp_transformed = kp_rot + exp_3d
    kp_transformed = kp_transformed * scale.unsqueeze(-1)  # (B, 21, 3) * (B, 1, 1)
    kp_transformed[:, :, 0:2] = kp_transformed[:, :, 0:2] + t[:, None, 0:2]

    return kp_transformed  # (B, 21, 3)


def motion_vec_to_keypoints(motion_265, kp_canonical):
    """
    Convert 265-dim motion vector + canonical keypoints → driving keypoints.

    Args:
        motion_265:    (B, 265) predicted motion vector
        kp_canonical:  (B, 63)  canonical keypoints from source identity

    Returns:
        x_d: (B, 21, 3) driving keypoints
    """
    scale = motion_265[:, 0:1]         # (B, 1)
    pitch = motion_265[:, 1:67]        # (B, 66)
    yaw = motion_265[:, 67:133]        # (B, 66)
    roll = motion_265[:, 133:199]      # (B, 66)
    t = motion_265[:, 199:202]         # (B, 3)
    exp = motion_265[:, 202:265]       # (B, 63)

    x_d = transform_keypoint_torch(kp_canonical, scale, pitch, yaw, roll, t, exp)
    return x_d


# ===========================================================================
# 2. Lip region extraction
# ===========================================================================

def extract_lip_region(frames, lip_h=48, lip_w=96):
    """
    Extract lip/mouth region from rendered face frames using a fixed crop.

    SyncNet expects the LOWER HALF of a 96×96 face = (B, 3, 48, 96).
    Input frames are 256×256 face-aligned crops from LivePortrait.

    Args:
        frames: (B, 3, H, W)  rendered face images, values in [0, 1]

    Returns:
        lips: (B, 3, lip_h, lip_w)  cropped and resized lip regions
    """
    _, _, H, W = frames.shape

    # Crop region: lower 40% height, middle 80% width
    y_start = int(H * 0.55)
    y_end = int(H * 0.95)
    x_start = int(W * 0.1)
    x_end = int(W * 0.9)

    lips = frames[:, :, y_start:y_end, x_start:x_end]

    # Resize to SyncNet input size (48×96)
    if lips.shape[2] != lip_h or lips.shape[3] != lip_w:
        lips = F.interpolate(lips, size=(lip_h, lip_w),
                             mode='bilinear', align_corners=False)

    return lips


# ===========================================================================
# 3. Frozen renderer wrapper
# ===========================================================================

class FrozenRenderer(nn.Module):
    """
    Frozen LivePortrait renderer: WarpingNetwork + SPADEDecoder.

    Given appearance features (f_s), source keypoints (x_s), and driving
    keypoints (x_d), produces rendered face images.

    All weights are frozen; gradients flow through the computation graph.
    """

    def __init__(self, warp_ckpt: str, decoder_ckpt: str, device: str = "cuda"):
        super().__init__()

        import sys
        import os
        import types

        # Find the LivePortrait modules directory
        # lip_sync_loss.py is at: ditto-train/MotionDiT/src/models/lip_sync_loss.py
        # We need:               ditto-inference/core/models/modules/
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(cur_dir)
        )))
        modules_dir = os.path.join(
            project_root, "ditto-inference", "core", "models", "modules"
        )

        # Create synthetic package for LivePortrait modules
        # This allows relative imports like `from .util import ...` to work
        if 'lp_modules' not in sys.modules:
            pkg = types.ModuleType('lp_modules')
            pkg.__path__ = [modules_dir]
            pkg.__package__ = 'lp_modules'
            pkg.__file__ = os.path.join(modules_dir, '__init__.py')
            sys.modules['lp_modules'] = pkg

        # Load WarpingNetwork
        from lp_modules.warping_network import WarpingNetwork
        self.warp_net = WarpingNetwork()
        warp_sd = torch.load(warp_ckpt, map_location="cpu", weights_only=True)
        self.warp_net.load_state_dict(warp_sd)
        self.warp_net = self.warp_net.to(device)
        self.warp_net.eval()
        self.warp_net.requires_grad_(False)

        # Load SPADEDecoder
        from lp_modules.spade_generator import SPADEDecoder
        self.decoder = SPADEDecoder()
        dec_sd = torch.load(decoder_ckpt, map_location="cpu", weights_only=True)
        self.decoder.load_state_dict(dec_sd)
        self.decoder = self.decoder.to(device)
        self.decoder.eval()
        self.decoder.requires_grad_(False)

        self.device = device
        print(f"[FrozenRenderer] Loaded WarpingNetwork: {warp_ckpt}")
        print(f"[FrozenRenderer] Loaded SPADEDecoder:   {decoder_ckpt}")
        print(f"[FrozenRenderer] Total frozen params:   "
              f"{sum(p.numel() for p in self.parameters()):,}")

    def forward(self, f_s, x_s, x_d):
        """
        Render a face frame given appearance features and keypoints.

        Args:
            f_s: (B, 32, 16, 64, 64)  source appearance features
            x_s: (B, 21, 3)           source keypoints (transformed)
            x_d: (B, 21, 3)           driving keypoints (transformed)

        Returns:
            rendered: (B, 3, 256, 256)  rendered face image, values in [0, 1]
        """
        # Warp the source features according to driving motion
        warped = self.warp_net(f_s, x_s, x_d)  # (B, 256, 64, 64)

        # Decode warped features to image
        rendered = self.decoder(warped)  # (B, 3, H, W), values in [0, 1] (sigmoid)

        return rendered


# ===========================================================================
# 4. LipSyncLoss — main module
# ===========================================================================

class LipSyncLoss(nn.Module):
    """
    SyncNet-based lip synchronisation loss for Ditto training.

    Computes two losses:
        L_sync   = 1 - cos(A, V_pred)        [direct lip-sync reward]
        L_stable = |sim_gt - sim_pred|        [stabilised against GT]

    All sub-modules (SyncNet, WarpNetwork, SPADEDecoder) are frozen.
    Gradients flow through their computation graphs to the predicted motion.
    """

    def __init__(
        self,
        syncnet_ckpt: str,
        warp_ckpt: str,
        decoder_ckpt: str,
        device: str = "cuda",
        lip_h: int = 48,
        lip_w: int = 96,
        num_frames: int = 5,
    ):
        super().__init__()

        self.device = device
        self.lip_h = lip_h
        self.lip_w = lip_w
        self.num_frames = num_frames

        # Load frozen SyncNet
        from .syncnet import load_syncnet
        self.syncnet = load_syncnet(syncnet_ckpt, device)

        # Load frozen renderer (WarpingNetwork + SPADEDecoder)
        self.renderer = FrozenRenderer(warp_ckpt, decoder_ckpt, device)

        # Cosine similarity
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def render_frames(self, pred_motion_window, kp_canonical, f_s, x_s):
        """
        Render multiple frames from predicted motion parameters.

        Args:
            pred_motion_window: (B, T, 265) predicted motion for T frames
            kp_canonical:       (B, 63)     canonical keypoints (source identity)
            f_s:                (B, 32, 16, 64, 64) source appearance features
            x_s:                (B, 21, 3)  source transformed keypoints

        Returns:
            rendered_frames: (B, T, 3, 256, 256) rendered face images
        """
        B, T, _ = pred_motion_window.shape
        rendered_list = []

        for t in range(T):
            motion_t = pred_motion_window[:, t, :]  # (B, 265)

            # Convert motion vector to driving keypoints
            x_d = motion_vec_to_keypoints(motion_t, kp_canonical)  # (B, 21, 3)

            # Render frame
            frame = self.renderer(f_s, x_s, x_d)  # (B, 3, H, W)
            rendered_list.append(frame)

        rendered_frames = torch.stack(rendered_list, dim=1)  # (B, T, 3, H, W)
        return rendered_frames

    def forward(
        self,
        pred_motion_window,
        kp_canonical,
        f_s,
        x_s,
        syncnet_A,
        sim_gt,
        debug=False,
        debug_dir=None,
        debug_step=0,
    ):
        """
        Compute lip-sync loss.

        Args:
            pred_motion_window: (B, 5, 265) predicted motion for 5 consecutive frames
            kp_canonical:       (B, 63)     canonical keypoints
            f_s:                (B, 32, 16, 64, 64) source appearance features
            x_s:                (B, 21, 3)  source keypoints
            syncnet_A:          (B, 512)    precomputed SyncNet audio embedding
            sim_gt:             (B,)        precomputed GT similarity score
            debug:              save diagnostic images and print stats
            debug_dir:          directory for debug images
            debug_step:         training step number (for filenames)

        Returns:
            l_sync:   scalar — 1 - cos(A, V_pred)
            l_stable: scalar — |sim_gt - sim_pred|
            sim_pred: scalar — average predicted similarity (for logging)
        """
        B = pred_motion_window.shape[0]

        # 1. Render 5 predicted frames
        with torch.cuda.amp.autocast(enabled=True):
            rendered = self.render_frames(
                pred_motion_window, kp_canonical, f_s, x_s
            )  # (B, 5, 3, H, W)

        if debug:
            print(f"  [RENDER] output shape: {rendered.shape}")
            print(f"  [RENDER] value range: [{rendered.min():.4f}, {rendered.max():.4f}]")
            print(f"  [RENDER] requires_grad: {rendered.requires_grad}")
            # Check inter-frame differences in rendered output
            frame_diff = (rendered[:, 1:] - rendered[:, :-1]).abs().mean()
            print(f"  [RENDER] inter-frame pixel diff: {frame_diff:.6f}")

        # 2. Extract lip regions from each frame
        _, T, C, H, W = rendered.shape
        rendered_flat = rendered.reshape(B * T, C, H, W)
        lips = extract_lip_region(rendered_flat, self.lip_h, self.lip_w)  # (B*T, 3, 48, 96)
        lips = lips.reshape(B, T, 3, self.lip_h, self.lip_w)

        if debug:
            print(f"  [LIPS] crop shape: {lips.shape}")
            print(f"  [LIPS] value range: [{lips.min():.4f}, {lips.max():.4f}]")

        # 3. Stack 5 frames along channel dim for SyncNet input
        # SyncNet expects (B, 15, 48, 96) = 5 frames × 3 channels
        lips_stacked = lips.reshape(B, T * 3, self.lip_h, self.lip_w)  # (B, 15, 48, 96)

        # 4. Get visual embedding through frozen SyncNet face encoder
        #    (audio encoder is not needed — A is precomputed)
        v_pred = self.syncnet.face_encoder(lips_stacked)
        v_pred = v_pred.view(v_pred.size(0), -1)
        v_pred = F.normalize(v_pred, p=2, dim=1)  # (B, 512)

        if debug:
            print(f"  [SYNCNET] v_pred shape: {v_pred.shape}")
            print(f"  [SYNCNET] v_pred norm: {v_pred.norm(dim=1).mean():.4f}")
            print(f"  [SYNCNET] syncnet_A norm: {syncnet_A.norm(dim=1).mean():.4f}")

        # 5. Compute cosine similarity
        sim_pred = self.cos_sim(syncnet_A, v_pred)  # (B,)

        if debug:
            print(f"  [SIM] per-sample sim_pred: {sim_pred.detach().cpu().numpy()}")
            print(f"  [SIM] per-sample sim_gt:   {sim_gt.detach().cpu().numpy()}")
            print(f"  [SIM] sim_pred mean={sim_pred.mean():.4f} std={sim_pred.std():.4f}")

            # Save debug images
            if debug_dir is not None:
                self._save_debug_images(
                    rendered, lips, debug_dir, debug_step
                )

        # 6. Compute losses
        l_sync = (1.0 - sim_pred).mean()                       # lip-sync loss
        l_stable = torch.abs(sim_gt - sim_pred).mean()          # stabilised loss

        # Clamp for safety (avoid NaN)
        l_sync = torch.clamp(l_sync, min=0.0, max=2.0)
        l_stable = torch.clamp(l_stable, min=0.0, max=2.0)

        return l_sync, l_stable, sim_pred.mean().detach()

    def _save_debug_images(self, rendered, lips, debug_dir, step):
        """Save rendered frames and lip crops for visual inspection."""
        import os
        os.makedirs(debug_dir, exist_ok=True)

        try:
            # Save first sample's 5 rendered frames
            for t in range(min(5, rendered.shape[1])):
                frame = rendered[0, t].detach().cpu().clamp(0, 1)  # (3, H, W)
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                import cv2
                cv2.imwrite(
                    os.path.join(debug_dir, f"step{step}_rendered_t{t}.png"),
                    cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                )

            # Save first sample's 5 lip crops
            for t in range(min(5, lips.shape[1])):
                lip = lips[0, t].detach().cpu().clamp(0, 1)  # (3, 48, 96)
                lip_np = (lip.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(
                    os.path.join(debug_dir, f"step{step}_lipcrop_t{t}.png"),
                    cv2.cvtColor(lip_np, cv2.COLOR_RGB2BGR)
                )

            print(f"  [DEBUG] Saved images to {debug_dir}")
        except Exception as e:
            print(f"  [DEBUG] Failed to save images: {e}")
