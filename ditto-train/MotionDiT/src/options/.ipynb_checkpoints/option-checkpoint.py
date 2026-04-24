from __future__ import annotations
from typing import Tuple
from dataclasses import dataclass
import os


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(CUR_DIR)))
DEFAULT_EXPERIMENT_DIR = os.path.join(PROJECT_DIR, 'experiments', 's2')


class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __repr__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)
    

@dataclass(repr=False)  # use repr from PrintableConfig
class TrainOptions(PrintableConfig):
    ########## experiment ##########
    experiment_dir: str = DEFAULT_EXPERIMENT_DIR    # experiment_dir
    experiment_name: str = ""   # experiment_name

    ########## dataset ##########
    data_list_json: str = ""    # train data list: [[kps_npy, aud_npy, frame_num],]
    data_preload: bool = False  # data_preload flag
    data_preload_pkl: str = ""  # save to data_preload_pkl
    reprepare_idx_map: bool = False    # reprepare_idx_map flag for dataset
    data_cache: bool = False    # data_cache flag

    mtn_mean_var_npy: str = ""    # for mtn norm

    motion_feat_start: int = 0  # motion feat start dim
    motion_feat_offset_dim_se: tuple[int, ...] = ()    # cal offset dim range for part D-S

    use_emo: bool = False    # use_emo flag
    use_eye_open: bool = False    # use_eye_open flag
    use_eye_ball: bool = False    # use_eye_ball flag
    use_sc: bool = False    # use source canonical keypoints flag
    use_last_frame: bool = False    # use last frame as cond frame flag
    use_lmk: bool = False    # use mediapipe lmk as cond flag
    use_cond_end: bool = False    # use clip start and end frame as cond flag

    dataset_version: str = "v2"    # dataset version: [v1, v2]

    ########## model ##########
    motion_feat_dim: int = 265          # motion_feat_dim
    audio_feat_dim: int = 1103     # audio_feat_dim (1024 + 63 + 8 + 2 + 6)
    seq_frames: int = int(3.2 * 25)     # clip length

    ########## train ##########
    use_accelerate: bool = True         # use_accelerate flag for multi gpu
    epochs: int = 1000      # epochs
    batch_size: int = 512   # batch_size
    num_workers: int = 0    # num_workers
    lr: float = 1e-4        # lr
    part_w_dict_json: str = ""    # part loss weights dict json
    use_last_frame_loss: bool = False    # use_last_frame_loss flag
    use_reg_loss: bool = False    # use_reg_loss flag
    dim_ws_npy: str = ""    # dim_ws npy

    checkpoint: str = ""        # checkpoint for load

    save_ckpt_freq: int = 50    # save ckpt freq (epoch)

    ########## lip-sync loss ##########
    use_lip_sync_loss: bool = False         # enable SyncNet lip-sync loss
    lip_sync_lambda1: float = 0.05          # weight for L_sync = 1 - cos(A, V_pred)
    lip_sync_lambda2: float = 0.02          # weight for L_stable = |sim_gt - sim_pred|
    lip_sync_every_n_steps: int = 5         # compute lip-sync loss every N steps
    lip_sync_batch_size: int = 2            # number of samples per batch for lip-sync
    lip_sync_num_frames: int = 5            # consecutive frames for SyncNet (default 5)
    lip_sync_warmup_epochs: int = 30        # disable lip-sync loss for first N epochs
    lip_sync_use_stable: bool = False       # enable L_stable (disable initially)
    lip_sync_grad_clip: float = 0.1         # max gradient norm for lip-sync loss
    lip_sync_debug: bool = True              # enable verbose diagnostics (saves images, prints grads)
    syncnet_checkpoint: str = ""            # path to Wav2Lip SyncNet checkpoint
    ditto_pytorch_path: str = ""            # path to ditto_pytorch/ (for renderer weights)


def check_train_opt(opt: TrainOptions):
    assert opt.experiment_dir, opt.experiment_dir
    assert opt.experiment_name, opt.experiment_name
    assert opt.data_list_json, opt.data_list_json
