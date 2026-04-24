import os
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import numpy as np
from tqdm import tqdm

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MotionDiT_DIR = os.path.dirname(os.path.dirname(CUR_DIR))
sys.path.append(MotionDiT_DIR)

from MotionDiT.src.datasets.s2_dataset_v2 import Stage2Dataset as Stage2DatasetV2


def cal_mtn_mean_var(dataset: Stage2DatasetV2):
    lst = []
    for arr_data in tqdm(dataset.v_list):
        v_mtn = arr_data['mtn']    # [n, dim]
        lst.append(v_mtn)
    arr = np.concatenate(lst, 0)
    v_mean = arr.mean(0)
    v_var = arr.var(0)
    v_mean_var = np.stack([v_mean, v_var], 0)    # [2, dim]
    return v_mean_var


def preload_data_to_pkl(
    data_list_json, 
    preload_pkl,
    motion_feat_dim=265,
    motion_feat_start=0,
    dataset_version='v2',
    use_emo=False,
    use_eye_open=False,
    use_eye_ball=False,
    use_sc=False,    # source canonical keypoints
    use_lmk=False,
    use_last_frame=False,    # last frame as cond frame
    use_lip_sync_loss=False,  # include lip-sync features
    save_mtn_mean_var_npy='',
    seq_frames=int(3.2 * 25),
):
    if dataset_version in ['v2']:
        Stage2Dataset = Stage2DatasetV2
    else:
        raise NotImplementedError()

    print('FROM:')
    print('data_list_json:', data_list_json)
    print('TO:')
    print('preload_pkl:', preload_pkl)

    print('args:')
    print('motion_feat_dim:', motion_feat_dim)
    print('motion_feat_start:', motion_feat_start)
    print('use_sc:', use_sc)
    print('use_emo:', use_emo)
    print('use_eye_open:', use_eye_open)
    print('use_eye_ball:', use_eye_ball)

    dataset = Stage2Dataset(
        data_list_json=data_list_json,
        seq_len=seq_frames,
        preload=True,
        cache=False,
        preload_pkl=preload_pkl,
        motion_feat_dim=motion_feat_dim,
        motion_feat_start=motion_feat_start,
        use_emo=use_emo,
        use_eye_open=use_eye_open,
        use_eye_ball=use_eye_ball,
        use_sc=use_sc,
        use_last_frame=use_last_frame,
        use_lmk=use_lmk,
        use_lip_sync_loss=use_lip_sync_loss,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "Preload dataset is empty (num_seq=0). Fix bridge_data_list_json / feature paths "
            "or lower seq_frames / min_frames so clips exist."
        )
    print(len(dataset))
    
    if save_mtn_mean_var_npy:
        print('cal mean var')
        v_mean_var = cal_mtn_mean_var(dataset)
        os.makedirs(os.path.dirname(save_mtn_mean_var_npy), exist_ok=True)
        np.save(save_mtn_mean_var_npy, v_mean_var)


@dataclass
class Options:
    data_list_json: str = ""    # s2 data list json: [[motion_npy, aud_npy, frame_num]]
    data_preload_pkl: str = ""  # save to data_preload_pkl

    motion_feat_dim: int = 265  # motion_feat_dim
    motion_feat_start: int = 0  # motion feat start dim

    seq_frames: int = int(3.2 * 25)     # clip length

    use_emo: bool = False    # use_emo flag
    use_eye_open: bool = False    # use_eye_open flag
    use_eye_ball: bool = False    # use_eye_ball flag
    use_sc: bool = False    # use source canonical keypoints flag
    use_lmk: bool = False    # use_lmk flag
    use_last_frame: bool = False    # match train_bridge_ditto.py --use_last_frame
    use_lip_sync_loss: bool = False  # include lip-sync features in pickle

    dataset_version: str = "v2"    # dataset version: [v1, v2]
    save_mtn_mean_var_npy: str = ""    # save_mtn_mean_var_npy file


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.data_list_json
    assert opt.data_preload_pkl

    preload_data_to_pkl(
        opt.data_list_json,
        opt.data_preload_pkl,
        opt.motion_feat_dim,
        opt.motion_feat_start,
        dataset_version=opt.dataset_version,
        use_emo=opt.use_emo,
        use_eye_open=opt.use_eye_open,
        use_eye_ball=opt.use_eye_ball,
        use_sc=opt.use_sc,
        use_lmk=opt.use_lmk,
        use_last_frame=opt.use_last_frame,
        use_lip_sync_loss=opt.use_lip_sync_loss,
        save_mtn_mean_var_npy=opt.save_mtn_mean_var_npy,
        seq_frames=opt.seq_frames,
    )


if __name__ == '__main__':
    main()

