import os
from tqdm import trange
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback
import numpy as np

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import load_json, dump_json


"""
[
    [motion_npy, aud_npy, frame_num]
]
"""


def check_one_v2(data, N_thre=81):
    """
    All modalities must exist on disk and share a common temporal length >= N_thre.
    N_thre should be seq_frames + 1 for training (e.g. 81 when seq_frames=80).
    """
    ns = []
    lengths = {}
    for k, v in data.items():
        if not os.path.isfile(v):
            return False, None, f"missing file ({k}): {v}"
        n = int(np.load(v).shape[0])
        lengths[k] = n
        ns.append(n)

    N = min(ns)
    if N < N_thre:
        return False, None, (
            f"min length {N} < threshold {N_thre} (per-modality lengths: {lengths})"
        )
    return True, N, None


def gather_and_filter_data_list_for_s2_v2(
    data_list_dict, 
    save_json='',
    use_emo=False,
    use_eye_open=False,
    use_eye_ball=False,
    use_lmk=False,
    flip=False,
    aud_feat_name='hubert_aud_npy_list',
    min_frames=81,
    use_lip_sync=False,
):
    """
    frame_num, mtn, aud, emo, eye_open, eye_ball
    LP_npy_list, hubert_aud_npy_list, emo_npy_list, eye_open_npy_list, eye_ball_npy_list
    """
    lst = []
    sample_errors = []
    num_v = len(data_list_dict[aud_feat_name])
    for i in trange(num_v):
        try:
            data = {
                'mtn': data_list_dict['LP_npy_list'][i],
                'aud': data_list_dict[aud_feat_name][i],
            }
            if use_emo:
                data['emo'] = data_list_dict['emo_npy_list'][i]
            if use_eye_open:
                data['eye_open'] = data_list_dict['eye_open_npy_list'][i]
            if use_eye_ball:
                data['eye_ball'] = data_list_dict['eye_ball_npy_list'][i]
            if use_lmk:
                data['lmk'] = data_list_dict['MP_lmk_npy_list'][i]

            if use_lip_sync:
                data['lipsync_f_s'] = data_list_dict['lipsync_f_s_list'][i]
                data['lipsync_x_s'] = data_list_dict['lipsync_x_s_list'][i]
                data['lipsync_kp_canonical'] = data_list_dict['lipsync_kp_canonical_list'][i]
                data['lipsync_A'] = data_list_dict['lipsync_A_list'][i]
                data['lipsync_sim_gt'] = data_list_dict['lipsync_sim_gt_list'][i]

            if flip:
                for k in ['mtn', 'eye_open', 'eye_ball', 'lmk']:
                    if k in data:
                        data[k] = flip_path(data[k])

            flag, N, err = check_one_v2(data, N_thre=min_frames)
            if not flag:
                if len(sample_errors) < 8:
                    sample_errors.append((i, err))
                continue

            data['frame_num'] = N

            lst.append(data)
        except Exception:
            traceback.print_exc()
            if len(sample_errors) < 8:
                sample_errors.append((i, "exception while building sample (see traceback above)"))

    print(len(lst))
    if sample_errors and len(lst) == 0:
        print(
            "[gather] First samples failed checks (fix paths or run video prep / bridge extraction):"
        )
        for idx, msg in sample_errors:
            print(f"  [{idx}] {msg}")
    if save_json:
        dump_json(lst, save_json)
    return lst        


def flip_path(p):
    items = p.split('/')
    items[-2] = items[-2] + '_flip'
    p = '/'.join(items)
    return p


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data info json
    output_data_json: Annotated[str, tyro.conf.arg(aliases=["-o"])] = ""  # s2 data list json: [[motion_npy, aud_npy, frame_num]]

    use_emo: bool = False    # use_emo flag
    use_eye_open: bool = False    # use_eye_open flag
    use_eye_ball: bool = False    # use_eye_ball flag

    use_lmk: bool = False    # use_lmk flag

    dataset_version: str = "v2"    # dataset version: [v1, v2]

    with_flip: bool = False    # with flip flag

    aud_feat_name: str = "hubert_aud_npy_list"    # aud_feat_key: ['hubert_aud_npy_list']

    min_frames: int = 81    # must be >= seq_frames+1 (default seq_frames=80 → 81)

    use_lip_sync: bool = False    # include lip-sync feature paths


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json
    assert opt.output_data_json

    data_info = load_json(opt.input_data_json)

    if opt.dataset_version in ['v2']:
        if opt.with_flip:
            lst = gather_and_filter_data_list_for_s2_v2(
                data_info, 
                save_json='',
                use_emo=opt.use_emo,
                use_eye_open=opt.use_eye_open,
                use_eye_ball=opt.use_eye_ball,
                use_lmk=opt.use_lmk,
                aud_feat_name=opt.aud_feat_name,
                min_frames=opt.min_frames,
            )
            flip_lst = gather_and_filter_data_list_for_s2_v2(
                data_info, 
                save_json='',
                use_emo=opt.use_emo,
                use_eye_open=opt.use_eye_open,
                use_eye_ball=opt.use_eye_ball,
                use_lmk=opt.use_lmk,
                flip=True,
                aud_feat_name=opt.aud_feat_name,
                min_frames=opt.min_frames,
            )
            lst = lst + flip_lst
            dump_json(lst, opt.output_data_json)

        else:
            lst = gather_and_filter_data_list_for_s2_v2(
                data_info, 
                save_json=opt.output_data_json,
                use_emo=opt.use_emo,
                use_eye_open=opt.use_eye_open,
                use_eye_ball=opt.use_eye_ball,
                use_lmk=opt.use_lmk,
                aud_feat_name=opt.aud_feat_name,
                min_frames=opt.min_frames,
                use_lip_sync=opt.use_lip_sync,
            )

        if len(lst) == 0:
            print(
                "\n[gather] ERROR: training list is empty. Typical causes:\n"
                "  • Phase 1 was skipped but motion/eye/emo .npy files are missing or paths in\n"
                "    data_info.json point to another machine (delete data_info.json and re-run prep).\n"
                "  • Clips are shorter than --min-frames (need at least seq_frames+1 frames).\n"
                "  • bridge / HuBERT audio .npy paths wrong — check aud_feat_name and disk.\n"
            )
            sys.exit(1)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    main()

    """
    data_info = {
        'fps25_video_list': fps25_video_list,
        'video_list': video_list,
        'wav_list': wav_list,
        'hubert_aud_npy_list': hubert_aud_npy_list,
        'LP_pkl_list': LP_pkl_list,
        'LP_npy_list': LP_npy_list,
        'MP_lmk_npy_list': MP_lmk_npy_list,
        'eye_open_npy_list': eye_open_npy_list,
        'eye_ball_npy_list': eye_ball_npy_list,
        'emo_npy_list': emo_npy_list,
    }
    """
