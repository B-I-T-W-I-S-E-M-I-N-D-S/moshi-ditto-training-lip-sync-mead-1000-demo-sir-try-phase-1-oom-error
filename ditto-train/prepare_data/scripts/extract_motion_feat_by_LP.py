import os
import numpy as np
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import dump_pkl, load_pkl, load_json, log_batch_progress
from LivePortrait.src.live_portrait_pipeline_sj import LP_Infer_SJ


"""
video to motion
1. video to pkl (dict)
2. pkl to npy (arr)
"""


def init_LP(ditto_pytorch_path):
    LP = LP_Infer_SJ(ditto_pytorch_path=ditto_pytorch_path)
    return LP


def video_to_motion_pkl(LP: LP_Infer_SJ, video, res_pkl, flip=False):
    if os.path.isfile(res_pkl):
        return
    res = LP._driving_video_to_motion_info(video, flip=flip)
    os.makedirs(os.path.dirname(res_pkl), exist_ok=True)
    dump_pkl(res, res_pkl)


def _cvt_LP_motion_info(inp, mode, ignore_keys=()):
    ks_shape_map = [
        ['scale', (1, 1), 1], 
        ['pitch', (1, 66), 66],
        ['yaw',   (1, 66), 66],
        ['roll',  (1, 66), 66],
        ['t',     (1, 3), 3], 
        ['exp', (1, 63), 63],
        ['kp',  (1, 63), 63],
    ]
    
    def _dic2arr(_dic):
        arr = []
        for k, _, ds in ks_shape_map:
            if k not in _dic or k in ignore_keys:
                continue
            v = _dic[k].reshape(ds)
            if k == 'scale':
                v = v - 1
            arr.append(v)
        arr = np.concatenate(arr, -1)  # (133)
        return arr
    
    def _arr2dic(_arr):
        dic = {}
        s = 0
        for k, ds, ss in ks_shape_map:
            if k in ignore_keys:
                continue
            v = _arr[s:s + ss].reshape(ds)
            if k == 'scale':
                v = v + 1
            dic[k] = v
            s += ss
            if s >= len(_arr):
                break
        return dic
    
    if mode == 'dic2arr':
        assert isinstance(inp, dict)
        return _dic2arr(inp)   # (dim)
    elif mode == 'arr2dic':
        assert inp.shape[0] >= 265
        return _arr2dic(inp)   # {k: (1, dim)}
    else:
        raise ValueError()
        


def cvt_live_motion_info(pkl, npy, mode='pkl2npy'):
    
    def _pkl_to_npy(pkl, npy):    
        lst = load_pkl(pkl)
        arr_list = []
        for _dic in lst:
            arr = _cvt_LP_motion_info(_dic, 'dic2arr')
            arr_list.append(arr)
        arr = np.stack(arr_list)   # (n, 133)
        arr = arr.astype(np.float32)
        os.makedirs(os.path.dirname(npy), exist_ok=True)
        np.save(npy, arr)
        return arr
        
    def _npy_to_pkl(npy, pkl):
        arr = np.load(npy)
        lst = []
        for i in range(len(arr)):
            _arr = arr[i]
            _dic = _cvt_LP_motion_info(_arr, 'arr2dic')
            lst.append(_dic)
        os.makedirs(os.path.dirname(pkl), exist_ok=True)
        dump_pkl(lst, pkl)
        return lst
            
    if mode == 'pkl2npy':
        ret = _pkl_to_npy(pkl, npy)
    elif mode == 'npy2pkl':
        ret = _npy_to_pkl(npy, pkl)
    else:
        raise ValueError()

    return ret


def flip_path(p):
    items = p.split('/')
    items[-2] = items[-2] + '_flip'
    p = '/'.join(items)
    return p


def process_data_list(video_list, pkl_list, npy_list, flip=False, ditto_pytorch_path=''):
    LP = init_LP(ditto_pytorch_path)
    total = len(video_list)
    for i, (video, pkl, npy) in enumerate(zip(video_list, pkl_list, npy_list), start=1):
        if flip:
            pkl = flip_path(pkl)
            npy = flip_path(npy)
        try:
            if not os.path.isfile(pkl):
                video_to_motion_pkl(LP, video, pkl, flip=flip)
            if not os.path.isfile(npy):
                cvt_live_motion_info(pkl, npy)
        except:
            traceback.print_exc()
        log_batch_progress(i, total, "extract_motion_feat_by_LP")


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data list json: {'video_list': video_list, 'LP_pkl_list': pkl_list, 'LP_npy_list': npy_list}
    flip_flag: bool = False    # flip video
    ditto_pytorch_path: str = ""  # ditto_pytorch_path


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json

    data_info = load_json(opt.input_data_json)

    video_list = data_info['video_list']
    pkl_list = data_info['LP_pkl_list']
    npy_list = data_info['LP_npy_list']

    process_data_list(video_list, pkl_list, npy_list, flip=opt.flip_flag, ditto_pytorch_path=opt.ditto_pytorch_path)

    
if __name__ == '__main__':
    main()
