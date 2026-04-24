# emo, eye, canonical keypoints
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import random
import traceback

from ..utils.utils import load_json, load_pkl, dump_pkl


FPS = 25


def norm_by_mean_var(arr, v_mean_var):
    mean = np.broadcast_to(v_mean_var[0], arr.shape)
    var = np.broadcast_to(v_mean_var[1], arr.shape)
    norm_arr = (arr - mean) / np.sqrt(var)
    return norm_arr


def norm_by_mean_std(arr, v_mean_std):
    mean = np.broadcast_to(v_mean_std[0], arr.shape)
    std = np.broadcast_to(v_mean_std[1], arr.shape)
    norm_arr = (arr - mean) / std
    return norm_arr


def denorm_by_mean_var(arr, v_mean_var):
    mean = np.broadcast_to(v_mean_var[0], arr.shape)
    var = np.broadcast_to(v_mean_var[1], arr.shape)
    denorm_arr = arr * np.sqrt(var) + mean
    return denorm_arr

# Regeneration hint shown when precomputed num_frames mismatches config
_REGEN_HINT = (
    "\n[LipSync] *** MISMATCH: precomputed lipsync_A was generated with "
    "num_frames={old} but training uses num_frames={new}. ***\n"
    "Regenerate with:\n"
    "  python ditto-train/preprocess_lipsync_features.py \\\n"
    "      -i /workspace/HDTF/data_info.json \\\n"
    "      --syncnet_ckpt ditto-train/checkpoints/lipsync_expert.pth \\\n"
    "      --ditto_pytorch_path ditto-train/checkpoints/ditto_pytorch \\\n"
    "      --num_frames {new}\n"
)


class Stage2Dataset(Dataset):
    def __init__(
            self,
            data_list_json,
            seq_len=int(3.2 * FPS),
            preload=False,
            cache=False,
            preload_pkl="",
            motion_feat_dim=100,
            motion_feat_start=0,
            motion_feat_offset_dim_se=None,
            use_eye_open=False,
            use_eye_ball=False,
            use_emo=False,
            use_sc=False,
            use_last_frame=False,
            use_lmk=False,
            use_cond_end=False,
            mtn_mean_var_npy="",
            reprepare_idx_map=False,
            use_lip_sync_loss=False,
            lip_sync_num_frames=5,    # must match preprocess_lipsync_features --num_frames
            use_lip_landmark_loss=False,   # load lmk window for landmark loss
            **kwargs):
        super().__init__()

        """
        data_list_json:
            [
                {
                    'frame_num': frame_num,
                    'aud': aud_npy,
                    'mtn': mtn_npy,
                    'emo': emo_npy,
                    'eye_open': eye_open_npy,
                    'eye_ball': eye_ball_npy,
                }
            ]
        motion_feat_offset_dim_se: for D-S
        """

        self.is_train = True
        self.preload = preload
        self.preload_pkl = preload_pkl
        self.cache = cache
        self.seq_len = seq_len
        self.motion_feat_dim = motion_feat_dim
        self.motion_feat_start = motion_feat_start
        self.motion_feat_offset_dim_se = motion_feat_offset_dim_se

        self.use_eye_open           = use_eye_open
        self.use_eye_ball           = use_eye_ball
        self.use_emo                = use_emo
        self.use_sc                 = use_sc
        self.use_last_frame         = use_last_frame
        self.use_lmk                = use_lmk
        self.use_cond_end           = use_cond_end
        self.use_lip_sync_loss      = use_lip_sync_loss
        self.lip_sync_num_frames    = lip_sync_num_frames
        self.use_lip_landmark_loss  = use_lip_landmark_loss
        self._compat_warned         = False   # warn once on meta mismatch

        self.data_list_json = data_list_json

        if preload and preload_pkl and os.path.isfile(preload_pkl):
            print('load data from preload_pkl:', preload_pkl)
            self.v_list, self.idx_map = load_pkl(preload_pkl)
            self.num_v = len(self.v_list)
            if reprepare_idx_map:
                print('reprepare_idx_map...')
                self.idx_map = self._prepare_idx_map()
            self.num_seq = len(self.idx_map)
        else:
            self.v_list = self._load_data(data_list_json)
            self.num_v = len(self.v_list)

            self.idx_map = self._prepare_idx_map()
            self.num_seq = len(self.idx_map)

        if preload and preload_pkl and not os.path.isfile(preload_pkl):
            print('save data to preload_pkl:', preload_pkl)
            dump_pkl([self.v_list, self.idx_map], preload_pkl)

        self.cache_dict = {}
        print(f'load [num_v: {self.num_v}, num_seq: {self.num_seq}]')

        self.mtn_mean_var = None
        if mtn_mean_var_npy:
            self.mtn_mean_var = np.load(mtn_mean_var_npy)    # [2, dim]
            self.mtn_mean_var[1][self.mtn_mean_var[1] == 0] = 1e-8
            self.mtn_mean_std = self.mtn_mean_var.copy()
            self.mtn_mean_std[1] = np.sqrt(self.mtn_mean_std[1])

            # process preload
            if preload:
                print("norm for preload")
                for v_idx in trange(len(self.v_list)):
                    self.v_list[v_idx]['mtn'] = norm_by_mean_std(self.v_list[v_idx]['mtn'], self.mtn_mean_std)

    def __len__(self):
        return self.num_seq
    
    def _load_one(self, data):
        ss = self.motion_feat_start
        ee = ss + self.motion_feat_dim

        frame_num = data['frame_num']
        mtn_arr = np.load(data['mtn'])[:frame_num]
        mtn = mtn_arr[:frame_num, ss:ee]   # [n, dim_mtn]
        aud = np.load(data['aud'])[:frame_num]    # [n, dim_aud]

        arr_data = {
            'frame_num': frame_num,
            'mtn': mtn,
            'aud': aud,
        }
        if self.use_sc:
            sc = mtn_arr[:, 265:]    # [n, 63]
            arr_data['sc'] = sc

        if self.use_emo:
            emo = np.load(data['emo'])[:frame_num]   # [n, 8]
            arr_data['emo'] = emo

        if self.use_eye_open:
            eye_open = np.load(data['eye_open'])[:frame_num]    # [n, 2]
            arr_data['eye_open'] = eye_open

        if self.use_eye_ball:
            eye_ball = np.load(data['eye_ball'])[:frame_num]   # [n, 3, 2]
            eye_ball = eye_ball.reshape(frame_num, -1)    # [n, 6]
            arr_data['eye_ball'] = eye_ball
        
        if self.use_lmk:
            lmk = np.load(data['lmk'])[:frame_num]  # [n, 478, 3]
            lmk = lmk.reshape(frame_num, -1)
            arr_data['lmk'] = lmk

        # Lip-sync features (per-video, loaded once)
        if self.use_lip_sync_loss:
            try:
                lipsync_paths_ok = all(
                    data.get(k, '') != ''
                    for k in ['lipsync_f_s', 'lipsync_x_s',
                              'lipsync_kp_canonical', 'lipsync_A', 'lipsync_sim_gt']
                )
                if lipsync_paths_ok:
                    # Store only paths for large arrays to prevent OOM during PKL preload
                    arr_data['lipsync_f_s_path']           = data['lipsync_f_s']
                    arr_data['lipsync_x_s_path']           = data['lipsync_x_s']
                    arr_data['lipsync_kp_canonical_path']  = data['lipsync_kp_canonical']
                    
                    # Small arrays (per-window) are safe to preload
                    arr_data['lipsync_A']             = np.load(data['lipsync_A'])      # (N_win, 512)
                    arr_data['lipsync_sim_gt']        = np.load(data['lipsync_sim_gt']) # (N_win,)

                    # Sanity check via optional meta file.
                    # NOTE: meta['syncnet_frames'] is ALWAYS 5 (SyncNet architecture
                    # constraint). Do NOT compare it to lip_sync_num_frames (16) —
                    # that is the training render window, a completely different concept.
                    # Render-window slicing happens in lip_sync_loss.py, not here.
                    import json as _json, os as _os
                    meta_path = data['lipsync_A'].replace('lipsync_syncnet_A.npy',
                                                          'lipsync_meta.json')
                    if _os.path.isfile(meta_path):
                        with open(meta_path) as _mf:
                            meta = _json.load(_mf)
                        syncnet_nf = meta.get('syncnet_frames', meta.get('num_frames', 5))
                        if syncnet_nf != 5 and not self._compat_warned:
                            self._compat_warned = True
                            print(
                                f"\n[LipSync] WARNING: lipsync_meta.json reports "
                                f"syncnet_frames={syncnet_nf}, expected 5. "
                                f"SyncNet requires exactly 5 frames. Regenerate:\n"
                                f"  python ditto-train/preprocess_lipsync_features.py "
                                f"-i /workspace/HDTF/data_info.json "
                                f"--syncnet_ckpt checkpoints/lipsync_expert.pth "
                                f"--ditto_pytorch_path checkpoints/ditto_pytorch\n"
                            )
                            arr_data['lipsync_valid'] = False
                            return arr_data

                    arr_data['lipsync_valid'] = True
                else:
                    arr_data['lipsync_valid'] = False
            except Exception:
                arr_data['lipsync_valid'] = False


        return arr_data
    
    def _load_data(self, data_list_json):
        # [kps_npy, aud_npy, frame_num]
        data_list = load_json(data_list_json)

        if not self.preload:
            return data_list
        
        arr_data_list = []
        for data in tqdm(data_list):
            arr_data = self._load_one(data)
            arr_data_list.append(arr_data)

        return arr_data_list
    
    def _prepare_idx_map(self):
        # idx -> v_idx, f_idx
        num_v = self.num_v
        seq_len = self.seq_len

        idx_map = []
        for v_idx in range(num_v):
            num_f = self.v_list[v_idx]['frame_num']
            for f_idx in range(1, num_f - seq_len + 1):
                idx_map.append([v_idx, f_idx])
        return idx_map
    
    def getitem(self, idx):
        """
        return:
            kp_seq      # (B, L, kp_dim)
            kp_cond     # (B, kp_dim)
            aud_cond    # (B, L, aud_dim)
        """
        seq_len = self.seq_len

        v_idx, f_idx = self.idx_map[idx]

        if self.preload:
            arr_data = self.v_list[v_idx]
        else:
            if v_idx in self.cache_dict:
                arr_data = self.cache_dict[v_idx]
            else:
                data = self.v_list[v_idx]
                arr_data = self._load_one(data)
                if self.mtn_mean_var is not None:
                    arr_data['mtn'] = norm_by_mean_std(arr_data['mtn'], self.mtn_mean_std)
                if self.cache:
                    self.cache_dict[v_idx] = arr_data
        
        v_mtn = arr_data['mtn']
        v_aud = arr_data['aud']

        # if self.mtn_mean_var is not None:
        #     # v_mtn = norm_by_mean_var(v_mtn.copy(), self.mtn_mean_var)
        #     v_mtn = norm_by_mean_std(v_mtn, self.mtn_mean_std)

        if self.use_last_frame:
            kp_cond = v_mtn[f_idx - 1]
        else:
            kp_cond = v_mtn[random.randint(0, len(v_mtn) - 1)]

        if self.use_cond_end:
            if f_idx + seq_len < len(v_mtn):
                kp_cond_end = v_mtn[f_idx + seq_len]
            else:
                kp_cond_end = v_mtn[f_idx + seq_len - 1]
        else:
            kp_cond_end = None

        kp_seq = v_mtn[f_idx: f_idx + seq_len]
        aud_cond = v_aud[f_idx: f_idx + seq_len]

        if self.motion_feat_offset_dim_se:
            _s, _e = self.motion_feat_offset_dim_se
            kp_seq = kp_seq.copy()
            kp_seq[:, _s:_e] = kp_seq[:, _s:_e] - kp_cond[_s:_e][None]

        # other cond: emo, eye
        more_cond = []
        if self.use_emo:
            v_emo = arr_data['emo']
            emo_seq = v_emo[f_idx: f_idx + seq_len]   # [n, 8]
            emo_avg = np.mean(emo_seq, 0)   # [8]
            emo_avg_seq = np.stack([emo_avg] * seq_len, 0)   # [n, 8]
            more_cond.append(emo_avg_seq)
        
        if self.use_eye_open:
            v_eye_open = arr_data['eye_open']
            eye_open_seq = v_eye_open[f_idx: f_idx + seq_len]   # [n, 2]
            more_cond.append(eye_open_seq)

        if self.use_eye_ball:
            v_eye_ball = arr_data['eye_ball']
            eye_ball_seq = v_eye_ball[f_idx: f_idx + seq_len]   # [n, 6]
            more_cond.append(eye_ball_seq)

        rand_f_idx = random.randint(0, len(v_mtn) - 1)
        if self.use_sc:
            v_sc = arr_data['sc']
            sc = v_sc[rand_f_idx]    # [63]
            sc_seq = np.stack([sc] * seq_len, 0)    # [n, 63]
            more_cond.append(sc_seq)

        if self.use_lmk:
            v_lmk = arr_data['lmk']
            lmk = v_lmk[rand_f_idx]   # 478x3
            lmk_seq = np.stack([lmk] * seq_len, 0)  # [n, dim]
            more_cond.append(lmk_seq)
            
        if more_cond:
            cond_seq = np.concatenate([aud_cond] + more_cond, -1)    # [n, dim_cond]
        else:
            cond_seq = aud_cond

        data_dict = {
            'kp_seq': kp_seq,
            'kp_cond': kp_cond,
            'aud_cond': cond_seq,
            'idx': f'{idx}_{v_idx}_{f_idx}',
        }

        if self.use_cond_end:
            data_dict['kp_cond_end'] = kp_cond_end

        # Lip-sync data: return precomputed features for a random T-frame window
        T_sync = self.lip_sync_num_frames
        if self.use_lip_sync_loss and arr_data.get('lipsync_valid', False):
            max_start  = max(0, seq_len - T_sync)
            t_start    = random.randint(0, max_start)
            window_idx = f_idx + t_start

            A_arr      = arr_data['lipsync_A']
            sim_gt_arr = arr_data['lipsync_sim_gt']
            max_window = min(len(A_arr) - 1, len(sim_gt_arr) - 1)

            # Guard: if A array shorter than expected (old 5-frame data),
            # skip gracefully with a single warning
            if max_window < 0:
                if not self._compat_warned:
                    self._compat_warned = True
                    print(_REGEN_HINT.format(old='unknown', new=T_sync))
                data_dict['lipsync_valid'] = False
            else:
                window_idx = min(window_idx, max_window)

                try:
                    # Lazily load the large arrays on-demand
                    data_dict['lipsync_f_s']          = np.load(arr_data['lipsync_f_s_path']).squeeze(0)
                    data_dict['lipsync_x_s']          = np.load(arr_data['lipsync_x_s_path']).squeeze(0)
                    data_dict['lipsync_kp_canonical'] = np.load(arr_data['lipsync_kp_canonical_path']).squeeze(0)

                    data_dict['lipsync_A']            = A_arr[window_idx]
                    data_dict['lipsync_sim_gt']       = sim_gt_arr[window_idx]
                    data_dict['lipsync_t_start']      = t_start
                    data_dict['lipsync_valid']        = True

                    # Optional: landmark window for lip landmark loss
                    if self.use_lip_landmark_loss and self.use_lmk and 'lmk' in arr_data:
                        v_lmk      = arr_data['lmk']  # (N, 478*3)
                        lmk_start  = f_idx + t_start
                        lmk_end    = lmk_start + T_sync
                        if lmk_end <= len(v_lmk):
                            data_dict['lipsync_lmk_window'] = v_lmk[lmk_start:lmk_end]
                        else:
                            avail = v_lmk[lmk_start:]
                            pad   = np.stack([v_lmk[-1]] * (T_sync - len(avail)), axis=0)
                            data_dict['lipsync_lmk_window'] = np.concatenate([avail, pad], axis=0)

                except Exception as _e:
                    # Path is stale or file was deleted — mark invalid and warn once
                    data_dict['lipsync_valid'] = False
                    if not self._compat_warned:
                        self._compat_warned = True
                        _bad_path = arr_data.get('lipsync_f_s_path', 'unknown')
                        print(f"\n[LipSync] WARNING: Failed to lazy-load lipsync features: {_e}")
                        print(f"  Bad path: {_bad_path!r}")
                        print(f"  Rebuild the PKL with correct paths (see Cell FIX in notebook).\n")

        elif self.use_lip_sync_loss:
            data_dict['lipsync_valid'] = False
        
        return data_dict
    
    def __getitem__(self, idx):
        while True:
            try:
                return self.getitem(idx)
            except:
                traceback.print_exc()
                idx = random.randint(0, self.num_seq-1)
