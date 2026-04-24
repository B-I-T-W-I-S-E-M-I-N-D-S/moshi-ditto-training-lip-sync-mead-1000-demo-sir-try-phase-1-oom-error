import torch
# torch.backends.cudnn.benchmark = True # disable CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR warning

import cv2
import copy
import numpy as np
import os
# import os.path as osp
# from rich.progress import track
from tqdm import trange, tqdm
import imageio

_TQDM_FRAME = max(1, int(os.environ.get("DITTO_PIPELINE_LOG_FRAME_EVERY", "25")))
import pickle

from .config.argument_config import ArgumentConfig
from .config.inference_config import InferenceConfig
from .config.crop_config import CropConfig
from .utils.cropper import Cropper
from .utils.camera import get_rotation_matrix, headpose_pred_to_degree
from .utils.video import images2video, concat_frames, get_fps, add_audio_to_video, has_audio_stream
from .utils.crop import _transform_img, prepare_paste_back, paste_back
from .utils.io import load_image_rgb, resize_to_limit, dump, load, load_video
from .utils.helper import mkdir, basename, dct2device, is_video, is_template, remove_suffix, is_image
from .utils.rprint import rlog as log
# # from .utils.viz import viz_lmk
from .live_portrait_wrapper import LivePortraitWrapper


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def get_cfgs(pretrained_weights_path='', ditto_pytorch_path='', **kwargs):
    args = ArgumentConfig()
    kwargs_arg = {k: v for k, v in args.__dict__.items()}
    for k, v in kwargs.items():
        kwargs_arg[k] = v
    inference_cfg = partial_fields(InferenceConfig, kwargs_arg)
    crop_cfg = partial_fields(CropConfig, kwargs_arg)

    if ditto_pytorch_path:
        DITTO_PYTORCH_PATH = ditto_pytorch_path
        inference_cfg.checkpoint_F = f'{DITTO_PYTORCH_PATH}/models/appearance_extractor.pth'  # path to checkpoint of F
        inference_cfg.checkpoint_M = f'{DITTO_PYTORCH_PATH}/models/motion_extractor.pth'  # path to checkpoint pf M
        inference_cfg.checkpoint_G = f'{DITTO_PYTORCH_PATH}/models/decoder.pth'  # path to checkpoint of G
        inference_cfg.checkpoint_W = f'{DITTO_PYTORCH_PATH}/models/warp_network.pth'  # path to checkpoint of W
        inference_cfg.checkpoint_S = f'{DITTO_PYTORCH_PATH}/models/stitch_network.pth'  # path to checkpoint to S and R_eyes, R_lip

        crop_cfg.insightface_root = f"{DITTO_PYTORCH_PATH}/aux_models/insightface"
        crop_cfg.landmark_ckpt_path = f"{DITTO_PYTORCH_PATH}/aux_models/landmark203.onnx"

    elif pretrained_weights_path:
        PRETRAINED_WEIGHTS_PATH = pretrained_weights_path

        inference_cfg.checkpoint_F = f'{PRETRAINED_WEIGHTS_PATH}/liveportrait/base_models/appearance_feature_extractor.pth'  # path to checkpoint of F
        inference_cfg.checkpoint_M = f'{PRETRAINED_WEIGHTS_PATH}/liveportrait/base_models/motion_extractor.pth'  # path to checkpoint pf M
        inference_cfg.checkpoint_G = f'{PRETRAINED_WEIGHTS_PATH}/liveportrait/base_models/spade_generator.pth'  # path to checkpoint of G
        inference_cfg.checkpoint_W = f'{PRETRAINED_WEIGHTS_PATH}/liveportrait/base_models/warping_module.pth'  # path to checkpoint of W
        inference_cfg.checkpoint_S = f'{PRETRAINED_WEIGHTS_PATH}/liveportrait/retargeting_models/stitching_retargeting_module.pth'  # path to checkpoint to S and R_eyes, R_lip

        crop_cfg.insightface_root = f"{PRETRAINED_WEIGHTS_PATH}/insightface"
        crop_cfg.landmark_ckpt_path = f"{PRETRAINED_WEIGHTS_PATH}/liveportrait/landmark.onnx"

        # crop_cfg.xpose_ckpt_path = f"{PRETRAINED_WEIGHTS_PATH}/liveportrait_animals/xpose.pth"
    
    return inference_cfg, crop_cfg


class LP_Infer_SJ:
    def __init__(self, pretrained_weights_path='', **kwargs):

        inference_cfg, crop_cfg = get_cfgs(pretrained_weights_path, **kwargs)
        self.live_portrait_wrapper: LivePortraitWrapper = LivePortraitWrapper(inference_cfg=inference_cfg)
        self.cropper: Cropper = Cropper(crop_cfg=crop_cfg)

    def _driving_video_to_motion_info(self, video, flip=False):
        reader = imageio.get_reader(video, "ffmpeg")
        res = []
        for frame_rgb in reader:
            if flip:
                frame_rgb = frame_rgb[:, ::-1]    # [h, w, c]
            I_i = self.live_portrait_wrapper.prepare_source(frame_rgb)   # [1, 3, 256, 256]
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i, flag_refine_info=False)
            res_i = {k: v.cpu().numpy() for k, v in x_i_info.items()}
            res.append(res_i)
        reader.close()
        return res
    
    def _load_source(self, origin, template_pkl='', frame_num=-1):
        inf_cfg = self.live_portrait_wrapper.inference_cfg

        # origin: image/video
        if is_video(origin):
            is_image_flag = False
            rgb_lst = load_video(origin, frame_num)
        elif is_image(origin):
            is_image_flag = True
            img_rgb = load_image_rgb(origin)
            rgb_lst = [img_rgb]
        else:
            raise ValueError()
        
        rgb_lst = [resize_to_limit(img, inf_cfg.source_max_dim, inf_cfg.source_division) for img in rgb_lst]
        if template_pkl and os.path.isfile(template_pkl):
            with open(template_pkl, 'rb') as f:
                source_info = pickle.load(f)
        else:
            source_info = self._source_to_info(rgb_lst)
            if template_pkl:
                os.makedirs(os.path.dirname(template_pkl), exist_ok=True)
                with open(template_pkl, 'wb') as f:
                    pickle.dump(source_info, f)
        
        source_info['img_rgb_lst'] = rgb_lst
        source_info['is_image_flag'] = is_image_flag
        return source_info
    
    def _source_to_info(self, rgb_lst):
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg

        rgb_lst = [resize_to_limit(img, inf_cfg.source_max_dim, inf_cfg.source_division) for img in rgb_lst]
        crop_info_video = self.cropper.crop_source_video(rgb_lst, crop_cfg)
        img_crop_256x256_lst = crop_info_video['frame_crop_lst']
        I_s_lst = self.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
        x_s_info_lst = []
        f_s_lst = []
        ori_shape = rgb_lst[0].shape
        for I_i in tqdm(
            I_s_lst,
            miniters=_TQDM_FRAME,
            mininterval=1.0,
        ):
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i, flag_refine_info=False)
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_i)

            x_i_info = {k: v.cpu().numpy().astype(np.float32) for k, v in x_i_info.items()}
            f_s = f_s.cpu().numpy().astype(np.float32)
            x_s_info_lst.append(x_i_info)
            f_s_lst.append(f_s)

        source_info = {
            'x_s_info_lst': x_s_info_lst,
            'f_s_lst': f_s_lst,
            'M_c2o_lst': crop_info_video['M_c2o_lst'],
            'ori_shape': ori_shape,
        }

        return source_info
            
    def _get_source_info(self, imp):
        """
        imp -> source info
        """
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg

        img_rgb = load_image_rgb(imp)
        img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)
        log(f"Load source image from {imp}")

        crop_info = self.cropper.crop_source_image(img_rgb, crop_cfg)
        if crop_info is None:
            raise Exception("No face detected in the source image!")
        img_crop_256x256 = crop_info['img_crop_256x256']

        if inf_cfg.flag_do_crop:
            I_s = self.live_portrait_wrapper.prepare_source(img_crop_256x256)
        else:
            I_s = self.live_portrait_wrapper.prepare_source(img_rgb)
        x_s_info = self.live_portrait_wrapper.get_kp_info(I_s, flag_refine_info=False)
        f_s = self.live_portrait_wrapper.extract_feature_3d(I_s)
        source_info = {
            'x_s_info': x_s_info,
            'f_s': f_s,
            'crop_info': crop_info,
            'ori_shape': img_rgb.shape,
            'img_rgb': img_rgb,
        }
        return source_info
    
    def _get_source_info_video(self, video):
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        crop_cfg = self.cropper.crop_cfg
        rgb_lst = load_video(video)
        rgb_lst = [resize_to_limit(img, inf_cfg.source_max_dim, inf_cfg.source_division) for img in rgb_lst]
        crop_info_video = self.cropper.crop_source_video(rgb_lst, crop_cfg)
        img_crop_256x256_lst = crop_info_video['frame_crop_lst']
        I_s_lst = self.live_portrait_wrapper.prepare_videos(img_crop_256x256_lst)
        x_s_info_lst = []
        f_s_lst = []
        ori_shape = rgb_lst[0].shape
        for I_i in tqdm(
            I_s_lst,
            miniters=_TQDM_FRAME,
            mininterval=1.0,
        ):
            x_i_info = self.live_portrait_wrapper.get_kp_info(I_i, flag_refine_info=False)
            f_s = self.live_portrait_wrapper.extract_feature_3d(I_i)

            x_i_info = {k: v.cpu().numpy().astype(np.float32) for k, v in x_i_info.items()}
            f_s = f_s.cpu().numpy().astype(np.float32)
            x_s_info_lst.append(x_i_info)
            f_s_lst.append(f_s)
        
        source_info_video = {
            'x_s_info_lst': x_s_info_lst,
            'f_s_lst': f_s_lst,
            'crop_info_lst': crop_info_video,
            'img_rgb_lst': rgb_lst,
            'ori_shape': ori_shape,
        }

        return source_info_video
    
    def kp_info_to_x(self, kp_info):
        x = self.live_portrait_wrapper.transform_keypoint(dct2device(kp_info, self.live_portrait_wrapper.device))
        return x
    
    def _decode_one(self, f_s, x_s, x_d):
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        if inf_cfg.flag_stitching:
            x_d = self.live_portrait_wrapper.stitching(x_s, x_d)

        out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d)
        I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
        return I_p_i
    
    @staticmethod
    def _fmt_kp_info(kp_info):
        bs = kp_info['exp'].shape[0]
        kp_info['pitch'] = headpose_pred_to_degree(kp_info['pitch'])[:, None]  # Bx1
        kp_info['yaw'] = headpose_pred_to_degree(kp_info['yaw'])[:, None]  # Bx1
        kp_info['roll'] = headpose_pred_to_degree(kp_info['roll'])[:, None]  # Bx1
        kp_info['exp'] = kp_info['exp'].reshape(bs, -1, 3)  # BxNx3
        if 'kp' in kp_info:
            kp_info['kp'] = kp_info['kp'].reshape(bs, -1, 3)  # BxNx3
        return kp_info
    
    def _choose_d0(self, x_d_info_list, x_s_info, s0=False):
        _s = x_s_info['exp'].clone().view(*x_d_info_list[0]['exp'].shape)
        _s_device = _s.device
        if s0:
            _s = 0
        _min = float('inf')
        _idx = None
        for i in range(len(x_d_info_list)):
            _d = x_d_info_list[i]['exp']
            _diff = torch.abs(_d.to(_s_device) - _s).mean()
            if _diff < _min:
                _min = _diff
                _idx = i
        x_d_i_info = copy.deepcopy(x_d_info_list[_idx])
        return x_d_i_info
    
    def _cvt_x_d_info(self, x_s_info, x_d_info, x_d0_info=None, keys_mode_alpha={}):
        def _cvt_by_mode(key, mode='r', alpha=1):
            device = x_d_info[key].device
            if mode == 's':
                v = x_s_info[key].to(device) * alpha
            elif mode == 'd':
                v = x_d_info[key] * alpha
            else:    # r
                if key == 'scale':
                    v = x_s_info[key].to(device) * (x_d_info[key] / x_d0_info[key]) * alpha
                else:
                    v = x_s_info[key].to(device) + (x_d_info[key] - x_d0_info[key]) * alpha
            return v

        keys = ['scale', 'pitch', 'yaw', 'roll', 'exp', 't']
        new_x_d_info = {}
        for key in keys:
            mode, alpha = keys_mode_alpha.get(key, ('r', 1))
            v = _cvt_by_mode(key, mode, alpha)
            new_x_d_info[key] = v
        return new_x_d_info
    
    def _paste_back(self, I_p_i, img_rgb, M_c2o, ori_shape):
        inf_cfg = self.live_portrait_wrapper.inference_cfg
        mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, M_c2o, dsize=(ori_shape[1], ori_shape[0]))
        I_p_pstbk = paste_back(I_p_i, M_c2o, img_rgb, mask_ori_float)
        return I_p_pstbk
    
    def _decode_list_v2(self, source_info, x_d_info_list):
        inf_cfg = self.live_portrait_wrapper.inference_cfg

        x_s_info = source_info['x_s_info']
        f_s = source_info['f_s']
        crop_info = source_info['crop_info']
        ori_shape = source_info['ori_shape']
        img_rgb = source_info['img_rgb']

        x_s = self.kp_info_to_x(x_s_info)
        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(ori_shape[1], ori_shape[0]))
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")

        I_p_lst = []

        n_frames = len(x_d_info_list)
        for i in trange(n_frames):
            x_d_i_info = x_d_info_list[i]
            x_d_i_info['kp'] = x_s_info['kp']
            x_d = self.kp_info_to_x(x_d_i_info)

            I_p_i = self._decode_one(f_s, x_s, x_d)
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback:
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        if inf_cfg.flag_pasteback:
            return I_p_pstbk_lst
        else:
            return I_p_lst

    def _decode_list(self, source_info, x_d_info_list, keys_mode):

        inf_cfg = self.live_portrait_wrapper.inference_cfg
        device =  self.live_portrait_wrapper.device

        x_s_info = source_info['x_s_info']
        f_s = source_info['f_s']
        crop_info = source_info['crop_info']
        ori_shape = source_info['ori_shape']
        img_rgb = source_info['img_rgb']

        x_c_s = x_s_info['kp']
        R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        x_s = self.live_portrait_wrapper.transform_keypoint(x_s_info)

        I_p_pstbk_lst = None
        if inf_cfg.flag_pasteback:
            mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(ori_shape[1], ori_shape[0]))
            I_p_pstbk_lst = []
            log("Prepared pasteback mask done.")

        I_p_lst = []
        R_d_0, x_d_0_info = None, None

        def _choose_d0(x_d_info_list, x_s_info, s0=False):
            _s = x_s_info['exp'].clone().view(*x_d_info_list[0]['exp'].shape)
            _s_device = _s.device
            if s0:
                _s = 0
            _min = float('inf')
            _idx = None
            for i in range(len(x_d_info_list)):
                _d = x_d_info_list[i]['exp']
                _diff = torch.abs(_d.to(_s_device) - _s).mean()
                if _diff < _min:
                    _min = _diff
                    _idx = i
            x_d_i_info = copy.deepcopy(x_d_info_list[_idx])
            x_d_i_info = dct2device(x_d_i_info, device)
            x_d_i_info = self._fmt_kp_info(x_d_i_info)
            R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
            return R_d_i, x_d_i_info
        
        use_clip_r = keys_mode.get('use_clip_r', False)
        clip_len = keys_mode.get('clip_len', 80)
        ref_s0 = keys_mode.get('ref_s0', False)

        R_d_0, x_d_0_info = _choose_d0(x_d_info_list, x_s_info, s0=ref_s0)

        n_frames = len(x_d_info_list)
        for i in trange(n_frames):
            if use_clip_r and i % clip_len == 0:
                R_d_0, x_d_0_info = _choose_d0(x_d_info_list[i:i+clip_len], x_s_info, s0=ref_s0)

            x_d_i_info = x_d_info_list[i]
            x_d_i_info = dct2device(x_d_i_info, device)
            x_d_i_info = self._fmt_kp_info(x_d_i_info)
            R_d_i = get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])

            if i == 0:
                R_d_0 = R_d_i
                x_d_0_info = x_d_i_info

            if keys_mode.get('R', 'r') == 's':
                R_new = R_s
            elif keys_mode.get('R', 'r') == 'd':
                R_new = R_d_i
            else:  # r
                R_new = (R_d_i @ R_d_0.permute(0, 2, 1)) @ R_s

            if keys_mode.get('t', 'r') == 's':
                t_new = x_s_info['t']
            elif keys_mode.get('t', 'r') == 'd':
                t_new = x_d_i_info['t']
            else:
                t_new = x_s_info['t'] + (x_d_i_info['t'] - x_d_0_info['t'])

            if keys_mode.get('scale', 'r') == 's':
                scale_new = x_s_info['scale']
            elif keys_mode.get('scale', 'r') == 'd':
                scale_new = x_d_i_info['scale']
            else:
                scale_new = x_s_info['scale'] * (x_d_i_info['scale'] / x_d_0_info['scale'])

            if keys_mode.get('exp', 'r') == 's':
                delta_new = x_s_info['exp']
            elif keys_mode.get('exp', 'r') == 'd':
                delta_new = x_d_i_info['exp']
            elif keys_mode.get('exp', 'r') == 'o':
                delta_new = x_s_info['exp'] + x_d_i_info['exp']
            else:
                delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - x_d_0_info['exp'])

            t_new[..., 2].fill_(0) # zero tz
            x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new

            if inf_cfg.flag_stitching:
                # with stitching and without retargeting
                x_d_i_new = self.live_portrait_wrapper.stitching(x_s, x_d_i_new)
                
            out = self.live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
            I_p_i = self.live_portrait_wrapper.parse_output(out['out'])[0]
            I_p_lst.append(I_p_i)

            if inf_cfg.flag_pasteback:
                I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori_float)
                I_p_pstbk_lst.append(I_p_pstbk)

        if inf_cfg.flag_pasteback:
            return I_p_pstbk_lst
        else:
            return I_p_lst
        
    

     