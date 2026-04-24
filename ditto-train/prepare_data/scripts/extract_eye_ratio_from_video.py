import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
# from multiprocess import Pool

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import load_json, log_batch_progress


class MediaPipeUtils:
    def __init__(self, task_path='face_landmarker.task'):
        base_options = BaseOptions(model_asset_path=task_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        detector = vision.FaceLandmarker.create_from_options(options)
        self.detector = detector

    def detect_from_imp(self, imp):
        image = mp.Image.create_from_file(imp)
        detection_result = self.detector.detect(image)
        return detection_result

    def detect_from_npimage(self, img):
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        detection_result = self.detector.detect(image)
        return detection_result

    @staticmethod
    def mpbs_to_npbs(results):
        bs_list = results.face_blendshapes
        np_bs = []
        for bs in bs_list:
            bss = [t.score for t in bs]
            np_bs.append(bss)
        return np.array(np_bs).astype(np.float32)

    @staticmethod
    def mplmk_to_nplmk(results):
        face_landmarks_list = results.face_landmarks
        np_lms = []
        for face_lms in face_landmarks_list:
            lms = [[lm.x, lm.y, lm.z] for lm in face_lms]
            np_lms.append(lms)
        return np.array(np_lms).astype(np.float32)

    @staticmethod
    def mptrans_to_nptrans(results):
        trans_list = results.facial_transformation_matrixes
        return np.array(trans_list).astype(np.float32)

    def _run_one(self, imp):
        detection_result = self.detect_from_imp(imp)
        bs_np = self.mpbs_to_npbs(detection_result)
        lmk_np = self.mplmk_to_nplmk(detection_result)
        trans_np = self.mptrans_to_nptrans(detection_result)
        return lmk_np, bs_np, trans_np

    def _run(self, imp_list):
        lmk_list = []
        bs_list = []
        trans_list = []
        for imp in imp_list:
            lmk_np, bs_np, trans_np = self._run_one(imp)
            lmk_list.append(lmk_np)
            bs_list.append(bs_np)
            trans_list.append(trans_np)
        lmk_list = np.concatenate(lmk_list, 0)
        bs_list = np.concatenate(bs_list, 0)
        trans_list = np.concatenate(trans_list, 0)
        return lmk_list, bs_list, trans_list
    

@dataclass
class EyeIdxMP:
    LO = [33]
    LI = [133]
    LD = [7, 163, 144, 145, 153, 154, 155]  # O -> I
    LU = [246, 161, 160, 159, 158, 157, 173]  # O -> I
    RO = [263]
    RI = [362]
    RD = [249, 390, 373, 374, 380, 381, 382]  # O -> I
    RU = [466, 388, 387, 386, 385, 384, 398]  # O -> I

    LW = [33, 133]    # oi
    LH0 = [145, 159]
    LH1 = [144, 160]
    LH2 = [153, 158]

    RW = [263, 362]   # oi
    RH0 = [374, 386]
    RH1 = [373, 387]
    RH2 = [380, 385]

    LB = [468]  # eye ball
    RB = [473]


class EyeAttrUtilsByMP:
    def __init__(self, lmks_mp):
        self.IDX = EyeIdxMP()
        self.lmks = lmks_mp    # [n, 478, 3]

        self.L_width = self._dist_idx(*self.IDX.LW)   # [n]
        self.R_width = self._dist_idx(*self.IDX.RW)

        self.L_h0 = self._dist_idx(*self.IDX.LH0)
        self.L_h1 = self._dist_idx(*self.IDX.LH1)
        self.L_h2 = self._dist_idx(*self.IDX.LH2)

        self.R_h0 = self._dist_idx(*self.IDX.RH0)
        self.R_h1 = self._dist_idx(*self.IDX.RH1)
        self.R_h2 = self._dist_idx(*self.IDX.RH2)

        self.L_open =  (self.L_h0 + self.L_h1 + self.L_h2) / (self.L_width + 1e-8)   # [n]
        self.R_open =  (self.R_h0 + self.R_h1 + self.R_h2) / (self.R_width + 1e-8)

        self.L_center = self._center_idx(*self.IDX.LW)    # [n, 3/2]
        self.R_center = self._center_idx(*self.IDX.RW)

        self.L_ball = self.lmks[:, self.IDX.LB[0]]   # [n, 3/2]
        self.R_ball = self.lmks[:, self.IDX.RB[0]]

        self.L_ball_direc = (self.L_ball - self.L_center) / (self.L_width[:, None] + 1e-8)   # [n, 3/2]
        self.R_ball_direc = (self.R_ball - self.R_center) / (self.R_width[:, None] + 1e-8)

        self.L_eye_direc = self._direc_idx(*self.IDX.LW)  # I->O
        self.R_eye_direc = self._direc_idx(*self.IDX.RW)

        self.L_ball_move_dist = self._dist(self.L_ball, self.L_center)
        self.R_ball_move_dist = self._dist(self.R_ball, self.R_center)

        self.L_ball_move_direc = self._direc(self.L_ball, self.L_center) - self.L_eye_direc
        self.R_ball_move_direc = self._direc(self.R_ball, self.R_center) - self.R_eye_direc

        self.L_ball_move = self.L_ball_move_direc * self.L_ball_move_dist[:, None]
        self.R_ball_move = self.R_ball_move_direc * self.R_ball_move_dist[:, None]

    def LR_open(self):
        LR_open = np.stack([self.L_open, self.R_open], -1)    # [n, 2]
        return LR_open
    
    def LR_ball_direc(self):
        LR_ball_direc = np.stack([self.L_ball_direc, self.R_ball_direc], -1)    # [n, 3, 2]
        return LR_ball_direc
    
    def LR_ball_move(self):
        LR_ball_move = np.stack([self.L_ball_move, self.R_ball_move], -1)
        return LR_ball_move

    @staticmethod
    def _dist(p1, p2):
        # p1/p2: [n, 3/2]
        return (((p1 - p2) ** 2).sum(-1)) ** 0.5    # [n]
    
    @staticmethod
    def _center(p1, p2):
        return (p1 + p2) * 0.5   # [n, 3/2]
    
    def _direc(self, p1, p2):
        # p1 - p2, (2->1)
        return (p1 - p2) / (self._dist(p1, p2)[:, None] + 1e-8)
    
    def _dist_idx(self, idx1, idx2):
        p1 = self.lmks[:, idx1]
        p2 = self.lmks[:, idx2]
        d = self._dist(p1, p2)
        return d
    
    def _center_idx(self, idx1, idx2):
        p1 = self.lmks[:, idx1]
        p2 = self.lmks[:, idx2]
        c = self._center(p1, p2)
        return c
    
    def _direc_idx(self, idx1, idx2):
        p1 = self.lmks[:, idx1]
        p2 = self.lmks[:, idx2]
        dir = self._direc(p1, p2)
        return dir

def det_mp_lmks_for_video(video, MP: MediaPipeUtils, npy='', flip=False):
    lmk_list = []
    cap = cv2.VideoCapture(video)
    while True:
        flag, frame = cap.read()
        if not flag:
            break
        if flip:
            frame = frame[:, ::-1].copy()
        im_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)            
        det = MP.detect_from_npimage(im_rbg)
        lmk = MP.mplmk_to_nplmk(det)   # [1, 478, 3]
        lmk_list.append(lmk)
    cap.release()

    lmks = np.concatenate(lmk_list, 0)    # [n, 478, 3]
    if npy:
        os.makedirs(os.path.dirname(npy), exist_ok=True)
        np.save(npy, lmks)
    return lmks


def lmks_to_eye_attr(lmks, open_npy='', ball_npy=''):
    attr = EyeAttrUtilsByMP(lmks)

    lr_open = attr.LR_open()
    # lr_ball = attr.LR_ball_direc()
    lr_ball = attr.LR_ball_move()


    if open_npy:
        os.makedirs(os.path.dirname(open_npy), exist_ok=True)
        np.save(open_npy, lr_open)
    if ball_npy:
        os.makedirs(os.path.dirname(ball_npy), exist_ok=True)
        np.save(ball_npy, lr_ball)

    return lr_open, lr_ball


def flip_path(p):
    items = p.split('/')
    items[-2] = items[-2] + '_flip'
    p = '/'.join(items)
    return p


def flip_eye_open_ball(eye_open_npy_list, eye_ball_npy_list):

    def _flip_eye_open(eye_open_npy):
        if not os.path.isfile(eye_ball_npy):
            return
        flip_eye_open_npy = flip_path(eye_open_npy)
        eye_open = np.load(eye_open_npy)
        eye_open_flip = eye_open[:, ::-1]
        os.makedirs(os.path.dirname(flip_eye_open_npy), exist_ok=True)
        np.save(flip_eye_open_npy, eye_open_flip)

    def _flip_eye_ball(eye_ball_npy):
        if not os.path.isfile(eye_ball_npy):
            return
        flip_eye_ball_npy = flip_path(eye_ball_npy)
        eye_ball = np.load(eye_ball_npy)   # [n, 3, 2]
        eye_ball_filp = eye_ball[:, :, ::-1]
        eye_ball_filp[:, 0] = -eye_ball_filp[:, 0]
        os.makedirs(os.path.dirname(flip_eye_ball_npy), exist_ok=True)
        np.save(flip_eye_ball_npy, eye_ball_filp)

    total = len(eye_open_npy_list)
    for i, (eye_open_npy, eye_ball_npy) in enumerate(
        zip(eye_open_npy_list, eye_ball_npy_list), start=1
    ):
        try:
            _flip_eye_open(eye_open_npy)
            _flip_eye_ball(eye_ball_npy)
        except:
            traceback.print_exc()
        log_batch_progress(i, total, "flip_eye_open_ball", unit="pairs")


def process_data_list(video_list, lmk_npy_list, eye_open_npy_list, eye_ball_npy_list, flip_lmk_flag=False, face_landmarker_task_path=''):
    MP = MediaPipeUtils(face_landmarker_task_path)
    
    total = len(video_list)
    for i, (video, lmk_npy, eye_open_npy, eye_ball_npy) in enumerate(
        zip(video_list, lmk_npy_list, eye_open_npy_list, eye_ball_npy_list), start=1
    ):
        if flip_lmk_flag:
            lmk_npy = flip_path(lmk_npy)
            eye_open_npy = flip_path(eye_open_npy)
            eye_ball_npy = flip_path(eye_ball_npy)
        try:
            if not os.path.isfile(lmk_npy):
                lmks = det_mp_lmks_for_video(video, MP, npy=lmk_npy, flip=flip_lmk_flag)
            else:
                lmks = np.load(lmk_npy)
            lmks_to_eye_attr(lmks, open_npy=eye_open_npy, ball_npy=eye_ball_npy)
        except:
            traceback.print_exc()
        log_batch_progress(i, total, "extract_eye_ratio_from_video")
        

@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data list json: {'video_list': video_list, 'MP_lmk_npy_list': lmk_npy_list, 'eye_open_npy_list': eye_open_npy_list, 'eye_ball_npy_list': eye_ball_npy_list}
    data_start: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0   # data start
    data_end: Annotated[int, tyro.conf.arg(aliases=["-e"])] = -1    # data end
    flip_flag: bool = False    # flip_flag
    flip_lmk_flag: bool = False    # flip lmk flag
    MP_face_landmarker_task_path: str = ""


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json

    data_info = load_json(opt.input_data_json)

    video_list = data_info['video_list']
    MP_lmk_npy_list = data_info['MP_lmk_npy_list']
    eye_open_npy_list = data_info['eye_open_npy_list']
    eye_ball_npy_list = data_info['eye_ball_npy_list']

    print(len(video_list))

    s = opt.data_start
    e = opt.data_end
    if e < 0:
        e = len(video_list)

    video_list = video_list[s:e]
    MP_lmk_npy_list = MP_lmk_npy_list[s:e]
    eye_open_npy_list = eye_open_npy_list[s:e]
    eye_ball_npy_list = eye_ball_npy_list[s:e]

    print(s, e, len(video_list))
    if opt.flip_flag and not opt.flip_lmk_flag:
        flip_eye_open_ball(eye_open_npy_list, eye_ball_npy_list)
    else:
        process_data_list(video_list, MP_lmk_npy_list, eye_open_npy_list, eye_ball_npy_list, flip_lmk_flag=opt.flip_lmk_flag, face_landmarker_task_path=opt.MP_face_landmarker_task_path)


if __name__ == '__main__':
    main()
