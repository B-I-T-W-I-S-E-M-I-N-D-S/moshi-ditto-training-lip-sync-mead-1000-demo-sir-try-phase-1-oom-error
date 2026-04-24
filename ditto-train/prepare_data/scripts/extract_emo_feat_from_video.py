import os
import numpy as np
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback
import math
import cv2

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import load_json, log_batch_progress

# HSEmotion 8-D order after reorder in _run_one_image: index 4 = Neutral
NEUTRAL_EMO_VEC = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32)


class EmoRec:
    def __init__(self, hse_name="enet_b2_8", device='cuda'):
        from facenet_pytorch import MTCNN
        from hsemotion.facial_emotions import HSEmotionRecognizer
        self.mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
        self.fer=HSEmotionRecognizer(model_name=hse_name, device=device)

    def _detect_face(self, frame):
        bounding_boxes, probs = self.mtcnn.detect(frame, landmarks=False)
        bounding_boxes=bounding_boxes[probs>0.9]
        return bounding_boxes

    def _run_one_image(self, img_rgb_uint8):
        bounding_boxes = self._detect_face(img_rgb_uint8)
        bbox = bounding_boxes[0]
        box = bbox.astype(int)
        x1,y1,x2,y2 = box[0:4]
        face_img = img_rgb_uint8[y1:y2,x1:x2,:]
        emotion, scores = self.fer.predict_emotions(face_img, logits=False)

        # {0: 'Anger', 1: 'Contempt', 2: 'Disgust', 3: 'Fear', 4: 'Happy', 5: 'Neutral', 6: 'Sad', 7: 'Surprise'}
        ret_scores = scores[[0, 2, 3, 4, 5, 6, 7, 1]]
        # ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise', 'Contempt']
        return ret_scores  # (8,)
    
    def run_one_video(self, video_path, run_fps=5, ori_fps=25):
        step = max(ori_fps // run_fps, 1)

        res_dict = {}
        cap = cv2.VideoCapture(video_path)
        fid = 0
        while True:
            flag, frame = cap.read()
            if not flag:
                break
            if fid % step == 0:
                im_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    emo_scores = self._run_one_image(im_rbg)
                    res_dict[fid] = emo_scores
                except:
                    pass

            fid += 1

        cap.release()

        return res_dict
    

def interpolate_missing_frames(total_frames, frame_dict):
    existing_indices = sorted(frame_dict.keys())
    result = {}

    left_known = np.full(total_frames, -1, dtype=int)
    last = -1
    for idx in existing_indices:
        left_known[idx] = idx
    for i in range(total_frames):
        if left_known[i] != -1:
            last = left_known[i]
        else:
            left_known[i] = last

    right_known = np.full(total_frames, -1, dtype=int)
    next = -1
    for idx in existing_indices:
        right_known[idx] = idx
    for i in reversed(range(total_frames)):
        if right_known[i] != -1:
            next = right_known[i]
        else:
            right_known[i] = next

    for i in range(total_frames):
        if i in frame_dict:
            result[i] = frame_dict[i]
        else:
            left = left_known[i]
            right = right_known[i]
            if left == -1 and right == -1:
                raise ValueError("")
            elif left == -1:
                result[i] = frame_dict[right].copy()
            elif right == -1:
                result[i] = frame_dict[left].copy()
            else:
                left_val = frame_dict[left]
                right_val = frame_dict[right]
                t = (i - left) / (right - left)
                result[i] = left_val * (1 - t) + right_val * t

    return result
    


def get_video_frame_num(video):
    cap = cv2.VideoCapture(video)
    frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_num


def process_one_video(video, emo_npy, ER: EmoRec, run_fps=5, ori_fps=25):
    """
    Always writes emo_npy when given. If MTCNN/HSEmotion sees too few faces, we save
    a neutral (T, 8) tensor so gather/train do not fail on missing files.
    """
    emo_frame_dict = ER.run_one_video(video, run_fps=run_fps, ori_fps=ori_fps)

    frame_num = get_video_frame_num(video)
    if frame_num <= 0:
        print(f"warn: zero frames for video, skip emo: {video}")
        return None

    sparse = len(emo_frame_dict) * ori_fps / run_fps < frame_num * 0.6

    if len(emo_frame_dict) == 0:
        print(f"warn: no face detections for emo, neutral fallback: {video}")
        emo_arr = np.tile(NEUTRAL_EMO_VEC, (frame_num, 1))
    else:
        try:
            all_emo_dict = interpolate_missing_frames(frame_num, emo_frame_dict)
            emo_arr = np.stack([all_emo_dict[i] for i in range(frame_num)], 0)  # [n, 8]
        except (ValueError, KeyError):
            print(f"warn: emo interpolation failed, neutral fallback: {video}")
            emo_arr = np.tile(NEUTRAL_EMO_VEC, (frame_num, 1))
        if sparse:
            print(
                f"warn: sparse emo ({len(emo_frame_dict)} sampled frames / {frame_num} total); "
                f"filled via interpolation or neutral: {video}"
            )

    if emo_npy:
        os.makedirs(os.path.dirname(emo_npy), exist_ok=True)
        np.save(emo_npy, emo_arr)

    return emo_arr


def process_data_list(video_list, emo_npy_list):
    ER = EmoRec()

    total = len(video_list)
    for i, (video, emo_npy) in enumerate(zip(video_list, emo_npy_list), start=1):
        try:
            if not os.path.isfile(emo_npy):
                process_one_video(video, emo_npy, ER=ER)
        except:
            traceback.print_exc()
        log_batch_progress(i, total, "extract_emo_feat_from_video")


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data list json: {'video_list': video_list, 'emo_npy_list': emo_npy_list}


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json

    data_info = load_json(opt.input_data_json)

    video_list = data_info['video_list']
    emo_npy_list = data_info['emo_npy_list']

    process_data_list(video_list, emo_npy_list)

    
if __name__ == '__main__':
    main()

    # pip install facenet-pytorch
    # pip install hsemotion
    # pip install timm==0.9.0