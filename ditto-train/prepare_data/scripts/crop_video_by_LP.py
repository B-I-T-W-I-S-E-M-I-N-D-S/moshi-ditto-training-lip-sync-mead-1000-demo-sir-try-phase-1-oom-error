import os
import numpy as np
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback
import imageio
from tqdm import tqdm
import cv2

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import dump_pkl, load_pkl, load_json, log_batch_progress, pipeline_log_frame_every
from LivePortrait.src.live_portrait_pipeline_sj import get_cfgs
from LivePortrait.src.utils.cropper import Cropper, Trajectory, contiguous, log, parse_bbox_from_landmark, average_bbox_lst, crop_image


class VideoWriterByImageIO:
    def __init__(self, video_path, fps=25, **kwargs):
        video_format = kwargs.get('format', 'mp4')  # default is mp4 format
        codec = kwargs.get('vcodec', 'libx264')  # default is libx264 encoding
        quality = kwargs.get('quality')  # video quality
        pixelformat = kwargs.get('pixelformat', 'yuv420p')  # video pixel format
        macro_block_size = kwargs.get('macro_block_size', 2)
        ffmpeg_params = ['-crf', str(kwargs.get('crf', 18))]

        writer = imageio.get_writer(
            video_path, fps=fps, format=video_format,
            codec=codec, quality=quality, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat, macro_block_size=macro_block_size
        )
        self.writer = writer
    
    def add_frame(self, img, fmt='bgr'):
        if fmt == 'bgr':
            frame = img[..., ::-1]
        else:
            frame = img
        self.writer.append_data(frame)

    def close(self):
        self.writer.close()


def init_cropper(ditto_pytorch_path):
    _, crop_cfg = get_cfgs(ditto_pytorch_path=ditto_pytorch_path)
    cropper = Cropper(crop_cfg=crop_cfg)
    return cropper


def extract_audio(path, out_path, sample_rate=16000, ffmpeg_bin='ffmpeg'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = f"{ffmpeg_bin} -loglevel error -y -i {path} -f wav -ar {sample_rate} {out_path}"
    os.system(cmd)


def crop_v2(self: Cropper, file_path, save_path, with_audio=False):
    reader = imageio.get_reader(file_path, "ffmpeg")
    frame_count = int(reader.count_frames())

    if with_audio:
        tmp_video = save_path + '_tmp.mp4'
    else:
        tmp_video = save_path

    writer = VideoWriterByImageIO(tmp_video)
    trajectory = Trajectory()
    direction = "large-small"
    _fe = pipeline_log_frame_every()
    for idx, frame_rgb in enumerate(
        tqdm(reader, total=frame_count, miniters=_fe, mininterval=1.0)
    ):
        if idx == 0 or trajectory.start == -1:
            src_face = self.face_analysis_wrapper.get(
                contiguous(frame_rgb[..., ::-1]),
                flag_do_landmark_2d_106=True,
                direction=direction,
                max_face_num=0,
            )
            if len(src_face) == 0:
                log(f"No face detected in the frame #{idx}")
                continue
            elif len(src_face) > 1:
                log(f"More than one face detected in the source frame_{idx}, only pick one face by rule {direction}.")
            src_face = src_face[0]
            lmk = src_face.landmark_2d_106
            lmk = self.human_landmark_runner.run(frame_rgb, lmk)
            trajectory.start, trajectory.end = idx, idx
        else:
            # TODO: add IOU check for tracking
            lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
            trajectory.end = idx

        trajectory.lmk_lst.append(lmk)

        ret_dct = crop_image(
            frame_rgb,  # ndarray
            lmk,  # 106x2 or Nx2
            dsize=512,
            scale=2.3,
            vx_ratio=0,
            vy_ratio=-0.125,
            flag_do_rot=False,
        )
        img_crop = ret_dct['img_crop']

        writer.add_frame(img_crop, fmt='rgb')

    writer.close()

    if with_audio:
        tmp_wav = save_path + '.wav'
        extract_audio(file_path, tmp_wav)
        cmd = f'ffmpeg -loglevel error -y -i {tmp_video} -i {tmp_wav} -map 0:v -map 1:a -c:v copy -c:a aac {save_path}'
        os.system(cmd)

        try:
            os.remove(tmp_wav)
            os.remove(tmp_video)
        except Exception as e:
            print(f'remove error: {e}')
            print(tmp_wav)
            print(tmp_video)


def get_global_bbox(self: Cropper, file_path, rgb_list=None, **kwargs):
    if rgb_list:
        reader = rgb_list
        frame_count = len(rgb_list)
    else:
        reader = imageio.get_reader(file_path, "ffmpeg")
        frame_count = int(reader.count_frames())

    trajectory = Trajectory()
    direction = kwargs.get("direction", "large-small")
    _fe = pipeline_log_frame_every()
    for idx, frame_rgb in enumerate(
        tqdm(
            reader,
            total=frame_count,
            desc="Processing frames",
            miniters=_fe,
            mininterval=1.0,
        )
    ):
        if idx == 0 or trajectory.start == -1:
            src_face = self.face_analysis_wrapper.get(
                contiguous(frame_rgb[..., ::-1]),
                flag_do_landmark_2d_106=True,
                direction=direction,
            )
            if len(src_face) == 0:
                log(f"No face detected in the frame #{idx}")
                continue
            elif len(src_face) > 1:
                log(f"More than one face detected in the driving frame_{idx}, only pick one face by rule {direction}.")
            src_face = src_face[0]
            lmk = src_face.landmark_2d_106
            lmk = self.human_landmark_runner.run(frame_rgb, lmk)
            trajectory.start, trajectory.end = idx, idx
        else:
            lmk = self.human_landmark_runner.run(frame_rgb, trajectory.lmk_lst[-1])
            trajectory.end = idx

        trajectory.lmk_lst.append(lmk)
        ret_bbox = parse_bbox_from_landmark(
            lmk,
            scale=self.crop_cfg.scale_crop_driving_video,
            vx_ratio_crop_driving_video=self.crop_cfg.vx_ratio_crop_driving_video,
            vy_ratio=self.crop_cfg.vy_ratio_crop_driving_video,
        )["bbox"]
        bbox = [
            ret_bbox[0, 0],
            ret_bbox[0, 1],
            ret_bbox[2, 0],
            ret_bbox[2, 1],
        ]  # 4,
        trajectory.bbox_lst.append(bbox)  # bbox

    global_bbox = average_bbox_lst(trajectory.bbox_lst)
    return global_bbox


def get_video_WH(video):
    cap = cv2.VideoCapture(video)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return [W, H]


def cvt_LP_bbox(bbox):
    x1, y1, x2, y2 = bbox
    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))
    half_w = int(round((x2 - x1) / 2))

    x1 = cx - half_w
    x2 = cx + half_w
    y1 = cy - half_w
    y2 = cy + half_w

    return [x1, y1, x2, y2]


def cvt_LP_bbox_nopad(bbox, HW=(float("inf"), float("inf"))):
    H, W = HW
    x1, y1, x2, y2 = bbox
    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2))
    half_w = int(round((x2 - x1) / 2))
    half_w = min(half_w, H // 2, W // 2)
    x1 = cx - half_w
    x2 = cx + half_w
    y1 = cy - half_w
    y2 = cy + half_w
    if x1 < 0:
        x2 = x2 - x1
        x1 = 0
    if y1 < 0:
        y2 = y2 - y1
        y1 = 0
    if x2 > W:
        x1 = x1 + W - x2
        x2 = W
    if y2 > H:
        y1 = y1 + H - y2
        y2 = H
    return [x1, y1, x2, y2]


def crop_pad_scale_video(video, crop_rect, src_h, src_w, dst_h, dst_w, res_video):
    x1, y1, x2, y2 = crop_rect
    dw = x2 - x1
    dh = y2 - y1
    px1, py1, px2, py2 = 0, 0 , 0, 0
    if x1 < 0:
        px1 = -x1
        x1 = 0
    if y1 < 0:
        py1 = -y1
        y1 = 0
    if x2 > src_w:
        px2 = x2 - src_w
        x2 = src_w
    if y2 > src_h:
        py2 = y2 - src_h
        y2 = src_h
    rw = x2 - x1
    rh = y2 - y1
    inp = video
    out = res_video
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = f'ffmpeg -loglevel error -y -i {inp} -vf "crop={rw}:{rh}:{x1}:{y1},pad={dw}:{dh}:{px1}:{py1},scale={dst_w}:{dst_h}" {out}'
    os.system(cmd)


def crop_one(video, res_video, cropper=None):
    bbox_pkl = video + '_LP_crop_bbox.pkl'
    if os.path.isfile(bbox_pkl):
        bbox = load_pkl(bbox_pkl)
    else:
        bbox = get_global_bbox(cropper, video)
        # dump_pkl(bbox, bbox_pkl)
    crop_rect = cvt_LP_bbox(bbox)

    src_w, src_h = get_video_WH(video)
    dst_w, dst_h = 512, 512

    crop_pad_scale_video(video, crop_rect, src_h, src_w, dst_h, dst_w, res_video)


def crop_one_v2(video, res_video, cropper=None):
    os.makedirs(os.path.dirname(res_video), exist_ok=True)
    crop_v2(cropper, video, res_video, with_audio=True)


def process_data_list(ori_video_list, res_video_list, ditto_pytorch_path):
    cropper = init_cropper(ditto_pytorch_path)
    total = len(ori_video_list)
    for i, (video, res_video) in enumerate(zip(ori_video_list, res_video_list), start=1):
        try:
            crop_one(video, res_video, cropper)
        except:
            traceback.print_exc()
        log_batch_progress(i, total, "crop_video_by_LP")


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data list json: {'fps25_video_list': fps25_video_list, 'video_list': video_list}
    ditto_pytorch_path: str = ""


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json

    data_info = load_json(opt.input_data_json)
    if 'fps25_video_list' not in data_info:
        print('skip')
    fps25_video_list = data_info['fps25_video_list']
    video_list = data_info['video_list']
    process_data_list(fps25_video_list, video_list, ditto_pytorch_path=opt.ditto_pytorch_path)


if __name__ == '__main__':
    main()
