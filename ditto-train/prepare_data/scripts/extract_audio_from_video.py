import os
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import load_json, log_batch_progress


def extract_audio(path, out_path, sample_rate=16000, ffmpeg_bin='ffmpeg'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = f"{ffmpeg_bin} -loglevel error -y -i {path} -f wav -ar {sample_rate} {out_path}"
    os.system(cmd)


def process_data_list(video_list, wav_list, ffmpeg_bin='ffmpeg'):
    total = len(video_list)
    for i, (video, wav) in enumerate(zip(video_list, wav_list), start=1):
        try:
            if os.path.isfile(wav):
                log_batch_progress(i, total, "extract_audio_from_video")
                continue
            extract_audio(video, wav, ffmpeg_bin=ffmpeg_bin)
        except:
            traceback.print_exc()
        log_batch_progress(i, total, "extract_audio_from_video")


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data list json: {'video_list': video_list, 'wav_list': wav_list}
    ffmpeg_bin: str = "ffmpeg"  # ffmpeg_bin path


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json

    data_info = load_json(opt.input_data_json)

    video_list = data_info['video_list']
    wav_list = data_info['wav_list']

    process_data_list(video_list, wav_list, opt.ffmpeg_bin)



if __name__ == '__main__':
    main()
