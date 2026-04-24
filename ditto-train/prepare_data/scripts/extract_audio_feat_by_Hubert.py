import os
from dataclasses import dataclass
from typing_extensions import Annotated
import tyro
import traceback

import sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CUR_DIR))

from utils.utils import load_json, log_batch_progress
from Wav2Feat.hubert.hubert_streaming_onnx import Wav2FeatHubert


def init_W2F(Hubert_onnx):
    w2f = Wav2FeatHubert(modelpath=Hubert_onnx)
    return w2f


def process_data_list(wav_list, feat_npy_list, Hubert_onnx):
    W2F = init_W2F(Hubert_onnx)
    total = len(wav_list)
    for i, (wav, npy) in enumerate(zip(wav_list, feat_npy_list), start=1):
        try:
            if os.path.isfile(npy):
                log_batch_progress(i, total, "extract_audio_feat_by_Hubert", unit="wavs")
                continue
            os.makedirs(os.path.dirname(npy), exist_ok=True)
            W2F(wav, npy)
        except:
            traceback.print_exc()
        log_batch_progress(i, total, "extract_audio_feat_by_Hubert", unit="wavs")


@dataclass
class Options:
    input_data_json: Annotated[str, tyro.conf.arg(aliases=["-i"])] = ""   # data list json: {'wav_list': wav_list, 'hubert_aud_npy_list': npy_list}
    Hubert_onnx: str = ""


def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)
    assert opt.input_data_json

    data_info = load_json(opt.input_data_json)
    wav_list = data_info['wav_list']
    feat_npy_list = data_info['hubert_aud_npy_list']
    process_data_list(wav_list, feat_npy_list, opt.Hubert_onnx)


if __name__ == '__main__':
    main()