import os
from dataclasses import dataclass
import tyro


def check_ckpt_path(ditto_pytorch_path):
    """
    ditto_pytorch
    ├── aux_models
    │   ├── 2d106det.onnx
    │   ├── det_10g.onnx
    │   ├── face_landmarker.task
    │   ├── hubert_streaming_fix_kv.onnx
    │   └── landmark203.onnx
    └── models
        ├── appearance_extractor.pth
        ├── decoder.pth
        ├── motion_extractor.pth
        ├── stitch_network.pth
        ├── warp_network.pth
        └── ...
    """

    check_models_names = [
        'appearance_extractor.pth',
        'decoder.pth',
        'motion_extractor.pth',
        'stitch_network.pth',
        'warp_network.pth'
    ]

    check_aux_models_names = [
        '2d106det.onnx',
        'det_10g.onnx',
        'face_landmarker.task',
        'hubert_streaming_fix_kv.onnx',
        'landmark203.onnx',
    ]

    assert all([os.path.isfile(f"{ditto_pytorch_path}/models/{name}") for name in check_models_names]), 'check ditto_pytorch ckpts error'
    assert all([os.path.isfile(f"{ditto_pytorch_path}/aux_models/{name}") for name in check_aux_models_names]), 'check ditto_pytorch ckpts error'
        
    

def check_softlink_for_insightface(ditto_pytorch_path):
    """
    insightface
    └── models
        └── buffalo_l
            ├── 2d106det.onnx
            └── det_10g.onnx
    """

    names = ['2d106det', 'det_10g']
    for name in names:
        target = f"{ditto_pytorch_path}/aux_models/insightface/models/buffalo_l/{name}.onnx"
        source = f"{ditto_pytorch_path}/aux_models/{name}.onnx"
        if os.path.exists(target):
            continue
        if not os.path.exists(source):
            raise FileNotFoundError(source)
        
        target_dir = os.path.dirname(target)
        os.makedirs(target_dir, exist_ok=True)

        try:
            rel_source = os.path.relpath(source, target_dir)
            os.symlink(rel_source, target)
            print(f"symlink: {target} -> {rel_source}")
        except OSError as e:
            print(f"symlink error: {e}")
            raise



@dataclass
class Options:
    ditto_pytorch_path: str = ""



def main():
    tyro.extras.set_accent_color("bright_cyan")
    opt: Options = tyro.cli(Options)

    ditto_pytorch_path = opt.ditto_pytorch_path
    assert ditto_pytorch_path

    check_ckpt_path(ditto_pytorch_path)
    check_softlink_for_insightface(ditto_pytorch_path)


if __name__ == "__main__":
    main()
    
