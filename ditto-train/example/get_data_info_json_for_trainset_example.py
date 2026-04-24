import os
import json

def dump_json(obj, p):
    with open(p, "w") as f:
        json.dump(obj, f)

def video_dir_to_data_info(fps25_mp4_video_dir, save_dir, data_info_json):
    fps25_video_list = []
    video_list = []
    wav_list = []
    hubert_aud_npy_list = []
    LP_pkl_list = []
    LP_npy_list = []
    MP_lmk_npy_list = []
    eye_open_npy_list = []
    eye_ball_npy_list = []
    emo_npy_list = []

    video_name_list = sorted(os.listdir(fps25_mp4_video_dir))
    count = 0
    for video_name in video_name_list:
        if not video_name.endswith('.mp4'):
            continue
        name = video_name.rsplit('.', 1)[0]
        fps25_video = os.path.join(fps25_mp4_video_dir, video_name)
        video         = f'{save_dir}/video/{name}.mp4'
        wav           = f'{save_dir}/wav/{name}.wav'
        hubert_aud_npy = f'{save_dir}/hubert_aud_npy/{name}.npy'
        LP_pkl        = f'{save_dir}/LP_pkl/{name}.pkl'
        LP_npy        = f'{save_dir}/LP_npy/{name}.npy'
        MP_lmk_npy    = f'{save_dir}/MP_lmk_npy/{name}.npy'
        eye_open_npy  = f'{save_dir}/eye_open_npy/{name}.npy'
        eye_ball_npy  = f'{save_dir}/eye_ball_npy/{name}.npy'
        emo_npy       = f'{save_dir}/emo_npy/{name}.npy'

        fps25_video_list.append(fps25_video)
        video_list.append(video)
        wav_list.append(wav)
        hubert_aud_npy_list.append(hubert_aud_npy)
        LP_pkl_list.append(LP_pkl)
        LP_npy_list.append(LP_npy)
        MP_lmk_npy_list.append(MP_lmk_npy)
        eye_open_npy_list.append(eye_open_npy)
        eye_ball_npy_list.append(eye_ball_npy)
        emo_npy_list.append(emo_npy)
        count += 1

    data_info = {
        'fps25_video_list':    fps25_video_list,
        'video_list':          video_list,
        'wav_list':            wav_list,
        'hubert_aud_npy_list': hubert_aud_npy_list,
        'LP_pkl_list':         LP_pkl_list,
        'LP_npy_list':         LP_npy_list,
        'MP_lmk_npy_list':    MP_lmk_npy_list,
        'eye_open_npy_list':   eye_open_npy_list,
        'eye_ball_npy_list':   eye_ball_npy_list,
        'emo_npy_list':        emo_npy_list,
    }

    os.makedirs(os.path.dirname(data_info_json), exist_ok=True)
    dump_json(data_info, data_info_json)
    return count

if __name__ == '__main__':
    # ── Paths for your HDTF dataset ──────────────────────────────────────────
    fps25_mp4_video_dir = '/workspace/HDTF/video'       # folder with all .mp4 files
    save_dir            = '/workspace/HDTF'       # processed features go here
    data_info_json      = '/workspace/HDTF/data_info.json'
    # ─────────────────────────────────────────────────────────────────────────

    count = video_dir_to_data_info(fps25_mp4_video_dir, save_dir, data_info_json)
    print(f'Saved: {data_info_json}')
    print(f'Total videos found: {count}')