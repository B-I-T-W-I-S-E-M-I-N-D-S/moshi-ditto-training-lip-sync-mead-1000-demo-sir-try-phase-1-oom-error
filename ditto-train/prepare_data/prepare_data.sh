set -euo pipefail
# set -x

conda activate ditto_train

SECONDS=0

DITTO_ROOT_DIR="$(dirname "$(dirname "$(readlink -f "$0")")")"

DITTO_PYTORCH_PATH="${DITTO_ROOT_DIR}/checkpoints/ditto_pytorch"
HUBERT_ONNX="${DITTO_PYTORCH_PATH}/aux_models/hubert_streaming_fix_kv.onnx"
MP_FACE_LMK_TASK="${DITTO_PYTORCH_PATH}/aux_models/face_landmarker.task"


cd "${DITTO_ROOT_DIR}/prepare_data"


data_info_json="$1"
data_list_json="$2"
data_preload_pkl="$3"


# check ckpt
python scripts/check_ckpt_path.py --ditto_pytorch_path ${DITTO_PYTORCH_PATH}


###############################################
# process data

### VIDEO ###
# crop video: 'fps25_video_list' -> 'video_list'
python scripts/crop_video_by_LP.py -i "${data_info_json}" --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"


### AUDIO ###
# extract audio: 'video_list' -> 'wav_list'
python scripts/extract_audio_from_video.py -i "${data_info_json}"


### Feature ###
# audio feat: 'wav_list' -> 'hubert_aud_npy_list'
python scripts/extract_audio_feat_by_Hubert.py -i "${data_info_json}" --Hubert_onnx "${HUBERT_ONNX}"

# motion feat: 'video_list' -> {'LP_pkl_list', 'LP_npy_list'} (_flip)
python scripts/extract_motion_feat_by_LP.py -i "${data_info_json}" --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"
python scripts/extract_motion_feat_by_LP.py -i "${data_info_json}" --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" --flip_flag

# eye feat: 'video_list' -> {'MP_lmk_npy_list', 'eye_open_npy_list', 'eye_ball_npy_list'} (_flip)
python scripts/extract_eye_ratio_from_video.py -i "${data_info_json}" --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}"
python scripts/extract_eye_ratio_from_video.py -i "${data_info_json}" --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" --flip_lmk_flag

# emo feat: 'video_list' -> 'emo_npy_list'
python scripts/extract_emo_feat_from_video.py -i "${data_info_json}"


###############################################
# get data_list_json for train
python scripts/gather_data_list_json_for_train.py \
    -i "${data_info_json}" \
    -o "${data_list_json}" \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --with_flip


# get preload data_pkl ([option] for faster training speed)
python scripts/preload_train_data_to_pkl.py \
    --data_list_json "${data_list_json}" \
    --data_preload_pkl "${data_preload_pkl}" \
    --use_sc \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --motion_feat_dim 265 \


cd "${DITTO_ROOT_DIR}"

###############################################


echo "[prepare_data]"

echo "data_list_json: ${data_list_json}"
echo "data_preload_pkl: ${data_preload_pkl}"

echo "DONE"


echo "Elapsed time: $SECONDS seconds"
