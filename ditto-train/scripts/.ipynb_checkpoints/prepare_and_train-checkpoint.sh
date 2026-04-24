#!/usr/bin/env bash
# =============================================================================
# prepare_and_train.sh — Complete Bridge-Ditto Training Pipeline
# =============================================================================
# End-to-end script for RunPod:
#   1. Prepare HDTF data (video → motion/eye/emo features) using Ditto's pipeline
#   2. Extract bridge audio features (replaces HuBERT extraction)
#   3. Gather training data list
#   4. Launch distributed training
#
# Prerequisites:
#   - Run scripts/setup_environment.sh first
#   - HDTF dataset available at $HDTF_ROOT
#   - Ditto checkpoints at ditto-train/checkpoints/ditto_pytorch/
#
# Usage:
#   bash scripts/prepare_and_train.sh /workspace/HDTF
#
# Or with custom settings:
#   HDTF_ROOT=/data/hdtf NUM_GPUS=4 EPOCHS=500 bash scripts/prepare_and_train.sh
#
# Do not prefix env vars with "!" in bash (that triggers history expansion).
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
HDTF_ROOT="${1:-${HDTF_ROOT:-/workspace/HDTF}}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-1000}"
LR="${LR:-1e-4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-bridge_ditto_hdtf_v1}"
AUDIO_FEAT_DIM="${AUDIO_FEAT_DIM:-1103}"
MOTION_FEAT_DIM="${MOTION_FEAT_DIM:-265}"
# Clip length — must stay in sync with gather --min-frames (seq_frames + 1)
SEQ_FRAMES="${SEQ_FRAMES:-80}"
MIN_FRAMES=$((SEQ_FRAMES + 1))

# Lip-sync loss (set USE_LIP_SYNC=1 to enable)
USE_LIP_SYNC="${USE_LIP_SYNC:-1}"
LIP_SYNC_LAMBDA1="${LIP_SYNC_LAMBDA1:-0.05}"
LIP_SYNC_LAMBDA2="${LIP_SYNC_LAMBDA2:-0.02}"
LIP_SYNC_EVERY_N="${LIP_SYNC_EVERY_N:-5}"
LIP_SYNC_BATCH="${LIP_SYNC_BATCH:-2}"
LIP_SYNC_WARMUP="${LIP_SYNC_WARMUP:-30}"
LIP_SYNC_DEBUG="${LIP_SYNC_DEBUG:-1}"
LIP_SYNC_USE_STABLE="${LIP_SYNC_USE_STABLE:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DITTO_TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DITTO_TRAIN_DIR")"

DITTO_PYTORCH_PATH="${DITTO_TRAIN_DIR}/checkpoints/ditto_pytorch"
BRIDGE_CKPT="${PROJECT_ROOT}/checkpoints/bridge_best.pt"
BRIDGE_CONFIG="${PROJECT_ROOT}/bridge_module/config.yaml"
SYNCNET_CKPT="${DITTO_TRAIN_DIR}/checkpoints/lipsync_expert.pth"

# Output paths
DATA_INFO_JSON="${HDTF_ROOT}/data_info.json"
DATA_LIST_JSON="${HDTF_ROOT}/bridge_data_list_train.json"
DATA_PRELOAD_PKL="${HDTF_ROOT}/bridge_preload.pkl"

BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
RED="\033[1;31m"
RESET="\033[0m"

log() { echo -e "${CYAN}[pipeline]${RESET} $*"; }
ok()  { echo -e "${GREEN}[pipeline]${RESET} ✅  $*"; }
err() { echo -e "${RED}[pipeline]${RESET} ❌  $*"; exit 1; }

# Crop + motion / eye / emotion extraction (writes paths listed in data_info.json)
run_phase1_ditto_preprocess() {
    log "Running Ditto's video processing pipeline..."
    log "(This extracts motion, eye, and emotion features from videos)"

    HUBERT_ONNX="${DITTO_PYTORCH_PATH}/aux_models/hubert_streaming_fix_kv.onnx"
    MP_FACE_LMK_TASK="${DITTO_PYTORCH_PATH}/aux_models/face_landmarker.task"

    cd "${DITTO_TRAIN_DIR}/prepare_data"

    python scripts/check_ckpt_path.py --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"

    python scripts/crop_video_by_LP.py -i "${DATA_INFO_JSON}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"

    python scripts/extract_audio_from_video.py -i "${DATA_INFO_JSON}"

    python scripts/extract_motion_feat_by_LP.py -i "${DATA_INFO_JSON}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}"
    python scripts/extract_motion_feat_by_LP.py -i "${DATA_INFO_JSON}" \
        --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" --flip_flag

    python scripts/extract_eye_ratio_from_video.py -i "${DATA_INFO_JSON}" \
        --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}"
    python scripts/extract_eye_ratio_from_video.py -i "${DATA_INFO_JSON}" \
        --MP_face_landmarker_task_path "${MP_FACE_LMK_TASK}" --flip_lmk_flag

    python scripts/extract_emo_feat_from_video.py -i "${DATA_INFO_JSON}"

    cd "${DITTO_TRAIN_DIR}"

    ok "Video processing complete."
}

SECONDS=0

echo -e "${BOLD}"
echo "═══════════════════════════════════════════════════════════════"
echo "   Bridge-Ditto Training Pipeline"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "   HDTF Root    : $HDTF_ROOT"
echo "   GPUs         : $NUM_GPUS"
echo "   Batch/GPU    : $BATCH_SIZE"
echo "   Epochs       : $EPOCHS"
echo "   Experiment   : $EXPERIMENT_NAME"
echo "   Audio dim    : $AUDIO_FEAT_DIM"
echo "   Motion dim   : $MOTION_FEAT_DIM"
echo "   Seq frames   : $SEQ_FRAMES (min_frames for gather: $MIN_FRAMES)"
echo "   Lip-sync     : $([ "$USE_LIP_SYNC" -eq 1 ] && echo 'ENABLED (λ1='$LIP_SYNC_LAMBDA1', λ2='$LIP_SYNC_LAMBDA2')' || echo 'disabled')"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo -e "${RESET}"

# ── Validate paths ────────────────────────────────────────────────────────────
[ -d "$HDTF_ROOT" ] || err "HDTF dataset not found: $HDTF_ROOT"
[ -f "$BRIDGE_CKPT" ] || err "Bridge checkpoint not found: $BRIDGE_CKPT"
[ -f "$BRIDGE_CONFIG" ] || err "Bridge config not found: $BRIDGE_CONFIG"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Video Processing (reuse Ditto's prepare_data.sh)
# Extracts: motion features, eye features, emotion features
# ══════════════════════════════════════════════════════════════════════════════
log "PHASE 1: Video Processing (Ditto data preparation)..."

[ -f "$DATA_INFO_JSON" ] || err "Missing ${DATA_INFO_JSON}. Create it first (paths under HDTF_ROOT): ditto-train/example/get_data_info_json_for_trainset_example.py"

log "Checking whether motion / emotion / eye / wav artifacts exist (sample paths from data_info.json)..."

if python3 << EOF
import json, os, sys
with open("${DATA_INFO_JSON}") as f:
    d = json.load(f)
keys = [
    ("LP_npy_list", "motion LP_npy"),
    ("emo_npy_list", "emotion"),
    ("eye_open_npy_list", "eye_open"),
    ("eye_ball_npy_list", "eye_ball"),
    ("wav_list", "wav"),
]
for k, label in keys:
    if k not in d or not d[k]:
        print(f"[check] missing key or empty list: {k}", file=sys.stderr)
        sys.exit(1)
    p = d[k][0]
    if not os.path.isfile(p):
        print(f"[check] missing {label} file: {p}", file=sys.stderr)
        sys.exit(1)
print("[check] OK: sample paths resolve on disk.")
sys.exit(0)
EOF
then
    ok "Feature files already present — skipping Phase 1."
    ok "  (Delete outputs under ${HDTF_ROOT} or re-run with missing .npy to regenerate.)"
else
    log "Sample paths in data_info.json do not exist on disk yet (normal on a fresh volume)."
    log "Running Phase 1 to generate motion / eye / emotion features from videos..."
    run_phase1_ditto_preprocess
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Bridge Audio Feature Extraction (replaces HuBERT)
# Audio → Mimi → Bridge → .npy features (T, 1024) @ 25 Hz
# ══════════════════════════════════════════════════════════════════════════════
log "PHASE 2: Bridge Audio Feature Extraction..."
log "  (Replaces HuBERT ONNX extraction with Mimi → Bridge pipeline)"

cd "${DITTO_TRAIN_DIR}"

if [ "$NUM_GPUS" -gt 1 ]; then
    log "Multi-GPU extraction on $NUM_GPUS GPUs..."

    # Launch one process per GPU in parallel
    pids=()
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$gpu_id python preprocess_bridge_features.py \
            -i "${DATA_INFO_JSON}" \
            --bridge_ckpt "${BRIDGE_CKPT}" \
            --bridge_config "${BRIDGE_CONFIG}" \
            --device "cuda" \
            --num_gpus "${NUM_GPUS}" \
            --gpu_id "${gpu_id}" \
            --output_key "bridge_aud_npy_list" &
        pids+=($!)
        log "  GPU $gpu_id: PID ${pids[-1]}"
    done

    # Wait for all processes
    for pid in "${pids[@]}"; do
        wait "$pid" || err "Bridge extraction failed on one GPU (PID: $pid)"
    done
else
    python preprocess_bridge_features.py \
        -i "${DATA_INFO_JSON}" \
        --bridge_ckpt "${BRIDGE_CKPT}" \
        --bridge_config "${BRIDGE_CONFIG}" \
        --device "cuda" \
        --output_key "bridge_aud_npy_list"
fi

ok "Bridge feature extraction complete."

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2.5: Lip-Sync Feature Preprocessing (optional)
# Precomputes per-video: appearance features, source keypoints,
# SyncNet audio embeddings, GT similarity scores
# ══════════════════════════════════════════════════════════════════════════════
if [ "$USE_LIP_SYNC" -eq 1 ]; then
    log "PHASE 2.5: Lip-Sync Feature Preprocessing..."

    [ -f "$SYNCNET_CKPT" ] || err "SyncNet checkpoint not found: $SYNCNET_CKPT (download lipsync_expert.pth first)"

    cd "${DITTO_TRAIN_DIR}"

    if [ "$NUM_GPUS" -gt 1 ]; then
        pids=()
        for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
            CUDA_VISIBLE_DEVICES=$gpu_id python preprocess_lipsync_features.py \
                -i "${DATA_INFO_JSON}" \
                --syncnet_ckpt "${SYNCNET_CKPT}" \
                --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
                --device "cuda" \
                --num_gpus "${NUM_GPUS}" \
                --gpu_id "${gpu_id}" &
            pids+=($!)
            log "  GPU $gpu_id: PID ${pids[-1]}"
        done
        for pid in "${pids[@]}"; do
            wait "$pid" || err "Lip-sync preprocessing failed (PID: $pid)"
        done

        # Merge shard files into data_info.json
        log "Merging lip-sync shard files..."
        python -c "
import json, glob, os
info_path = '${DATA_INFO_JSON}'
with open(info_path) as f:
    data_info = json.load(f)
N = len(data_info.get('wav_list', []))
for key in ['lipsync_f_s_list','lipsync_x_s_list','lipsync_kp_canonical_list','lipsync_A_list','lipsync_sim_gt_list']:
    if key not in data_info:
        data_info[key] = [''] * N
shard_pattern = info_path.replace('.json', '_lipsync_shard_*.json')
for shard_path in sorted(glob.glob(shard_pattern)):
    with open(shard_path) as f:
        shard = json.load(f)
    for idx_str, paths in shard.items():
        idx = int(idx_str)
        data_info['lipsync_f_s_list'][idx] = paths.get('lipsync_f_s', '')
        data_info['lipsync_x_s_list'][idx] = paths.get('lipsync_x_s', '')
        data_info['lipsync_kp_canonical_list'][idx] = paths.get('lipsync_kp_canonical', '')
        data_info['lipsync_A_list'][idx] = paths.get('lipsync_A', '')
        data_info['lipsync_sim_gt_list'][idx] = paths.get('lipsync_sim_gt', '')
    os.remove(shard_path)
    print(f'  Merged + removed: {shard_path}')
with open(info_path, 'w') as f:
    json.dump(data_info, f, indent=2)
filled = sum(1 for x in data_info['lipsync_f_s_list'] if x)
print(f'  Merged {filled}/{N} lip-sync entries into {info_path}')
"
    else
        python preprocess_lipsync_features.py \
            -i "${DATA_INFO_JSON}" \
            --syncnet_ckpt "${SYNCNET_CKPT}" \
            --ditto_pytorch_path "${DITTO_PYTORCH_PATH}" \
            --device "cuda"
    fi

    ok "Lip-sync features preprocessed."
else
    log "PHASE 2.5: Lip-sync preprocessing SKIPPED (USE_LIP_SYNC=0)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Gather Training Data List
# Creates the data_list_json that Stage2Dataset reads
# ══════════════════════════════════════════════════════════════════════════════
log "PHASE 3: Gathering training data list..."

cd "${DITTO_TRAIN_DIR}/prepare_data"

python scripts/gather_data_list_json_for_train.py \
    -i "${DATA_INFO_JSON}" \
    -o "${DATA_LIST_JSON}" \
    --aud_feat_name "bridge_aud_npy_list" \
    --min-frames "${MIN_FRAMES}" \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --with_flip \
    $([ "$USE_LIP_SYNC" -eq 1 ] && echo "--use_lip_sync")

ok "Training data list created: ${DATA_LIST_JSON}"

# ── Optional: preload data into pickle for faster training ────────────────────
log "Creating preloaded data pickle (optional, speeds up training)..."

python scripts/preload_train_data_to_pkl.py \
    --data_list_json "${DATA_LIST_JSON}" \
    --data_preload_pkl "${DATA_PRELOAD_PKL}" \
    --seq_frames "${SEQ_FRAMES}" \
    --use_sc \
    --use_emo \
    --use_eye_open \
    --use_eye_ball \
    --use_last_frame \
    --motion_feat_dim ${MOTION_FEAT_DIM} \
    $([ "$USE_LIP_SYNC" -eq 1 ] && echo "--use_lip_sync_loss")

ok "Preloaded data pickle: ${DATA_PRELOAD_PKL}"

cd "${DITTO_TRAIN_DIR}"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Training
# Only LMDM diffusion model is trained, everything else is frozen/offline
# ══════════════════════════════════════════════════════════════════════════════
log "PHASE 4: Starting training..."
log "  Experiment: ${EXPERIMENT_NAME}"
log "  GPUs: ${NUM_GPUS}, Batch/GPU: ${BATCH_SIZE}, Epochs: ${EPOCHS}"

cd "${DITTO_TRAIN_DIR}"

if [ "$NUM_GPUS" -gt 1 ]; then
    log "Launching distributed training with accelerate..."

    accelerate launch \
        --num_processes "${NUM_GPUS}" \
        --mixed_precision no \
        train_bridge_ditto.py \
        --experiment_name "${EXPERIMENT_NAME}" \
        --data_list_json "${DATA_LIST_JSON}" \
        --data_preload \
        --data_preload_pkl "${DATA_PRELOAD_PKL}" \
        --audio_feat_dim "${AUDIO_FEAT_DIM}" \
        --motion_feat_dim "${MOTION_FEAT_DIM}" \
        --seq_frames "${SEQ_FRAMES}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --save_ckpt_freq 50 \
        --num_workers 4 \
        --use_accelerate \
        --use_emo \
        --use_eye_open \
        --use_eye_ball \
        --use_sc \
        --use_last_frame \
        $([ "$USE_LIP_SYNC" -eq 1 ] && echo "\
        --use_lip_sync_loss \
        --syncnet_checkpoint ${SYNCNET_CKPT} \
        --ditto_pytorch_path ${DITTO_PYTORCH_PATH} \
        --lip_sync_lambda1 ${LIP_SYNC_LAMBDA1} \
        --lip_sync_lambda2 ${LIP_SYNC_LAMBDA2} \
        --lip_sync_every_n_steps ${LIP_SYNC_EVERY_N} \
        --lip_sync_batch_size ${LIP_SYNC_BATCH} \
        --lip_sync_warmup_epochs ${LIP_SYNC_WARMUP} \
        $([ \"$LIP_SYNC_DEBUG\" -eq 1 ] && echo '--lip_sync_debug') \
        $([ \"$LIP_SYNC_USE_STABLE\" -eq 1 ] && echo '--lip_sync_use_stable')")
else
    python train_bridge_ditto.py \
        --experiment_name "${EXPERIMENT_NAME}" \
        --data_list_json "${DATA_LIST_JSON}" \
        --data_preload \
        --data_preload_pkl "${DATA_PRELOAD_PKL}" \
        --audio_feat_dim "${AUDIO_FEAT_DIM}" \
        --motion_feat_dim "${MOTION_FEAT_DIM}" \
        --seq_frames "${SEQ_FRAMES}" \
        --batch_size "${BATCH_SIZE}" \
        --lr "${LR}" \
        --epochs "${EPOCHS}" \
        --save_ckpt_freq 50 \
        --num_workers 4 \
        --use_emo \
        --use_eye_open \
        --use_eye_ball \
        --use_sc \
        --use_last_frame \
        $([ "$USE_LIP_SYNC" -eq 1 ] && echo "\
        --use_lip_sync_loss \
        --syncnet_checkpoint ${SYNCNET_CKPT} \
        --ditto_pytorch_path ${DITTO_PYTORCH_PATH} \
        --lip_sync_lambda1 ${LIP_SYNC_LAMBDA1} \
        --lip_sync_lambda2 ${LIP_SYNC_LAMBDA2} \
        --lip_sync_every_n_steps ${LIP_SYNC_EVERY_N} \
        --lip_sync_batch_size ${LIP_SYNC_BATCH} \
        --lip_sync_warmup_epochs ${LIP_SYNC_WARMUP} \
        $([ \"$LIP_SYNC_DEBUG\" -eq 1 ] && echo '--lip_sync_debug') \
        $([ \"$LIP_SYNC_USE_STABLE\" -eq 1 ] && echo '--lip_sync_use_stable')")
fi

echo ""
echo -e "${BOLD}${GREEN}"
echo "═══════════════════════════════════════════════════════════════"
echo "   ✅  Training Pipeline Complete!"
echo ""
echo "   Time elapsed: ${SECONDS}s"
echo ""
echo "   Checkpoints: ${DITTO_TRAIN_DIR}/experiments/s2/${EXPERIMENT_NAME}/ckpts/"
echo "   Loss log:    ${DITTO_TRAIN_DIR}/experiments/s2/${EXPERIMENT_NAME}/loss.log"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${RESET}"
