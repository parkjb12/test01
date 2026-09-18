#!/usr/bin/env bash
# LoRA SFT 학습만 단독 실행한다. 학습은 TRAIN_IMAGE(Dockerfile-train) 컨테이너에서 돈다.
#   export HF_TOKEN=hf_...
#   bash scripts/train.sh [config] [output-dir]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/docker_env.sh"

CONFIG_PATH="${1:-configs/train_qwen3_8b_sft.yaml}"
OUTPUT_DIR="${2:-runs/qwen3_8b_sft_high}"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
# 짧은 확인용 실행: TRAIN_FRACTION=0.1 처럼 데이터 일부만 사용.
TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"

export CUDA_VISIBLE_DEVICES="${GPU}"
require_docker
require_hf_token
trap docker_cleanup EXIT INT TERM

docker_run_train train \
  python train/train_lora.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --train-fraction "${TRAIN_FRACTION}"
