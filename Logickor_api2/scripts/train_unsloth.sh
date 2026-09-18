#!/usr/bin/env bash
# unsloth 기반 LoRA 학습. 학습이므로 TRAIN_IMAGE(Dockerfile-train) 컨테이너에서 돈다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/docker_env.sh"

CONFIG_PATH="${1:-configs/train_gemma4_e4b_unsloth_sft.yaml}"
OUTPUT_DIR="${2:-runs/gemma4_e4b_unsloth_sft}"
SEED="${SEED:-42}"
SAVE_MERGED="${SAVE_MERGED:-false}"
GPU="${GPU:-0}"

args=(
  --config "${CONFIG_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --seed "${SEED}"
)

if [[ "${SAVE_MERGED}" == "true" ]]; then
  args+=(--save-merged)
fi

export CUDA_VISIBLE_DEVICES="${GPU}"
require_docker
require_hf_token
trap docker_cleanup EXIT INT TERM

docker_run_train train-unsloth python train/train_unsloth_lora.py "${args[@]}"
