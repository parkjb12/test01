#!/usr/bin/env bash
# LogicKor 답변 생성(vLLM). 학습이 아니므로 INFER_IMAGE(Dockerfile-infer) 컨테이너를 쓴다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/docker_env.sh"

MODEL_PATH="${1:-models/qwen3_8b_sft_high/merged}"
GPU_DEVICES="${GPU_DEVICES:-0}"
MODEL_LEN="${MODEL_LEN:-8192}"

export CUDA_VISIBLE_DEVICES="${GPU_DEVICES}"
require_docker
require_hf_token
trap docker_cleanup EXIT INT TERM

docker_run_infer generate \
  python logickor_eval/generator.py \
  --model "${MODEL_PATH}" \
  --gpu_devices "${GPU_DEVICES}" \
  --model_len "${MODEL_LEN}"
