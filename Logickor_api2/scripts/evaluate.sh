#!/usr/bin/env bash
# 생성 결과 채점. 학습이 아니므로 INFER_IMAGE(Dockerfile-infer) 컨테이너를 쓴다.
#   bash scripts/evaluate.sh [generated-dir] [evaluator.py 추가 인자...]
# 예) 실패 항목만 다시 채점:
#   bash scripts/evaluate.sh generated/runs/<모델>/merged --rejudge-failed
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/docker_env.sh"

MODEL_OUTPUT_DIR="${1:-generated/models/qwen3_8b_sft_high/merged}"
shift || true
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4.1}"
THREADS="${THREADS:-30}"
GPU="${GPU:-0}"
MODEL_LEN="${JUDGE_MODEL_LEN:-8192}"

# OpenAI 심판 모델일 때만 키가 필요하다. 로컬 심판 모델(gemma/llama)은 GPU 로 돈다.
case "${JUDGE_MODEL}" in
  gpt-*|chatgpt-*|o1*|o3*|o4*)
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
      echo "OPENAI_API_KEY is required. Example: export OPENAI_API_KEY='...'" >&2
      exit 1
    fi
    ;;
esac
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"
require_docker
require_hf_token
trap docker_cleanup EXIT INT TERM

# API 키는 인자가 아니라 환경변수로만 넘긴다(ps 노출 방지).
docker_run_infer evaluate \
  python logickor_eval/evaluator.py \
  -o "${MODEL_OUTPUT_DIR}" \
  -j "${JUDGE_MODEL}" \
  -g "${GPU}" \
  -ml "${MODEL_LEN}" \
  -t "${THREADS}" \
  "$@"
