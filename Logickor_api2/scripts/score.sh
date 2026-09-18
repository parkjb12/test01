#!/usr/bin/env bash
# 채점 결과 집계. 학습이 아니므로 INFER_IMAGE(Dockerfile-infer) 컨테이너를 쓴다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
source "${REPO_ROOT}/scripts/docker_env.sh"

EVALUATED_GLOB="${1:-evaluated/runs/qwen3_8b_sft_high/merged/*.jsonl}"

require_docker
trap docker_cleanup EXIT INT TERM

# glob 은 셸이 아니라 score.py 가 해석한다(따옴표 그대로 전달).
docker_run_infer score \
  python logickor_eval/score.py -p "${EVALUATED_GLOB}"
