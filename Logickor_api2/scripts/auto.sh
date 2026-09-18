#!/usr/bin/env bash
# Run the full LogicKor pipeline in one shot:
#   0) clean previous outputs  1) train  2) generate  3) evaluate  4) score
#
# 모든 단계는 도커 컨테이너에서 실행된다.
#   Step 1 (학습)            -> TRAIN_IMAGE (Dockerfile-train)
#   Step 2~4 (생성/채점/집계) -> INFER_IMAGE (Dockerfile-infer)
# 호스트에는 파이썬 패키지가 없어도 되고, 도커와 NVIDIA 드라이버만 있으면 된다.
#
# 실행 전에 허깅페이스 토큰을 등록해야 한다(모델 다운로드용):
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
set -euo pipefail

# Always run from the repository root, whatever directory the script is called from.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 도커 실행 공통 설정(REPO_ROOT / INFER_IMAGE / TRAIN_IMAGE / docker_run_* ...).
source "${REPO_ROOT}/scripts/docker_env.sh"

CONFIG_PATH="${CONFIG_PATH:-configs/train_gemma4_e4b_sft.yaml}"
RUN_DIR="${RUN_DIR:-runs/gemma4_e4b_sft_high}"
MERGED_DIR="${RUN_DIR}/merged"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
# 8192: 4096 에서는 답변 생성 한도가 896 토큰까지밖에 확보되지 않는다(generator.py 의
# 예산 계산 참고). VRAM 이 부족하면 낮출 수 있지만 답변이 잘릴수록 점수가 떨어진다.
GEN_MODEL_LEN="${GEN_MODEL_LEN:-8192}"
JUDGE_MODEL="${JUDGE_MODEL:-gemma}"
JUDGE_MODEL_LEN="${JUDGE_MODEL_LEN:-8192}"
# OpenAI 심판 모델(gpt-4.1 등)에서만 쓰인다. evaluator.py 가 환경변수로 읽으므로
# 명령행 인자로 넘기지 않는다(ps 에 키가 노출되지 않도록).
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
JUDGE_THREADS="${JUDGE_THREADS:-30}"
# OpenAI 심판 모델의 분당 토큰 한도(TPM). evaluator.py 가 이 예산에 맞춰 요청 속도를
# 조절한다. 계정 등급에 맞게 올리면 채점이 그만큼 빨라진다(0 이면 속도 조절 없음).
# 이 값이 실제 한도보다 크면 429 가 나고, evaluator 가 재시도하느라 느려진다.
JUDGE_TPM="${JUDGE_TPM:-30000}"
export OPENAI_API_KEY

# 허깅페이스 토큰. 컨테이너 안으로 그대로 전달된다(scripts/docker_env.sh).
HF_TOKEN="${HF_TOKEN:-}"
export HF_TOKEN

# MODE=debug 는 짧은 확인용 실행: 학습/평가 데이터의 일부(TRAIN_FRACTION, 기본 10%)만 쓴다.
# MODE=full(기본)은 전체 데이터로 학습한다. TRAIN_FRACTION 을 직접 주면 그 값이 우선한다.
MODE="${MODE:-full}"
if [[ "${MODE}" == "debug" ]]; then
  TRAIN_FRACTION="${TRAIN_FRACTION:-0.1}"
else
  TRAIN_FRACTION="${TRAIN_FRACTION:-1.0}"
fi

# GPU 0 only. 컨테이너에는 GPU 를 전부 붙이고(--gpus all) 여기서 고른 번호만 보이게 한다.
export CUDA_VISIBLE_DEVICES="${GPU}"

# 도커 / 이미지 / 토큰을 파이프라인 시작 전에 미리 확인한다.
require_docker
require_image "${TRAIN_IMAGE}" "Dockerfile-train"
require_image "${INFER_IMAGE}" "Dockerfile-infer"
require_hf_token

# Ctrl-C / 중지 버튼으로 끊겼을 때 남은 컨테이너를 정리한다.
trap docker_cleanup EXIT INT TERM

# OpenAI 심판 모델은 API 키가 있어야 한다. 학습을 다 끝낸 뒤 Step 3 에서 실패하지 않도록
# 파이프라인 시작 전에 미리 확인한다.
case "${JUDGE_MODEL}" in
  gpt-*|chatgpt-*|o1*|o3*|o4*)
    if [[ -z "${OPENAI_API_KEY}" ]]; then
      echo "JUDGE_MODEL='${JUDGE_MODEL}' 는 OpenAI API 키가 필요합니다." >&2
      echo "  export OPENAI_API_KEY='sk-...' 후 다시 실행하세요 (웹 UI 는 OPENAI_API_KEY 입력칸)." >&2
      exit 1
    fi
    ;;
esac

step() {
  echo ""
  echo "=============================================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "=============================================================="
}

echo "==> docker images"
echo "  train (Step 1)      : ${TRAIN_IMAGE}"
echo "  infer (Step 2/3/4)  : ${INFER_IMAGE}"
echo "  project mount       : ${REPO_ROOT}"
echo "  HF cache mount      : ${HF_CACHE_DIR}"
echo "  container HOME      : ${DOCKER_HOME}"
echo "  GPU                 : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (--gpus ${DOCKER_GPUS})"

step "Step 0/4: cleaning previous outputs"
# The generator and evaluator both skip work when an output file already exists,
# so stale results must be removed before a fresh run.
# DOCKER_DRY_RUN=1 은 설정 점검용이므로 삭제까지 하지는 않는다.
if [[ "${DOCKER_DRY_RUN:-0}" == "1" ]]; then
  echo "(dry-run: 삭제 생략) rm -rf evaluated/${MERGED_DIR} generated/runs runs"
else
  rm -rf "evaluated/${MERGED_DIR}"
  rm -rf "generated/runs"
  rm -rf "runs"
fi

step "Step 1/4: LoRA SFT training (mode=${MODE}, data=${TRAIN_FRACTION}) -> ${RUN_DIR}"
docker_run_train train \
  python train/train_lora.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${RUN_DIR}" \
  --seed "${SEED}" \
  --gpu "${GPU}" \
  --train-fraction "${TRAIN_FRACTION}"

step "Step 2/4: generating LogicKor answers with ${MERGED_DIR}"
docker_run_infer generate \
  python logickor_eval/generator.py \
  --model "${MERGED_DIR}" \
  --gpu_devices "${GPU}" \
  --model_len "${GEN_MODEL_LEN}"

# --allow-partial: 채점에 실패한 항목이 있어도 파이프라인을 끝까지 진행한다.
# 실패 항목은 0 점이 아니라 '측정 불가'(null)로 기록되고 Step 4 에서 평균에서 제외되며,
# 몇 개가 빠졌는지 경고로 출력된다. 그 항목만 다시 채점하려면 파일을 지우지 말고
#   bash scripts/evaluate.sh "generated/${MERGED_DIR}" --rejudge-failed
# 를 실행한다(이미 성공한 채점은 그대로 두고 빈 곳만 메운다).
step "Step 3/4: judging with '${JUDGE_MODEL}' -> evaluated/${MERGED_DIR}/"
docker_run_infer evaluate \
  python logickor_eval/evaluator.py \
  -o "generated/${MERGED_DIR}" \
  -j "${JUDGE_MODEL}" \
  -g "${GPU}" \
  -ml "${JUDGE_MODEL_LEN}" \
  -t "${JUDGE_THREADS}" \
  --tpm "${JUDGE_TPM}" \
  --allow-partial

step "Step 4/4: scoring evaluated/${MERGED_DIR}/*.jsonl"
docker_run_infer score \
  python logickor_eval/score.py -p "evaluated/${MERGED_DIR}/*.jsonl"

step "Pipeline finished successfully"
