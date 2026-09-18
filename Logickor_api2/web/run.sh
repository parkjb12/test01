#!/usr/bin/env bash
# Logickor 평가 웹 서비스 실행 (도커)
#
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx   # 실행 전에 반드시 등록
#   bash web/run.sh                               # http://0.0.0.0:8000
#   PORT=8080 bash web/run.sh
#   WEB_IN_DOCKER=0 bash web/run.sh               # 웹 UI 만 호스트 파이썬으로 실행
#
# 기본 동작
#   - 웹 UI 를 INFER_IMAGE(Dockerfile-infer) 컨테이너에서 띄운다.
#   - 이 컨테이너에는 호스트의 도커 소켓이 물려 있어, ▶ 실행 버튼이 돌리는
#     scripts/auto.sh 가 학습은 TRAIN_IMAGE, 생성/채점/집계는 INFER_IMAGE 컨테이너를
#     형제 컨테이너로 띄운다.
#   - WEB_IN_DOCKER=0 이면 웹 UI 자체는 호스트 파이썬으로 뜨지만, auto.sh 는 그대로
#     도커 이미지를 쓴다(호스트에 fastapi/uvicorn 이 있어야 한다).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"                      # 프로젝트 루트에서 실행

source "${REPO_ROOT}/scripts/docker_env.sh"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WEB_IN_DOCKER="${WEB_IN_DOCKER:-1}"
WEB_CONTAINER_NAME="${WEB_CONTAINER_NAME:-logickor-web}"
DOCKER_SOCK="${DOCKER_SOCK:-/var/run/docker.sock}"

require_docker
# 학습 이미지는 웹에서 ▶ 실행을 누를 때 필요하다. 그때 실패하지 않도록 미리 확인한다.
require_image "${INFER_IMAGE}" "Dockerfile-infer"
require_image "${TRAIN_IMAGE}" "Dockerfile-train"
require_hf_token

if [[ "${WEB_IN_DOCKER}" != "1" ]]; then
  echo "==> 웹 UI: 호스트 파이썬 (auto.sh 는 그대로 도커를 사용합니다)"
  export HOST PORT
  exec python web/app.py
fi

if docker ps -q --filter "name=^/${WEB_CONTAINER_NAME}$" | grep -q .; then
  die "이미 '${WEB_CONTAINER_NAME}' 컨테이너가 실행 중입니다. (docker rm -f ${WEB_CONTAINER_NAME})"
fi

[[ -S "${DOCKER_SOCK}" ]] \
  || die "도커 소켓 '${DOCKER_SOCK}' 을 찾을 수 없습니다. DOCKER_SOCK 으로 경로를 지정하세요."

mkdir -p "${HF_CACHE_DIR}" "${DOCKER_HOME}" "${REPO_ROOT}/web/logs"

# 웹 UI 컨테이너 안에서 다시 docker run 을 하므로(형제 컨테이너),
#   * 프로젝트를 호스트와 똑같은 절대경로로 마운트하고
#   * 도커 소켓과 docker 그룹을 넘겨준다.
DOCKER_SOCK_GID="$(stat -c '%g' "${DOCKER_SOCK}")"

web_args=(
  run --rm
  --name "${WEB_CONTAINER_NAME}"
  --network host
  --user "${DOCKER_UID}:${DOCKER_GID}"
  --group-add "${DOCKER_SOCK_GID}"
  -v "${REPO_ROOT}:${REPO_ROOT}"
  -v "${HF_CACHE_DIR}:${HF_CACHE_DIR}"
  -v "${DOCKER_HOME}:${DOCKER_HOME}"
  -v "${DOCKER_SOCK}:/var/run/docker.sock"
  -w "${REPO_ROOT}"
  -e "HOME=${DOCKER_HOME}"
  -e "HF_HOME=${HF_CACHE_DIR}"
  -e "HF_CACHE_DIR=${HF_CACHE_DIR}"
  -e "USER=${DOCKER_USER_NAME}"
  -e "LOGNAME=${DOCKER_USER_NAME}"
  -e "DOCKER_USER_NAME=${DOCKER_USER_NAME}"
  -e "DOCKER_HOME=${DOCKER_HOME}"
  -e "REPO_ROOT=${REPO_ROOT}"
  -e "INFER_IMAGE=${INFER_IMAGE}"
  -e "TRAIN_IMAGE=${TRAIN_IMAGE}"
  -e "DOCKER_GPUS=${DOCKER_GPUS}"
  -e "DOCKER_SHM_SIZE=${DOCKER_SHM_SIZE}"
  -e "DOCKER_UID=${DOCKER_UID}"
  -e "DOCKER_GID=${DOCKER_GID}"
  -e "HOST=${HOST}"
  -e "PORT=${PORT}"
  -e PYTHONUNBUFFERED=1
)
for name in HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY DOCKER_EXTRA_ARGS; do
  if [[ -n "${!name:-}" ]]; then
    web_args+=(-e "${name}=${!name}")
  fi
done
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  web_args+=(-e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi
if [[ -t 1 ]]; then
  web_args+=(-it)          # 터미널에서 직접 띄웠을 때 Ctrl-C 로 바로 멈출 수 있게
fi

echo "==> 웹 UI 컨테이너: ${INFER_IMAGE}  (학습 컨테이너: ${TRAIN_IMAGE})"
echo "==> http://${HOST}:${PORT}"

if [[ "${DOCKER_DRY_RUN:-0}" == "1" ]]; then
  printf 'docker'; printf ' %q' "${web_args[@]}" "${INFER_IMAGE}" python web/app.py; printf '\n'
  exit 0
fi

exec docker "${web_args[@]}" "${INFER_IMAGE}" python web/app.py
