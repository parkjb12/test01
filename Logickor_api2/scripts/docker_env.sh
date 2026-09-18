#!/usr/bin/env bash
# 파이프라인 각 단계를 도커 컨테이너에서 실행하기 위한 공통 설정.
#   - scripts/*.sh 와 web/run.sh 에서 `source` 해서 쓴다. 단독 실행용이 아니다.
#   - 학습은 TRAIN_IMAGE, 그 외(생성/채점/집계/웹 UI)는 INFER_IMAGE 를 쓴다.
#
# 규칙
#   * 프로젝트를 컨테이너 안에 "호스트와 똑같은 절대경로"로 마운트한다.
#     그래야 웹 UI 컨테이너 안에서 다시 docker run 을 해도(-v 는 호스트 데몬이 해석한다)
#     같은 경로가 그대로 통한다.
#   * 컨테이너를 호스트 사용자 uid:gid 로 실행한다 -> runs/ generated/ evaluated/ 가
#     root 소유로 남지 않는다.
#   * GPU 는 항상 전부 붙이고(--gpus all) 어느 GPU 를 쓸지는 기존처럼
#     CUDA_VISIBLE_DEVICES / --gpu 인자로 고른다. 호스트와 GPU 번호가 같게 유지된다.

# 호스트 기준 프로젝트 루트(컨테이너 안에서도 동일 경로).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

INFER_IMAGE="${INFER_IMAGE:-logickor-infer:latest}"
TRAIN_IMAGE="${TRAIN_IMAGE:-logickor-train:latest}"

# 모델 캐시(기본 ~/.cache/huggingface). 이미 받아둔 모델을 그대로 재사용한다.
HF_CACHE_DIR="${HF_CACHE_DIR:-${HOME}/.cache/huggingface}"
# 컨테이너 안 HOME. torch.compile / triton / vLLM 캐시가 실행 간에 유지된다.
DOCKER_HOME="${DOCKER_HOME:-${REPO_ROOT}/.docker-home}"

DOCKER_GPUS="${DOCKER_GPUS:-all}"
DOCKER_SHM_SIZE="${DOCKER_SHM_SIZE:-16g}"
# 추가로 넘기고 싶은 docker run 옵션(공백 구분). 예: DOCKER_EXTRA_ARGS="--network host"
DOCKER_EXTRA_ARGS="${DOCKER_EXTRA_ARGS:-}"

# 한 번의 파이프라인 실행에 속한 컨테이너를 한꺼번에 정리하기 위한 라벨.
LOGICKOR_RUN_ID="${LOGICKOR_RUN_ID:-$$-$(date +%s)}"

DOCKER_UID="${DOCKER_UID:-$(id -u)}"
DOCKER_GID="${DOCKER_GID:-$(id -g)}"
# 컨테이너를 호스트 uid 로 실행하므로 이미지의 /etc/passwd 에는 이 uid 에 해당하는
# 계정이 없다. 이 상태에서 파이썬 getpass.getuser() 를 부르면
#   KeyError: 'getpwuid(): uid not found: 1005'
# 가 난다(torch 의 _inductor 캐시 경로 계산이 import 중에 이 함수를 쓴다).
# getuser() 는 LOGNAME/USER 환경변수를 먼저 보므로 이름만 알려주면 된다.
DOCKER_USER_NAME="${DOCKER_USER_NAME:-$(id -un 2>/dev/null || echo logickor)}"

die() { echo "$*" >&2; exit 1; }

require_docker() {
  command -v docker >/dev/null 2>&1 \
    || die "docker 명령을 찾을 수 없습니다. 도커를 설치하고 다시 실행하세요."
  docker info >/dev/null 2>&1 \
    || die "도커 데몬에 접근할 수 없습니다. (sudo usermod -aG docker \$USER 후 재로그인)"
}

require_image() {
  local image="$1" dockerfile="$2"
  if [[ "${DOCKER_DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi
  if docker image inspect "${image}" >/dev/null 2>&1; then
    return 0
  fi
  cat >&2 <<EOF
도커 이미지 '${image}' 가 없습니다. 먼저 빌드하세요:

  docker build -f ${dockerfile} -t ${image} .

EOF
  exit 1
}

require_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return 0
  fi
  cat >&2 <<'EOF'
HF_TOKEN 이 설정되어 있지 않습니다. 실행 전에 허깅페이스 토큰을 등록하세요:

  export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

(이미 ~/.cache/huggingface 에 받아둔 모델만 쓴다면 HF_TOKEN_OPTIONAL=1 로 건너뛸 수 있습니다.)
EOF
  if [[ "${HF_TOKEN_OPTIONAL:-0}" != "1" ]]; then
    exit 1
  fi
  echo "HF_TOKEN_OPTIONAL=1 -> 토큰 없이 계속합니다." >&2
  return 0
}

# 실행 전에 마운트 대상 디렉터리를 만들어 둔다.
# (없는 경로를 -v 로 넘기면 도커가 root 소유로 만들어 버린다.)
_ensure_mount_dirs() {
  mkdir -p "${HF_CACHE_DIR}" "${DOCKER_HOME}" 2>/dev/null || true
}

# 이번 실행이 만든 컨테이너를 모두 제거한다(중지 버튼 / Ctrl-C 대비).
docker_cleanup() {
  local ids
  ids="$(docker ps -aq --filter "label=logickor.run=${LOGICKOR_RUN_ID}" 2>/dev/null || true)"
  if [[ -n "${ids}" ]]; then
    # shellcheck disable=SC2086
    docker rm -f ${ids} >/dev/null 2>&1 || true
  fi
  return 0
}

# docker_run <image> <step-name> <command...>
#
# 컨테이너에서 명령 하나를 실행하고 종료 코드를 그대로 돌려준다.
# docker run 은 기본적으로 시그널을 컨테이너로 전달하므로 웹 UI 의 중지 버튼
# (프로세스 그룹 SIGTERM)이 그대로 동작한다.
docker_run() {
  local image="$1" step="$2"
  shift 2
  _ensure_mount_dirs

  local -a args=(
    run --rm -i
    --label "logickor.run=${LOGICKOR_RUN_ID}"
    --name "logickor-${step}-${LOGICKOR_RUN_ID}"
    --gpus "${DOCKER_GPUS}"
    --shm-size "${DOCKER_SHM_SIZE}"
    --user "${DOCKER_UID}:${DOCKER_GID}"
    -v "${REPO_ROOT}:${REPO_ROOT}"
    -v "${HF_CACHE_DIR}:${HF_CACHE_DIR}"
    -v "${DOCKER_HOME}:${DOCKER_HOME}"
    -w "${REPO_ROOT}"
    -e "HOME=${DOCKER_HOME}"
    -e "HF_HOME=${HF_CACHE_DIR}"
    -e "USER=${DOCKER_USER_NAME}"
    -e "LOGNAME=${DOCKER_USER_NAME}"
    # 기본값은 /tmp/torchinductor_<user> 라 컨테이너가 끝나면 사라진다.
    # 마운트된 HOME 아래로 옮겨서 실행 간에 재사용한다.
    -e "TORCHINDUCTOR_CACHE_DIR=${DOCKER_HOME}/torchinductor"
    -e PYTHONUNBUFFERED=1
  )

  # 값이 설정된 것만 전달한다(미설정 변수는 아예 넘기지 않는다).
  local name
  for name in HF_TOKEN HUGGING_FACE_HUB_TOKEN OPENAI_API_KEY CUDA_VISIBLE_DEVICES \
              TRAIN_FRACTION TOKENIZERS_PARALLELISM; do
    if [[ -n "${!name:-}" ]]; then
      args+=(-e "${name}=${!name}")
    fi
  done
  # evaluator.py 는 HF_TOKEN 만 읽지만 huggingface_hub 는 양쪽을 본다.
  if [[ -n "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    args+=(-e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
  fi

  if [[ -n "${DOCKER_EXTRA_ARGS}" ]]; then
    # 공백으로 나뉜 추가 옵션을 그대로 펼친다.
    # shellcheck disable=SC2206
    args+=(${DOCKER_EXTRA_ARGS})
  fi

  if [[ "${DOCKER_DRY_RUN:-0}" == "1" ]]; then
    # 실제로 실행하지 않고 만들어진 docker 명령만 보여준다(설정 점검용).
    printf 'docker'; printf ' %q' "${args[@]}" "${image}" "$@"; printf '\n'
    return 0
  fi

  docker "${args[@]}" "${image}" "$@"
}

# 학습 단계 전용(도커 이미지가 다르다).
docker_run_train() {
  local step="$1"; shift
  require_image "${TRAIN_IMAGE}" "Dockerfile-train"
  docker_run "${TRAIN_IMAGE}" "${step}" "$@"
}

# 생성 / 채점 / 집계 등 학습 외 모든 단계.
docker_run_infer() {
  local step="$1"; shift
  require_image "${INFER_IMAGE}" "Dockerfile-infer"
  docker_run "${INFER_IMAGE}" "${step}" "$@"
}
