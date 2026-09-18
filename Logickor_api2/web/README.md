# Logickor 평가 웹 서비스

`scripts/auto.sh` 파이프라인(**clean → train → generate → evaluate → score**)을 브라우저에서
설정·실행·중지하고 결과를 확인하는 웹 UI 입니다.

프로젝트 소스는 수정하지 않고 `scripts/auto.sh` 를 서브프로세스로 실행합니다. 단, auto.sh 의
**Step 0** 이 `runs/`, `generated/runs/`, `evaluated/*.jsonl` 을 삭제하므로 이전 실행 산출물은
남지 않습니다.

## 도커 구성

웹 UI 자체와 파이프라인 각 단계가 모두 컨테이너에서 돕니다.

```
web/run.sh ──> [logickor-infer] 웹 UI (app.py)
                     │  (호스트 도커 소켓 마운트)
                     └─ scripts/auto.sh
                          ├─ Step 1 학습     -> [logickor-train]
                          └─ Step 2~4 생성/채점/집계 -> [logickor-infer]
```

- 학습만 `logickor-train`(`Dockerfile-train`), 나머지는 전부 `logickor-infer`(`Dockerfile-infer`).
- 프로젝트는 컨테이너에 **호스트와 같은 절대경로**로 마운트됩니다. 그래서 웹 UI 컨테이너 안에서
  다시 `docker run` 을 해도 경로가 그대로 통합니다(형제 컨테이너).
- 컨테이너는 호스트 사용자 uid 로 실행되어 산출물이 root 소유로 남지 않습니다.

## 실행

이미지 빌드는 프로젝트 루트 [`README.md`](../README.md) 1장을 참고하세요.

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx   # 실행 전에 반드시 등록
bash web/run.sh

# 웹 UI 만 호스트 파이썬으로 (auto.sh 는 그대로 도커 사용)
WEB_IN_DOCKER=0 bash web/run.sh
```

기본 주소: <http://0.0.0.0:8000> (환경변수 `HOST`, `PORT` 로 변경 가능)

원격 서버라면 SSH 포트포워딩 후 접속하세요:

```bash
ssh -L 8000:localhost:8000 parkjb@<서버>
```

## 기능

- **HF_TOKEN 표시** — 모델 다운로드용 토큰은 UI 입력칸이 아니라 서버를 띄운 셸의 환경변수로 받습니다
  (`export HF_TOKEN=hf_...` 후 `bash web/run.sh`). `JUDGE_MODEL` 아래에 `HF_TOKEN: 설정됨` 으로
  표시되고, 없으면 ▶ 실행이 학습 전에 막힙니다(`HF_TOKEN_OPTIONAL=1` 로 건너뛸 수 있음).
- **설정 표시/수정** — 입력값은 `auto.sh` 의 환경변수와 1:1로 대응합니다
  (`CONFIG_PATH`, `RUN_DIR`, `SEED`, `GPU`, `GEN_MODEL_LEN`, `JUDGE_MODEL`, `OPENAI_API_KEY`,
  `JUDGE_MODEL_LEN`).
  `RUN_DIR` 을 입력하면 `merged model`, `generated dir`, `evaluated glob` 이 auto.sh 와 동일한
  규칙으로 자동 유도되어 표시됩니다. `CONFIG_PATH` 는 `configs/*.yaml` 목록,
  `JUDGE_MODEL` 은 evaluator 프리셋(`gemma`, `llama`, `gpt-4.1`) 목록이 자동완성됩니다.
- **심판 모델 선택 (로컬 / OpenAI)** — `gemma`, `llama` 는 로컬 GPU 에서 vLLM 으로 구동하고,
  `gpt-4.1` 은 OpenAI API 로 채점합니다(GPU 미사용, 유료 호출).
  `gpt-4.1` 을 고르면 `JUDGE_MODEL` 바로 아래 **`OPENAI_API_KEY` 입력칸**에 키를 넣고 저장하세요.
  키를 비워두면 서버 환경변수 `OPENAI_API_KEY` 를 사용하며, 둘 다 없으면 실행 시작 시점에
  (학습을 시작하기 전에) 오류로 알려줍니다. 로컬 모델을 고르면 이 칸은 쓰이지 않습니다.
  입력한 키는 `web/settings.json` 에 평문으로 저장되므로(`.gitignore` 처리됨) 공용 서버에서는
  파일 권한에 유의하세요.
- **실행 모드 (full / debug)** — `debug` 를 고르면 `TRAIN_FRACTION` 비율(기본 `0.1` = 10%)만큼만
  학습·평가합니다. 샘플링은 `question_id` 단위라 turn1/turn2 쌍이 깨지지 않습니다.
  파이프라인 전체를 짧게 점검할 때 쓰고, 이때 나오는 점수는 참고용입니다.
  `full` 모드는 `TRAIN_FRACTION=1.0` 으로 전체 데이터를 학습합니다.
  (생성·평가 단계는 LogicKor 질문 42개 전체를 그대로 사용합니다.)
- **config 파일 편집** — 선택한 yaml 내용을 직접 보고 저장.
- **실행 / 중지** — ▶ 실행 버튼으로 파이프라인 시작, ■ 중지 버튼으로 프로세스 그룹 종료.
- **실시간 로그 · 진행 상황** — auto.sh 의 `[시각] Step n/4:` 마커와 tqdm 출력을 파싱해
  단계별 상태와 진행률을 표시합니다.
- **결과** — 평가 완료 시 카테고리별 Single/Multi 점수, 레이더 차트, 종합 점수를 표시.

## 파일

| 파일 | 설명 |
|---|---|
| `web/app.py` | FastAPI 백엔드 (실행/중지/설정/결과 API) |
| `web/index.html` | 프론트엔드 (검정 배경 + `sample.png` 디자인) |
| `web/run.sh` | 실행 스크립트 (웹 UI 컨테이너 기동) |
| `web/settings.json` | 저장된 설정 (실행 중 자동 생성) |
| `web/logs/pipeline.log` | 최근 실행 로그 (자동 생성) |

## 설정 기본값 (auto.sh 와 동일)

| 항목 | 환경변수 | 기본값 |
|---|---|---|
| config | `CONFIG_PATH` | `configs/train_gemma4_e4b_sft.yaml` |
| run dir | `RUN_DIR` | `runs/gemma4_e4b_sft_high` |
| seed | `SEED` | `42` |
| GPU | `GPU` | `0` |
| 생성 모델 최대 길이 | `GEN_MODEL_LEN` | `4096` |
| 심판 모델 | `JUDGE_MODEL` | `gemma` (`gemma` \| `llama` \| `gpt-4.1`) |
| OpenAI API 키 | `OPENAI_API_KEY` | (비어 있음, `gpt-4.1` 일 때만 필요) |
| 심판 모델 최대 길이 | `JUDGE_MODEL_LEN` | `8192` (로컬 심판 모델에만 적용) |
| 실행 모드 | `MODE` | `full` (`full` \| `debug`) |
| 학습 데이터 비율 | `TRAIN_FRACTION` | `1.0` (debug 모드 기본 `0.1`) |
| merged model | (유도) | `<RUN_DIR>/merged` |
| generated dir | (유도) | `generated/<RUN_DIR>/merged` |
| evaluated glob | (유도) | `evaluated/*.jsonl` |

auto.sh 에는 단계 건너뛰기(SKIP) 기능이 없으므로 UI 에서도 제공하지 않습니다.
