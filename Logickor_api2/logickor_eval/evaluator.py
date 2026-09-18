import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

# vLLM runtime options (must be set before importing vllm)
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")

import pandas as pd
from tqdm import tqdm

from hf_compat import heterogeneous_config_overrides
from templates import JUDGE_TEMPLATE

# Constants
TIME_START = datetime.now().strftime("%Y%m%d_%H%M%S")

# Named judge-model presets. Pass the key to -j/--judge-model to select one,
# or pass any custom local path / HF repo id directly.
JUDGE_MODEL_PRESETS = {
    "llama": "/home/parkjb/Llama-3.1-8B-Instruct",
    # HF cache dir; vLLM resolves the repo id from the local cache automatically.
    "gemma": "google/gemma-4-E4B-it",
    # OpenAI API judge. Needs OPENAI_API_KEY (env or -k); no GPU is used.
    "gpt-4.1": "gpt-4.1",
}

DEFAULT_JUDGE_MODEL = "llama"

# Model names served through the OpenAI API instead of a local vLLM engine.
OPENAI_MODEL_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")

DEFAULT_THREADS = 30

# OpenAI enforces a tokens-per-minute ceiling per organisation, and unpaced threads walk
# straight into it: a full LogicKor run is ~250 judgements of ~2k tokens, so 30 threads
# ask for ~60k tokens at once against a 30k/min budget. That is what cost the 2026-09-11
# run 18 judgements -- every one of them a 429, not a bad answer. Pace the requests
# against the same budget instead of discovering the limit by being rejected.
DEFAULT_TOKENS_PER_MINUTE = 30000

# How long a single item may stay stuck behind rate limits before it is given up on.
# Waiting out a 429 costs seconds; recording the item as unscored costs a judgement.
DEFAULT_RATE_LIMIT_BUDGET_SECONDS = 600

# A judgement the judge could not produce is NOT a zero. Scoring it zero silently
# drags the benchmark average down, so unscored items carry a null score and this
# marker, and score.py excludes them from the average instead of averaging in a 0.
JUDGE_FAILED_SCORE = None

# How much of the judge's raw reply to keep on failure. Without it a failed judgement is
# a dead end: rerunning is the only way to learn whether the judge rambled past its token
# cap, was rate-limited, or answered off-format.
RAW_EXCERPT_CHARS = 2000


def resolve_judge_model(judge_model: str) -> str:
    """Map a preset name to its model path/id; pass anything else through unchanged."""
    return JUDGE_MODEL_PRESETS.get(judge_model, judge_model)


def is_openai_model(judge_model: str) -> bool:
    """True if the (already resolved) judge model is served by the OpenAI API."""
    return judge_model.startswith(OPENAI_MODEL_PREFIXES)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--model-output-dir", help="Model Output Directory", required=True)
    parser.add_argument(
        "-j",
        "--judge-model",
        help=(
            "Judge Model: a preset name "
            f"({', '.join(JUDGE_MODEL_PRESETS)}), a local path, or an HF repo id"
        ),
        default=DEFAULT_JUDGE_MODEL,
    )
    parser.add_argument(
        "--hf-token",
        help="Hugging Face access token (used to download/authenticate the judge model)",
        default=os.environ.get("HF_TOKEN", None),
    )
    parser.add_argument(
        "-k",
        "--openai-api-key",
        help="OpenAI API key (only used for OpenAI judge models such as gpt-4.1)",
        default=os.environ.get("OPENAI_API_KEY", None),
    )
    parser.add_argument(
        "-t",
        "--threads",
        help="Parallel requests for OpenAI judge models",
        default=DEFAULT_THREADS,
        type=int,
    )
    parser.add_argument(
        "--tpm",
        help=(
            "Tokens-per-minute budget for OpenAI judge models; requests are paced to stay "
            "under it. Match your account's TPM limit (0 disables pacing)"
        ),
        default=int(os.environ.get("OPENAI_TPM", DEFAULT_TOKENS_PER_MINUTE)),
        type=int,
    )
    parser.add_argument(
        "--rate-limit-budget",
        help="Seconds a single item may spend waiting out OpenAI rate limits",
        default=DEFAULT_RATE_LIMIT_BUDGET_SECONDS,
        type=float,
    )
    parser.add_argument("-g", "--gpu-devices", help="CUDA_VISIBLE_DEVICES", default="0")
    parser.add_argument("-ml", "--model-len", help="Maximum Model Length", default=8192, type=int)
    parser.add_argument(
        "-mo",
        "--max-output-tokens",
        # The judge is asked for a 4-5 sentence verdict plus a score line; real verdicts
        # run 130-320 tokens. A cap in the thousands buys no quality -- it only gives a
        # judge that has fallen into a repetition loop room to ramble past the score, and
        # inflates the per-request token estimate that OpenAI pacing is based on.
        help="Maximum tokens generated by the judge model",
        default=1024,
        type=int,
    )
    parser.add_argument(
        "--allow-partial",
        help="Exit 0 even when some items could not be judged (default: exit 1)",
        action="store_true",
    )
    parser.add_argument(
        "--rejudge-failed",
        help=(
            "Re-judge only the unscored items of an already-evaluated file, in place, "
            "instead of skipping the file (the rest of the judgements are kept)"
        ),
        action="store_true",
    )
    return parser.parse_args()


# The judge is told to answer "점수: 숫자" and usually does, but a drifting judge still
# states a score -- reading it is strictly better than discarding a real judgement. The
# strict forms are tried first so a stray number can never outrank an explicit verdict.
_SCORE_PATTERN_GROUPS = (
    (r"점수\s*[:=]\s*(\d+(?:\.\d+)?)",),
    (r"점수는\s*(\d+(?:\.\d+)?)", r"점수\s+(\d+(?:\.\d+)?)\s*점"),
    (r"(?i)\bscore\s*[:=]\s*(\d+(?:\.\d+)?)", r"\[\[\s*(\d+(?:\.\d+)?)\s*\]\]"),
)

# Last resort: a verdict whose closing line is nothing but the number ("8", "8/10", "8점").
_FINAL_LINE_PATTERN = r"^\s*(\d+(?:\.\d+)?)\s*(?:/\s*10|점)?\s*$"


def normalise_judgement(content: str) -> str:
    """Strip the markdown the judge decorates its verdict with and unify the colon."""
    return content.replace("*", "").replace("#", "").replace("：", ":")


def find_score(cleaned: str) -> Optional[float]:
    """Pull the judge's score out of its (already normalised) reply."""
    for patterns in _SCORE_PATTERN_GROUPS:
        matches = [m for pattern in patterns for m in re.finditer(pattern, cleaned)]
        if matches:
            # The judge sometimes mentions a score inside its prose before stating the
            # verdict, so the final occurrence is the authoritative one.
            return float(max(matches, key=lambda m: m.start()).group(1))

    lines = cleaned.strip().splitlines()
    final_match = re.match(_FINAL_LINE_PATTERN, lines[-1]) if lines else None
    return float(final_match.group(1)) if final_match else None


def parse_judgement(content: Optional[str]) -> Optional[Dict[str, Union[str, float]]]:
    """Parse the judge's raw text into a message/score dict, or None if it has no score.

    Returning None (rather than a 0.0) is what lets the callers retry and, failing
    that, record the item as unjudged instead of as a zero.
    """
    if not content:
        return None
    cleaned = normalise_judgement(content)

    score = find_score(cleaned)
    if score is None:
        return None
    judge_score = max(0.0, min(10.0, score))

    judge_message_match = re.search(r"평가:(.*?)점수:", cleaned, re.DOTALL)
    if judge_message_match:
        judge_message = judge_message_match.group(1).strip()
    else:
        # The judge wrote plain prose instead of the 평가:/점수: frame. Keep the prose --
        # it is the reasoning behind the score we just accepted.
        judge_message = re.split(r"점수\s*[:=]", cleaned)[0].strip() or "No judge message found"
    return {"judge_message": judge_message, "judge_score": judge_score}


def failed_judgement(reason: str, raw: Optional[str] = None) -> Dict[str, Union[str, None]]:
    """Record an item the judge could not score, keeping why and what it did say."""
    failed: Dict[str, Union[str, None]] = {
        "judge_message": f"Impossible to judge: {reason}",
        "judge_score": JUDGE_FAILED_SCORE,
    }
    if raw:
        failed["judge_raw"] = raw[-RAW_EXCERPT_CHARS:]
    return failed


# A judge that never reached its score line has still written the analysis; asking it for
# the score alone is a small, always-in-context request that recovers the judgement.
RESCUE_SYSTEM_PROMPT = "너는 이미 작성된 한국어 평가문을 읽고 최종 점수만 뽑아내는 역할이다."


def rescue_conversation(verdict: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": RESCUE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "아래는 어떤 답변에 대한 평가문이다. 평가 내용을 근거로 1~10 사이의 최종 점수를 정하라.\n\n"
                f"**평가문**\n{verdict}\n\n"
                "출력은 반드시 '점수: 숫자' 한 줄이어야 한다."
            ),
        },
    ]


class LocalJudge:
    """Local judge model served by vLLM (aphrodite fallback)."""

    MAX_RETRIES = 3
    # Room left for whatever the chat template adds on top of the rendered prompt.
    CONTEXT_MARGIN_TOKENS = 128
    # A verdict needs a few sentences plus the score line; never budget less than this.
    MIN_OUTPUT_TOKENS = 256
    # The rescue pass only has to emit "점수: 숫자".
    RESCUE_OUTPUT_TOKENS = 32
    # How much of the failed verdict to feed back into the rescue pass. The analysis is
    # at the front, so the head is what carries the information the score follows from.
    RESCUE_CONTEXT_CHARS = 3000

    def __init__(self, judge_model: str, gpu_devices: str, model_len: int, max_output_tokens: int):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
        gpu_counts = len(gpu_devices.split(","))

        try:
            from aphrodite import LLM, SamplingParams

            print("- Using aphrodite-engine")
        except ImportError:
            from vllm import LLM, SamplingParams

            print("- Using vLLM")

        self.sampling_params_cls = SamplingParams
        self.model_len = model_len
        self.max_output_tokens = max_output_tokens
        self.llm = LLM(
            model=judge_model,
            tensor_parallel_size=gpu_counts,
            max_model_len=model_len,
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            enforce_eager=True,
            # Needed for models with per-layer (heterogeneous) HF configs, e.g. gemma-4-E4B.
            hf_overrides=heterogeneous_config_overrides,
        )
        self.tokenizer = self._resolve_tokenizer()

    def _resolve_tokenizer(self):
        """The tokenizer with apply_chat_template(), wherever this engine keeps it."""
        tokenizer = None
        if hasattr(self.llm, "get_tokenizer"):
            try:
                tokenizer = self.llm.get_tokenizer()
            except Exception:
                tokenizer = None
        if tokenizer is None:
            tokenizer = getattr(getattr(self.llm, "llm_engine", None), "tokenizer", None)

        current = tokenizer
        for _ in range(4):
            if current is None:
                break
            if hasattr(current, "apply_chat_template"):
                return current
            current = getattr(current, "tokenizer", None)
        return None

    def _prompt_tokens(self, conversation: List[Dict[str, str]]) -> Optional[int]:
        if self.tokenizer is None:
            return None
        try:
            text = self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            return None

    def _budget(self, conversation: List[Dict[str, str]], requested: int) -> int:
        """Cap the reply so prompt + reply stays inside the context window.

        Asking for more than the window holds does not produce a longer verdict -- the
        engine simply stops early, mid-sentence, before the score line. The old retry
        ladder doubled max_tokens to 7680 on an 8192 window, which left a ~4k-token
        judging prompt no room at all to answer.
        """
        prompt_tokens = self._prompt_tokens(conversation)
        if prompt_tokens is None:
            return requested
        room = self.model_len - prompt_tokens - self.CONTEXT_MARGIN_TOKENS
        return max(self.MIN_OUTPUT_TOKENS, min(requested, room))

    def _sampling_params(self, max_tokens: int, temperature: float, repetition_penalty: float):
        try:
            return self.sampling_params_cls(
                temperature=temperature,
                max_tokens=max_tokens,
                repetition_penalty=repetition_penalty,
            )
        except TypeError:  # engine without repetition_penalty support
            return self.sampling_params_cls(temperature=temperature, max_tokens=max_tokens)

    def _run(
        self,
        conversations,
        requested_tokens: int,
        temperature: float,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        params = [
            self._sampling_params(
                self._budget(conversation, requested_tokens), temperature, repetition_penalty
            )
            for conversation in conversations
        ]
        outputs = self.llm.chat(conversations, params, use_tqdm=True)
        return [out.outputs[0].text for out in outputs]

    @staticmethod
    def _attempt_settings(attempt: int) -> Tuple[float, float]:
        """Greedy first; then break the repetition loops that eat the whole token budget.

        Re-running a greedy decode unchanged just reproduces the same unparseable text,
        so each retry has to change something about how the text is sampled.
        """
        return ((0.0, 1.0), (0.3, 1.1), (0.7, 1.15))[min(attempt, 2)]

    def judge(self, conversations: List[List[Dict[str, str]]]) -> List[Dict]:
        results: List[Optional[Dict]] = [None] * len(conversations)
        last_raw: List[str] = [""] * len(conversations)
        pending = list(range(len(conversations)))

        for attempt in range(self.MAX_RETRIES):
            if not pending:
                break
            temperature, repetition_penalty = self._attempt_settings(attempt)
            if attempt:
                print(
                    f"- Retrying {len(pending)} unparseable judgement(s) "
                    f"(temperature={temperature}, repetition_penalty={repetition_penalty})"
                )
            contents = self._run(
                [conversations[i] for i in pending],
                self.max_output_tokens,
                temperature,
                repetition_penalty,
            )
            still_pending = []
            for idx, content in zip(pending, contents):
                last_raw[idx] = content
                parsed = parse_judgement(content)
                if parsed is None:
                    still_pending.append(idx)
                else:
                    results[idx] = parsed
            pending = still_pending

        pending = self._rescue(pending, last_raw, results)

        for idx in pending:
            print(f"! Judge produced no score for item {idx} after {self.MAX_RETRIES} attempts.")
            results[idx] = failed_judgement(
                "the judge returned no parseable score", last_raw[idx]
            )
        # Positional: the caller zips these back onto (row, turn), so never drop an entry.
        return [r if r is not None else failed_judgement("judging did not run") for r in results]

    def _rescue(
        self, pending: List[int], last_raw: List[str], results: List[Optional[Dict]]
    ) -> List[int]:
        """Ask for the score alone on verdicts that never got to one."""
        rescuable = [idx for idx in pending if last_raw[idx].strip()]
        if not rescuable:
            return pending

        print(f"- Asking the judge for the score alone on {len(rescuable)} verdict(s)")
        conversations = [
            rescue_conversation(last_raw[idx][: self.RESCUE_CONTEXT_CHARS]) for idx in rescuable
        ]
        contents = self._run(conversations, self.RESCUE_OUTPUT_TOKENS, 0.0)

        still_pending = [idx for idx in pending if idx not in set(rescuable)]
        for idx, content in zip(rescuable, contents):
            score = find_score(normalise_judgement(content or ""))
            if score is None:
                still_pending.append(idx)
            else:
                results[idx] = {
                    "judge_message": last_raw[idx].strip(),
                    "judge_score": max(0.0, min(10.0, score)),
                }
        return sorted(still_pending)


class TokenRateLimiter:
    """Client-side tokens-per-minute budget shared by every judging thread.

    A 429 is not free: the request is rejected after the wait, so the only way to keep a
    judging run inside the account's limit is to not send more than the limit allows.
    """

    WINDOW_SECONDS = 60.0

    def __init__(self, tokens_per_minute: int):
        self.limit = max(0, tokens_per_minute)
        self.lock = threading.Lock()
        self.spent: Deque[Tuple[float, int]] = deque()

    def acquire(self, tokens: int) -> None:
        if not self.limit:
            return
        tokens = max(1, min(tokens, self.limit))
        while True:
            with self.lock:
                now = time.monotonic()
                while self.spent and now - self.spent[0][0] >= self.WINDOW_SECONDS:
                    self.spent.popleft()
                if sum(t for _, t in self.spent) + tokens <= self.limit:
                    self.spent.append((now, tokens))
                    return
                wait = self.WINDOW_SECONDS - (now - self.spent[0][0])
            time.sleep(max(0.05, min(wait, 5.0)) + random.uniform(0, 0.1))


def rate_limit_delay(exc: Exception) -> Optional[float]:
    """Seconds to wait for a rate-limit error, or None if this is a different failure.

    OpenAI says exactly how long to wait, in a `retry-after` header or in the message
    ("Please try again in 3.218s"). Honouring it beats a blind exponential ladder, which
    is what turned four transient 429s into four permanently unscored items.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    message = str(exc)
    if status != 429 and "rate_limit" not in message and type(exc).__name__ != "RateLimitError":
        return None

    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        for key, scale in (("retry-after-ms", 0.001), ("retry-after", 1.0)):
            try:
                value = headers.get(key)
            except AttributeError:
                value = None
            if value:
                try:
                    return max(0.5, float(value) * scale)
                except ValueError:
                    pass

    match = re.search(r"try again in\s+([\d.]+)\s*(ms|s)\b", message)
    if match:
        seconds = float(match.group(1)) * (0.001 if match.group(2) == "ms" else 1.0)
        # The stated wait only clears the current window; add a beat so the retry does
        # not arrive at the exact moment the budget frees up and lose the race again.
        return max(0.5, seconds) + 1.0
    return 10.0


class OpenAIJudge:
    """Judge model served by the OpenAI API (e.g. gpt-4.1). No GPU is used."""

    MAX_RETRIES = 6
    BASE_BACKOFF_SECONDS = 5
    # Tokens a verdict actually takes, for the pacing estimate (the cap is far higher).
    EXPECTED_OUTPUT_TOKENS = 400
    RESCUE_OUTPUT_TOKENS = 16
    RESCUE_CONTEXT_CHARS = 3000

    def __init__(
        self,
        judge_model: str,
        api_key: str,
        threads: int,
        max_output_tokens: int,
        tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE,
        rate_limit_budget: float = DEFAULT_RATE_LIMIT_BUDGET_SECONDS,
    ):
        if not api_key:
            raise SystemExit(
                f"OPENAI_API_KEY is required for the judge model '{judge_model}'. "
                "Set the env var or pass -k/--openai-api-key."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = judge_model
        self.threads = max(1, threads)
        self.max_output_tokens = max_output_tokens
        self.rate_limit_budget = rate_limit_budget
        self.limiter = TokenRateLimiter(tokens_per_minute)
        self.encoder = self._resolve_encoder()

    def _resolve_encoder(self):
        try:
            import tiktoken

            try:
                return tiktoken.encoding_for_model(self.model)
            except KeyError:
                return tiktoken.get_encoding("o200k_base")
        except Exception:
            return None

    def _count_tokens(self, text: str) -> int:
        if self.encoder is not None:
            try:
                return len(self.encoder.encode(text))
            except Exception:
                pass
        # Korean runs about two characters per token; erring high only slows pacing.
        return len(text) // 2 + 1

    def _estimate_tokens(self, conversation: List[Dict[str, str]]) -> int:
        prompt = "".join(message["content"] for message in conversation)
        return self._count_tokens(prompt) + min(self.max_output_tokens, self.EXPECTED_OUTPUT_TOKENS)

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(60, self.BASE_BACKOFF_SECONDS * (2 ** attempt))
        time.sleep(delay + random.uniform(0, 1))

    def _complete(self, conversation: List[Dict[str, str]], max_tokens: int) -> Tuple[str, str]:
        """One paced chat completion. Returns (text, finish_reason)."""
        self.limiter.acquire(self._estimate_tokens(conversation))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=conversation,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        choice = response.choices[0]
        return choice.message.content or "", choice.finish_reason or ""

    def _judge_one(self, conversation: List[Dict[str, str]]) -> Dict:
        """One judgement, retried on transport errors, rate limits and unparseable output."""
        max_tokens = self.max_output_tokens
        last_error = "no score in judge response"
        last_raw = ""
        rate_limited_for = 0.0
        attempt = 0

        while attempt < self.MAX_RETRIES:
            try:
                content, finish_reason = self._complete(conversation, max_tokens)
            except Exception as exc:  # rate limit / transient network errors
                delay = rate_limit_delay(exc)
                if delay is not None and rate_limited_for + delay <= self.rate_limit_budget:
                    # A 429 means the account's per-minute ceiling, not a bad request:
                    # it says nothing about this item, so it must not burn a retry.
                    time.sleep(delay)
                    rate_limited_for += delay
                    continue
                last_error = f"{type(exc).__name__}: {exc}"
                attempt += 1
                if attempt < self.MAX_RETRIES:
                    self._sleep_backoff(attempt - 1)
                continue

            last_raw = content or last_raw
            parsed = parse_judgement(content)
            if parsed is not None:
                return parsed

            # Deterministic decoding means an identical retry returns an identical
            # answer, so change something: a verdict cut off by the token cap is the
            # usual cause, so widen it before trying again.
            if finish_reason == "length":
                last_error = "judge response hit the max_tokens cap before stating a score"
                max_tokens = min(16384, max_tokens * 2)
            attempt += 1
            if attempt < self.MAX_RETRIES:
                self._sleep_backoff(attempt - 1)

        rescued = self._rescue(last_raw)
        if rescued is not None:
            return rescued

        print(f"! Judge failed after {self.MAX_RETRIES} attempts ({last_error}); leaving unscored.")
        return failed_judgement(last_error, last_raw)

    def _rescue(self, verdict: str) -> Optional[Dict]:
        """Ask for the score alone when the judge wrote a verdict but never scored it."""
        if not verdict.strip():
            return None
        try:
            content, _ = self._complete(
                rescue_conversation(verdict[: self.RESCUE_CONTEXT_CHARS]),
                self.RESCUE_OUTPUT_TOKENS,
            )
        except Exception:
            return None
        score = find_score(normalise_judgement(content or ""))
        if score is None:
            return None
        return {"judge_message": verdict.strip(), "judge_score": max(0.0, min(10.0, score))}

    def judge(self, conversations: List[List[Dict[str, str]]]) -> List[Dict]:
        results: List[Optional[Dict]] = [None] * len(conversations)
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            futures = {
                pool.submit(self._judge_one, conv): i for i, conv in enumerate(conversations)
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Judging", dynamic_ncols=True
            ):
                results[futures[future]] = future.result()
        return [r if r is not None else failed_judgement("judging did not run") for r in results]


def create_judge(args, judge_model: str):
    """Pick the judge backend from the resolved model name."""
    if is_openai_model(judge_model):
        pacing = f"{args.tpm} tokens/min" if args.tpm else "no token pacing"
        print(f"- Using OpenAI API ({args.threads} threads, {pacing})")
        return OpenAIJudge(
            judge_model,
            args.openai_api_key,
            args.threads,
            args.max_output_tokens,
            tokens_per_minute=args.tpm,
            rate_limit_budget=args.rate_limit_budget,
        )
    return LocalJudge(judge_model, args.gpu_devices, args.model_len, args.max_output_tokens)


def build_prompt(model_output, is_multi_turn: bool) -> Tuple[str, str]:
    """Build the (system, user) messages for a single evaluation."""
    model_questions = model_output["questions"]
    model_outputs = model_output["outputs"]
    model_references = model_output["references"]

    prompt = (
        f"아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. 특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.\n\n"
        f"**Question**\n{model_questions[0]}"
    )

    if model_references and model_references[0]:
        prompt += f"\n\n**Additional Reference**\n{model_references[0]}"

    prompt += f"\n\n**Model's Response**\n{model_outputs[0]}"

    if is_multi_turn:
        prompt += f"\n\n**Follow-up Question.**\n{model_questions[1]}"
        if model_references and model_references[1]:
            prompt += f"\n\n**Additional Reference**\n{model_references[1]}"
        prompt += f"\n\n**Model's Response**\n{model_outputs[1]}"

    prompt += "\n\n[[대화 종료. 평가 시작.]]"

    system_content = JUDGE_TEMPLATE["multi_turn" if is_multi_turn else "single_turn"]
    return system_content, prompt


def build_conversation(row, is_multi_turn: bool) -> List[Dict[str, str]]:
    system_content, user_content = build_prompt(row, is_multi_turn)
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def write_rows(output_file: Path, rows: List[Dict]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def read_rows(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def report_failures(label: str, failed: List[Tuple[int, str]], total: int) -> None:
    if not failed:
        print(f"- {total}/{total} judgements OK -> {label}")
        return
    listed = ", ".join(f"id={qid}/{turn}" for qid, turn in failed)
    print(f"! {len(failed)}/{total} judgements failed for {label} (written as unscored): {listed}")


def process_file(judge, file_path: Path, output_file: Path) -> int:
    """Judge one generated file. Returns the number of items that stayed unscored."""
    print(f"- 현재 Processing : {file_path}")
    df_model_outputs = pd.read_json(file_path, lines=True)
    rows = [row for _, row in df_model_outputs.iterrows()]

    conversations: List[List[Dict[str, str]]] = []
    meta: List[Tuple[int, str]] = []  # (row index, "single" | "multi")
    for idx, row in enumerate(rows):
        for turn_key, is_multi in (("single", False), ("multi", True)):
            conversations.append(build_conversation(row, is_multi))
            meta.append((idx, turn_key))

    judgements = judge.judge(conversations)

    failed: List[Tuple[int, str]] = []
    results: Dict[int, Dict[str, Dict]] = {}
    for (idx, turn_key), parsed in zip(meta, judgements):
        if parsed["judge_score"] is None:
            failed.append((int(rows[idx]["id"]), turn_key))
        results.setdefault(idx, {})[turn_key] = parsed

    out_rows = []
    for idx, row in enumerate(rows):
        row_dict = row.to_dict()
        row_dict["query_single"] = results[idx]["single"]
        row_dict["query_multi"] = results[idx]["multi"]
        out_rows.append(row_dict)
    write_rows(output_file, out_rows)

    report_failures(str(file_path), failed, len(conversations))
    return len(failed)


def unscored_targets(rows: List[Dict]) -> List[Tuple[int, str]]:
    """(row index, turn key) of every judgement recorded as unscored."""
    return [
        (idx, key)
        for idx, row in enumerate(rows)
        for key in ("query_single", "query_multi")
        if (row.get(key) or {}).get("judge_score") is None
    ]


def rejudge_file(judge, output_file: Path) -> int:
    """Re-judge only the unscored items of an evaluated file, in place.

    Deleting the file and starting over re-judges everything, which for a handful of
    rate-limited items means paying for ~250 judgements to recover 18 -- and rolling the
    dice on the ones that already succeeded.
    """
    rows = read_rows(output_file)
    targets = unscored_targets(rows)
    if not targets:
        print(f"이미 평가 완료(실패 항목 없음).. : {output_file}")
        return 0

    print(f"- 실패 항목만 재채점 : {output_file} ({len(targets)}개)")
    conversations = [build_conversation(rows[idx], key == "query_multi") for idx, key in targets]
    judgements = judge.judge(conversations)

    failed: List[Tuple[int, str]] = []
    for (idx, key), parsed in zip(targets, judgements):
        rows[idx][key] = parsed
        if parsed["judge_score"] is None:
            failed.append((int(rows[idx]["id"]), key.replace("query_", "")))
    write_rows(output_file, rows)

    report_failures(str(output_file), failed, len(targets))
    return len(failed)


def is_hidden(filepath: Path) -> bool:
    return any(part.startswith(".") for part in filepath.parts)


def resolve_output_dir(input_dir: Path) -> Path:
    """Mirror the model's path under ./evaluated so different models never collide.

    `-o generated/<model>/<variant>` used to flatten onto `evaluated/<file>.jsonl`, so
    every model wrote to the same three filenames and the "이미 평가 완료" skip below
    then silently reused the previous model's judgements.
    """
    parts = input_dir.resolve().parts
    if "generated" in parts:
        last = len(parts) - 1 - parts[::-1].index("generated")
        sub = Path(*parts[last + 1 :]) if parts[last + 1 :] else Path(input_dir.name)
    else:
        sub = Path(input_dir.name)
    return Path("./evaluated") / sub


def main():
    args = get_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = args.hf_token

    judge_model = resolve_judge_model(args.judge_model)
    print(f"- Judge model: {judge_model}")

    input_dir = Path(args.model_output_dir)
    output_dir = resolve_output_dir(input_dir)
    print(f"- Writing judgements to: {output_dir}")

    # Filter out hidden files
    json_files = [file for file in input_dir.rglob("*.jsonl") if not is_hidden(file)]
    print(f"Found {len(json_files)} JSON files to process")

    todo: List[Tuple[Optional[Path], Path]] = []  # (generated file | None, evaluated file)
    for file_path in json_files:
        output_file_path = output_dir / file_path.relative_to(input_dir)
        if not output_file_path.exists():
            todo.append((file_path, output_file_path))
        elif args.rejudge_failed:
            # Checking first costs one file read; not checking costs minutes of GPU time
            # loading a judge that turns out to have nothing to do.
            if unscored_targets(read_rows(output_file_path)):
                todo.append((None, output_file_path))
            else:
                print(f"이미 평가 완료(실패 항목 없음).. : {output_file_path}")
        else:
            print(
                f"이미 평가 완료.. : {file_path}"
                " (실패 항목만 다시 채점하려면 --rejudge-failed)"
            )

    if not todo:
        return

    # Loading a local judge takes minutes of GPU time, so only pay for it once there is
    # something to judge.
    judge = create_judge(args, judge_model)

    total_failures = 0
    for file_path, output_file_path in todo:
        if file_path is None:
            total_failures += rejudge_file(judge, output_file_path)
        else:
            total_failures += process_file(judge, file_path, output_file_path)

    if total_failures:
        print(
            f"\n! {total_failures} judgement(s) could not be scored and are excluded from the "
            "average. The reported score is based on the remaining items — rerun just those "
            "items with `evaluator.py --rejudge-failed` before trusting the number.",
            file=sys.stderr,
        )
        if not args.allow_partial:
            sys.exit(1)


if __name__ == "__main__":
    main()
