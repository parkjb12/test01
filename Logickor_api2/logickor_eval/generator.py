import argparse
import os
# vLLM runtime options (must be set before importing vllm)
os.environ.setdefault("VLLM_USE_V1", "0")
os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")


import pandas as pd

from hf_compat import heterogeneous_config_overrides
from templates import PROMPT_STRATEGY

# Use aphrodite-engine or vLLM
try:
    from aphrodite import LLM, SamplingParams

    print("- Using aphrodite-engine")

except ImportError:
    from vllm import LLM, SamplingParams

    print("- Using vLLM")

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu_devices", help=" : CUDA_VISIBLE_DEVICES", default="0")
parser.add_argument(
    "-m",
    "--model",
    help=" : Model to evaluate",
    default="yanolja/EEVE-Korean-Instruct-2.8B-v1.0",
)
parser.add_argument("-ml", "--model_len", help=" : Maximum Model Length", default=4096, type=int)
parser.add_argument(
    "-mo",
    "--max-gen-tokens",
    dest="max_gen_tokens",
    help=" : Maximum tokens generated per answer",
    default=2048,
    type=int,
)
args = parser.parse_args()

print(f"Args - {args}")

# ⬇️추가한 코드
CONTEXT_MARGIN_TOKENS = 256
# Room to reserve for the fixed parts of a turn-2 prompt: the few-shot prefix plus both
# questions. The longest LogicKor question is ~1.3k characters and the 1-shot prefix
# ~1.4k, so ~2k tokens covers the worst case.
RESERVED_PROMPT_TOKENS = 2048

# LogicKor answers (coding / writing / math) routinely run past 512 tokens. The old hard
# cap of `min(512, model_len // 4)` cut them off mid-sentence -- one answer ended inside
# a URL -- and the judge then marked them down for being incomplete.
#
# The cap has to stay self-consistent with the turn-2 prompt, which carries the whole
# turn-1 answer: model_len must fit (prefix + q1 + answer1 + q2) + answer2 + margin,
# i.e. RESERVED + 2*G + margin <= model_len. Solving for G keeps a long answer from
# crowding out the context it is fed back into.
MAX_GENERATION_TOKENS = max(
    256,
    min(
        args.max_gen_tokens,
        (args.model_len - CONTEXT_MARGIN_TOKENS - RESERVED_PROMPT_TOKENS) // 2,
    ),
)
PROMPT_TRIM_STEP_TOKENS = 128
MIN_QUESTION_TOKENS = 64
MIN_CARRYOVER_TOKENS = 64
PROMPT_TOKEN_BUDGET = max(
    RESERVED_PROMPT_TOKENS, args.model_len - MAX_GENERATION_TOKENS - CONTEXT_MARGIN_TOKENS
)
print(
    f"- Budgets: prompt<={PROMPT_TOKEN_BUDGET} tokens, generation<={MAX_GENERATION_TOKENS} tokens"
    f" (model_len={args.model_len}). Raise --model_len for longer answers."
)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
gpu_counts = len(args.gpu_devices.split(","))

llm = LLM(
    model=args.model,
    tensor_parallel_size=gpu_counts,
    max_model_len=args.model_len,
    gpu_memory_utilization=0.8,
    # ⬇️추가한 코드
    trust_remote_code=True,
    attention_config={"backend": "triton_attn"},
    enforce_eager=True,
    compilation_config={"mode": 0},  # NONE
    # Needed for models with per-layer (heterogeneous) HF configs, e.g. gemma-4-E4B.
    hf_overrides=heterogeneous_config_overrides,
)

sampling_params = SamplingParams(
    temperature=0,
    skip_special_tokens=True,
    # max_tokens=args.model_len,
    # ⬇️추가한 코드
    max_tokens=MAX_GENERATION_TOKENS,
    stop=["<|endoftext|>", "[INST]", "[/INST]", "<|im_end|>", "<|end|>", "<|eot_id|>", "<end_of_turn>", "<eos>"],
)

# ⬇️추가한 코드
base_dir = os.path.dirname(os.path.abspath(__file__))
questions_path = os.path.join(base_dir, "questions.jsonl")
df_questions = pd.read_json(questions_path, orient="records", encoding="utf-8-sig", lines=True)

# df_questions = pd.read_json("questions.jsonl", orient="records", encoding="utf-8-sig", lines=True)

if not os.path.exists("./generated/" + args.model):
    os.makedirs("./generated/" + args.model)

# ⬇️추가한 코드
def resolve_chat_tokenizer(llm_instance):
    tokenizer = None
    if hasattr(llm_instance, "get_tokenizer"):
        try:
            tokenizer = llm_instance.get_tokenizer()
        except Exception:
            tokenizer = None

    if tokenizer is None:
        tokenizer = getattr(getattr(llm_instance, "llm_engine", None), "tokenizer", None)

    current = tokenizer
    for _ in range(4):
        if current is None:
            break
        if hasattr(current, "apply_chat_template"):
            return current
        current = getattr(current, "tokenizer", None)

    raise RuntimeError("Failed to resolve a tokenizer with apply_chat_template().")


# ⬇️추가한 코드
chat_tokenizer = resolve_chat_tokenizer(llm)


# ⬇️추가한 코드
def encode_text(text):
    try:
        return chat_tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        return chat_tokenizer.encode(text)


# ⬇️추가한 코드
def decode_text(token_ids):
    try:
        return chat_tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        return chat_tokenizer.decode(token_ids)


# ⬇️추가한 코드
def token_len(text):
    return len(encode_text(text))


# ⬇️추가한 코드
def trim_tail_tokens(text, trim_step_tokens, min_tokens):
    ids = encode_text(text)
    if len(ids) <= min_tokens:
        return text
    cut_to = max(min_tokens, len(ids) - trim_step_tokens)
    return decode_text(ids[:cut_to]).strip()


def finalize_output_text(text):
    """Normalise whitespace only.

    This used to trim the answer back to the last sentence-ending mark to hide the
    512-token truncation, but that deleted real content (code blocks and URLs end on
    no such mark, and `www.example.` looks like a sentence end). Answers are no longer
    truncated, so nothing is dropped here.
    """
    return text.strip()


for strategy_name, prompts in PROMPT_STRATEGY.items():

    # llm.llm_engine.tokenizer.tokenizer.apply_chat_template(...) 직접 접근은
    # vLLM/transformers 조합에 따라 실패할 수 있어 유지하지 않음.
    def format_single_turn_question(question):
        question_1 = question[0]

        # ⬇️추가한 코드
        def build_prompt(q1):
            return chat_tokenizer.apply_chat_template(
                prompts + [{"role": "user", "content": q1}],
                tokenize=False,
                add_generation_prompt=True,
            )

        # ⬇️추가한 코드
        prompt = build_prompt(question_1)
        while token_len(prompt) > PROMPT_TOKEN_BUDGET:
            next_question_1 = trim_tail_tokens(
                question_1,
                trim_step_tokens=PROMPT_TRIM_STEP_TOKENS,
                min_tokens=MIN_QUESTION_TOKENS,
            )
            if next_question_1 == question_1:
                break
            question_1 = next_question_1
            prompt = build_prompt(question_1)

        # return llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
        #     prompts + [{"role": "user", "content": question[0]}],
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        # ⬇️추가한 코드
        return prompt

    single_turn_questions = df_questions["questions"].map(format_single_turn_question)
    print(single_turn_questions.iloc[0])
    # single_turn_outputs = [
    #     output.outputs[0].text.strip() for output in llm.generate(single_turn_questions, sampling_params)
    # ]
    # ⬇️추가한 코드
    single_turn_outputs = [
        finalize_output_text(output.outputs[0].text) for output in llm.generate(single_turn_questions, sampling_params)
    ]

    def format_double_turn_question(question, single_turn_output):
        question_1 = question[0]
        question_2 = question[1]
        carry_output = single_turn_output

        # ⬇️추가한 코드
        def build_prompt(q1, carry, q2):
            return chat_tokenizer.apply_chat_template(
                prompts
                + [
                    {"role": "user", "content": q1},
                    {"role": "assistant", "content": carry},
                    {"role": "user", "content": q2},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

        # ⬇️추가한 코드
        prompt = build_prompt(question_1, carry_output, question_2)
        while token_len(prompt) > PROMPT_TOKEN_BUDGET:
            next_carry = trim_tail_tokens(
                carry_output,
                trim_step_tokens=PROMPT_TRIM_STEP_TOKENS,
                min_tokens=MIN_CARRYOVER_TOKENS,
            )
            if next_carry == carry_output:
                break
            carry_output = next_carry
            prompt = build_prompt(question_1, carry_output, question_2)

        # ⬇️추가한 코드
        while token_len(prompt) > PROMPT_TOKEN_BUDGET:
            next_question_1 = trim_tail_tokens(
                question_1,
                trim_step_tokens=PROMPT_TRIM_STEP_TOKENS,
                min_tokens=MIN_QUESTION_TOKENS,
            )
            if next_question_1 == question_1:
                break
            question_1 = next_question_1
            prompt = build_prompt(question_1, carry_output, question_2)

        # ⬇️추가한 코드
        while token_len(prompt) > PROMPT_TOKEN_BUDGET:
            next_question_2 = trim_tail_tokens(
                question_2,
                trim_step_tokens=PROMPT_TRIM_STEP_TOKENS,
                min_tokens=MIN_QUESTION_TOKENS,
            )
            if next_question_2 == question_2:
                break
            question_2 = next_question_2
            prompt = build_prompt(question_1, carry_output, question_2)

        # ⬇️추가한 코드
        if token_len(prompt) > PROMPT_TOKEN_BUDGET:
            fallback_question_2 = question_2
            fallback_prompt = chat_tokenizer.apply_chat_template(
                prompts + [{"role": "user", "content": fallback_question_2}],
                tokenize=False,
                add_generation_prompt=True,
            )
            while token_len(fallback_prompt) > PROMPT_TOKEN_BUDGET:
                next_fallback_question_2 = trim_tail_tokens(
                    fallback_question_2,
                    trim_step_tokens=PROMPT_TRIM_STEP_TOKENS,
                    min_tokens=MIN_QUESTION_TOKENS,
                )
                if next_fallback_question_2 == fallback_question_2:
                    break
                fallback_question_2 = next_fallback_question_2
                fallback_prompt = chat_tokenizer.apply_chat_template(
                    prompts + [{"role": "user", "content": fallback_question_2}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            return fallback_prompt

        # return llm.llm_engine.tokenizer.tokenizer.apply_chat_template(
        #     prompts
        #     + [
        #         {"role": "user", "content": question[0]},
        #         {"role": "assistant", "content": single_turn_output},
        #         {"role": "user", "content": question[1]},
        #     ],
        #     tokenize=False,
        #     add_generation_prompt=True,
        # )
        # ⬇️추가한 코드
        return prompt

    multi_turn_questions = [
        format_double_turn_question(questions, single_turn_outputs[row_idx])
        for row_idx, questions in enumerate(df_questions["questions"])
    ]
    # multi_turn_outputs = [
    #     output.outputs[0].text.strip() for output in llm.generate(multi_turn_questions, sampling_params)
    # ]
    # ⬇️추가한 코드
    multi_turn_outputs = [
        finalize_output_text(output.outputs[0].text) for output in llm.generate(multi_turn_questions, sampling_params)
    ]

    df_output = pd.DataFrame(
        {
            "id": df_questions["id"],
            "category": df_questions["category"],
            "questions": df_questions["questions"],
            "outputs": list(zip(single_turn_outputs, multi_turn_outputs)),
            "references": df_questions["references"],
        }
    )
    df_output.to_json(
        "./generated/" + os.path.join(args.model, f"{strategy_name}.jsonl"),
        orient="records",
        lines=True,
        force_ascii=False,
    )
