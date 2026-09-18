from typing import Any


def make_training_chat_template(model_name: str | None = None) -> str:
    # TRL assistant-only loss expects {% generation %} markers in assistant spans.
    if model_name and "EXAONE" in model_name.upper():
        return (
            "{% for message in messages %}"
            "{% if loop.first and message['role'] != 'system' %}"
            "{{ '[|system|][|endofturn|]\n' }}"
            "{% endif %}"
            "{% if message['role'] == 'system' %}"
            "{{ '[|system|]' + message['content'] }}{{ eos_token }}\n"
            "{% elif message['role'] == 'user' %}"
            "{{ '[|user|]' + message['content'] }}\n"
            "{% elif message['role'] == 'assistant' %}"
            "{{ '[|assistant|]' }}{% generation %}{{ message['content'] }}{% endgeneration %}{{ eos_token }}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '[|assistant|]' }}{% endif %}"
        )
    return (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|system|>\n{{ message['content'] }}{{ eos_token }}\n"
        "{% elif message['role'] == 'user' %}"
        "<|user|>\n{{ message['content'] }}{{ eos_token }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "<|assistant|>\n{% generation %}{{ message['content'] }}{% endgeneration %}{{ eos_token }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )


def _template_holders(obj: Any) -> list:
    """chat_template 을 들고 있는 객체들을 모은다.

    AutoProcessor 는 멀티모달 모델에서는 Processor(내부에 .tokenizer 보유)를,
    텍스트 전용 모델(Llama 등)에서는 토크나이저 자체를 돌려준다. 두 경우 모두
    save_pretrained() 가 chat_template 을 기록하므로, 저장 전에 전부 같은
    템플릿으로 맞춰야 학습/추론 프롬프트 형식이 어긋나지 않는다.
    """
    holders = []
    current = obj
    for _ in range(4):
        if current is None or any(h is current for h in holders):
            break
        if hasattr(current, "chat_template"):
            holders.append(current)
        current = getattr(current, "tokenizer", None)
    return holders


def ensure_training_chat_template(tokenizer: Any, model_name: str | None = None) -> None:
    """Replace non-training-compatible templates with a stable training template.

    tokenizer 자리에는 토크나이저뿐 아니라 AutoProcessor 결과도 넘길 수 있다.
    학습에 쓴 템플릿이 merged/ 에 그대로 저장되어야 generator 가 같은 형식으로
    프롬프트를 만든다.
    """
    if tokenizer is None:
        return
    template = make_training_chat_template(model_name)
    for holder in _template_holders(tokenizer):
        holder.chat_template = template
