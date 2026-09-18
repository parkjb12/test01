"""Compatibility shims between newer `transformers` configs and vLLM.

transformers >= 5.6 models such as `google/gemma-4-E4B-it` describe layers that
differ from each other (here: the full-attention layers use a larger head_dim)
with a *heterogeneous* config. Two things then break vLLM, which still assumes a
homogeneous config:

1. Reading a per-layer attribute off the global config raises
   `AmbiguousGlobalPerLayerAttributeError` (`config.head_dim` in
   `vllm/transformers_utils/model_arch_config_convertor.py`).
2. Values that used to be plain attributes are folded into the per-layer
   overrides and disappear, e.g. Gemma 4's `global_head_dim`. vLLM's Gemma 4
   implementation still looks for it and would silently fall back to the small
   head_dim for full-attention layers.

`heterogeneous_config_overrides` re-exposes both, so it can be handed to vLLM as
`LLM(..., hf_overrides=heterogeneous_config_overrides)`.
"""

from typing import Any

# Sub-configs that may carry their own per-layer heterogeneity.
_SUB_CONFIG_KEYS = ("text_config", "vision_config", "audio_config")


def _iter_configs(config: Any):
    yield config
    for key in _SUB_CONFIG_KEYS:
        sub = getattr(config, key, None)
        if sub is not None:
            yield sub


def _restore_global_head_dim(config: Any) -> None:
    """Re-expose Gemma 4's `global_head_dim` (head_dim of full-attention layers)."""
    if getattr(config, "global_head_dim", None):
        return

    layer_types = getattr(config, "layer_types", None) or []
    head_dims = {
        config.per_layer_config[i].head_dim
        for i, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    }
    if len(head_dims) == 1:
        config.global_head_dim = head_dims.pop()


def heterogeneous_config_overrides(config: Any) -> Any:
    """Make a heterogeneous HF config readable by vLLM. Used as `hf_overrides`."""
    for cfg in _iter_configs(config):
        if not getattr(cfg, "is_heterogeneous", False):
            continue
        # Allow `cfg.<per-layer attr>` to return the global value again.
        cfg.allow_global_per_layer_attribute_access = True
        _restore_global_head_dim(cfg)
    return config
