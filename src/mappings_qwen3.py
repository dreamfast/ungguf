"""Shared tensor name mappings and helpers for Qwen3 architecture.

Used by both the converter (gguf_to_safetensors_qwen3.py) and the
verifier (verify_conversion_qwen3.py) to ensure mapping consistency.
"""

from __future__ import annotations

import re

GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
}

LAYER_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}


def gguf_name_to_hf(name: str) -> str | None:
    """Map a GGUF tensor name to its HuggingFace equivalent for Qwen3."""
    if name in GLOBAL_MAP:
        return GLOBAL_MAP[name]
    m = re.match(r"blk\.(\d+)\.(.*)", name)
    if m:
        layer, suffix = m.group(1), m.group(2)
        if suffix in LAYER_SUFFIX_MAP:
            return f"model.layers.{layer}.{LAYER_SUFFIX_MAP[suffix]}"
    return None
