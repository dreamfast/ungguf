"""Shared tensor name mappings, conventions, and transforms for Qwen3.5 architecture.

Used by both the converter (gguf_to_safetensors_qwen35.py) and the
verifier (verify_conversion_qwen35.py) to ensure mapping consistency.

Qwen3.5 is a hybrid Mamba2 + Transformer architecture. Key differences from
standard GGUF-to-HF conversion:
  - reverse_shape=False: GGUF stores Qwen3.5 tensors in C-order (row-major)
  - Norm convention: GGUF stores norm weights +1.0, HF expects raw values
  - A_log convention: GGUF stores -A, HF expects log(-A)
  - V-head reorder: when num_v_heads > num_k_heads, inverse reorder is needed
"""

from __future__ import annotations

import re

import torch

GLOBAL_MAP = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output_norm.weight": "model.language_model.norm.weight",
    "output.weight": "model.language_model.lm_head.weight",
}

LAYER_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ssm_a": "linear_attn.A_log",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
}

NORM_SUFFIXES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "q_norm.weight",
    "k_norm.weight",
)

FINAL_NORM_NAMES = {
    "model.language_model.norm.weight",
    "model.norm.weight",
    "norm.weight",
}


def gguf_name_to_hf(gguf_name: str) -> str | None:
    """Map a GGUF tensor name to its HuggingFace equivalent for Qwen3.5."""
    if gguf_name in GLOBAL_MAP:
        return GLOBAL_MAP[gguf_name]
    m = re.match(r"blk\.(\d+)\.(.*)", gguf_name)
    if m:
        layer_num = m.group(1)
        suffix = m.group(2)
        if suffix in LAYER_SUFFIX_MAP:
            hf_suffix = LAYER_SUFFIX_MAP[suffix]
            return f"model.language_model.layers.{layer_num}.{hf_suffix}"
    return None


def get_linear_attn_config(config: dict) -> dict | None:
    """Extract linear attention V-head config from model config.json.

    Returns None if V-head reorder is not needed (num_k == num_v).
    """
    tc = config.get("text_config", config)
    num_k = tc.get("linear_num_key_heads")
    num_v = tc.get("linear_num_value_heads")
    head_k = tc.get("linear_key_head_dim")
    head_v = tc.get("linear_value_head_dim")
    if num_k is None or num_v is None or head_k is None or head_v is None:
        return None
    if num_v <= num_k:
        return None
    return {
        "num_k_heads": num_k,
        "num_v_heads": num_v,
        "head_k_dim": head_k,
        "head_v_dim": head_v,
        "v_per_k": num_v // num_k,
        "key_dim": num_k * head_k,
        "value_dim": num_v * head_v,
    }


def _inverse_v_reorder(tensor, dim, v_per_k, k_heads, head_dim):
    """Undo GGUF's interleaved V-head layout to produce grouped HF layout."""
    shape = list(tensor.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = [*shape[:dim], v_per_k, k_heads, head_dim, *shape[dim + 1 :]]
    t = tensor.reshape(*new_shape)
    perm = list(range(len(new_shape)))
    perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
    return t.permute(*perm).contiguous().reshape(*shape)


def apply_inverse_v_reorder(tensor, hf_name, la_cfg):  # noqa: PLR0911
    """Apply inverse V-head reorder to a tensor based on its HF name."""
    v_per_k = la_cfg["v_per_k"]
    k_heads = la_cfg["num_k_heads"]
    head_v = la_cfg["head_v_dim"]
    key_dim = la_cfg["key_dim"]

    if hf_name.endswith("linear_attn.in_proj_qkv.weight"):
        qk = tensor[: key_dim * 2]
        v = tensor[key_dim * 2 :]
        v = _inverse_v_reorder(v, 0, v_per_k, k_heads, head_v)
        return tensor.new_empty(key_dim * 2 + v.shape[0], tensor.shape[1]).copy_(
            torch.cat([qk, v], dim=0)
        )

    if hf_name.endswith("linear_attn.in_proj_z.weight"):
        return _inverse_v_reorder(tensor, 0, v_per_k, k_heads, head_v)

    if hf_name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
        return _inverse_v_reorder(tensor, 0, v_per_k, k_heads, 1)

    if hf_name.endswith("linear_attn.A_log") or hf_name.endswith("linear_attn.dt_bias"):
        return _inverse_v_reorder(tensor.unsqueeze(0), 1, v_per_k, k_heads, 1).squeeze(0)

    if hf_name.endswith("linear_attn.conv1d.weight"):
        qk = tensor[: key_dim * 2]
        v = tensor[key_dim * 2 :]
        v = _inverse_v_reorder(v, 0, v_per_k, k_heads, head_v)
        return torch.cat([qk, v], dim=0)

    if hf_name.endswith("linear_attn.out_proj.weight"):
        return _inverse_v_reorder(tensor, 1, v_per_k, k_heads, head_v)

    return tensor


def apply_conventions(hf_name, tensor):
    """Apply Qwen3.5-specific convention transforms to a tensor.

    - Norm weights: GGUF stores +1.0, HF expects raw (subtract 1.0)
    - A_log: GGUF stores -A, HF expects log(-A)
    """
    if hf_name.endswith(NORM_SUFFIXES) or hf_name in FINAL_NORM_NAMES:
        orig_dtype = tensor.dtype
        tensor = (tensor.float() - 1.0).to(orig_dtype)

    if hf_name.endswith("linear_attn.A_log"):
        tensor = torch.log(-tensor.float())

    return tensor
