"""Shared tensor name mappings, conventions, and transforms for Qwen3.6-35B-A3B MoE.

Qwen3.6-35B-A3B is a Mixture-of-Experts variant using Qwen3_5MoeForConditionalGeneration.
It shares the same hybrid linear-attention + full-attention layer structure as the dense
Qwen3.5/3.6 models, but replaces the standard MLP with a MoE block containing:
  - 256 routed experts (stored stacked in GGUF as ffn_*_exps tensors)
  - 1 shared expert (ffn_*_shexp tensors)
  - A router gate (ffn_gate_inp)
  - A shared expert gate (ffn_gate_inp_shexp)

Key differences from the dense model mappings (mappings_qwen35.py):
  - MoE MLP tensors replace standard FFN tensors
  - Expert tensors need concatenation (gate+up -> gate_up_proj) and permutation
  - Shared expert projections need transposition (GGUF [H,I] vs HF [I,H])
  - Router gate needs transposition
  - GLOBAL_MAP maps output.weight to lm_head.weight (no language_model namespace)
  - V-head reorder logic is imported from mappings_qwen35 (shared architecture)
  - V-head config: 16 K-heads / 32 V-heads -> v_per_k=2
"""

from __future__ import annotations

import re

import torch

# Import shared V-head reorder logic — identical to dense Qwen3.5 architecture
from mappings_qwen35 import (
    apply_inverse_v_reorder,
    get_linear_attn_config,
)

__all__ = [
    "apply_conventions",
    "apply_inverse_v_reorder",
    "build_expert_tensors",
    "get_expert_layer",
    "get_expert_suffix",
    "get_linear_attn_config",
    "gguf_name_to_hf",
    "is_missing_tensor",
    "resolve_hf_name",
]

# ---------------------------------------------------------------------------
# Global tensors (non-layer-specific)
# ---------------------------------------------------------------------------

GLOBAL_MAP = {
    "token_embd.weight": "model.language_model.embed_tokens.weight",
    "output_norm.weight": "model.language_model.norm.weight",
    "output.weight": "lm_head.weight",
}

# ---------------------------------------------------------------------------
# Per-layer suffix mapping (attention + linear attention, same as dense model)
# ---------------------------------------------------------------------------

LAYER_SUFFIX_MAP = {
    # Attention norms (all layers)
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    # Linear attention (DeltaNet layers)
    "ssm_a": "linear_attn.A_log",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
    # Full attention (transformer layers)
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    # MoE shared expert (needs transposition via apply_conventions)
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    # MoE router and shared expert gate
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
}

# MoE expert tensor GGUF suffixes — handled specially in the converter
# (concat gate+up, permute dims for HF layout)
EXPERT_SUFFIXES = frozenset(
    {
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
    }
)

# ---------------------------------------------------------------------------
# Norm convention suffixes (GGUF stores +1.0, HF expects raw values)
# ---------------------------------------------------------------------------

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

# NOTE: 2D transposition is auto-detected by the converter based on shape
# comparison (GGUF stores weights in F-order [in,out], HF uses C-order [out,in]).
# No explicit suffix list needed — the converter checks if the decoded shape
# is the reverse of the reference shape and transposes accordingly.

# ---------------------------------------------------------------------------
# Tensor name prefixes absent from GGUF — copied verbatim from reference
# ---------------------------------------------------------------------------

MISSING_PREFIXES: tuple[str, ...] = ("mtp.", "model.visual.")


def is_missing_tensor(hf_name: str) -> bool:
    """Check if a tensor belongs to MTP or vision modules (absent from GGUF)."""
    return any(hf_name.startswith(p) for p in MISSING_PREFIXES)


def gguf_name_to_hf(gguf_name: str) -> str | None:
    """Map a GGUF tensor name to its HuggingFace equivalent for Qwen3.6 MoE.

    Returns None for expert tensors (ffn_*_exps) which are handled separately.
    """
    if gguf_name in GLOBAL_MAP:
        return GLOBAL_MAP[gguf_name]
    m = re.match(r"blk\.(\d+)\.(.*)", gguf_name)
    if m:
        layer_num = m.group(1)
        suffix = m.group(2)
        # Skip expert suffixes — they need special handling
        if suffix in EXPERT_SUFFIXES:
            return None
        if suffix in LAYER_SUFFIX_MAP:
            hf_suffix = LAYER_SUFFIX_MAP[suffix]
            return f"model.language_model.layers.{layer_num}.{hf_suffix}"
    return None


def resolve_hf_name(hf_name: str, ref_shapes: dict[str, list[int]]) -> str | None:
    """Try multiple HF name variations to find a match in reference shapes.

    Handles the case where lm_head may or may not be under model.language_model.
    """
    for candidate in (
        hf_name,
        hf_name.replace("model.language_model.", "model."),
        hf_name.replace("model.language_model.", ""),
    ):
        if candidate in ref_shapes:
            return candidate
    return None


def get_expert_layer(gguf_name: str) -> int | None:
    """Extract the layer number from an expert GGUF tensor name.

    Returns None if the name is not an expert tensor.
    """
    if not any(gguf_name.endswith(s) for s in EXPERT_SUFFIXES):
        return None
    m = re.match(r"blk\.(\d+)\.ffn_", gguf_name)
    if m:
        return int(m.group(1))
    return None


def get_expert_suffix(gguf_name: str) -> str | None:
    """Extract the expert suffix (e.g. 'ffn_gate_exps.weight') from a GGUF name."""
    for s in EXPERT_SUFFIXES:
        if gguf_name.endswith(s):
            return s
    return None


def apply_conventions(hf_name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Apply Qwen3.6-specific VALUE conventions to a tensor.

    This function handles value-only transforms — it does NOT perform structural
    transforms (transpose/reshape). Shape alignment is handled by the converter
    based on auto-detection of GGUF→HF layout differences.

    - Norm weights: GGUF stores +1.0, HF expects raw (subtract 1.0)
    - A_log: GGUF stores -A, HF expects log(-A)
    """
    if hf_name.endswith(NORM_SUFFIXES) or hf_name in FINAL_NORM_NAMES:
        orig_dtype = tensor.dtype
        tensor = (tensor.float() - 1.0).to(orig_dtype)

    if hf_name.endswith("linear_attn.A_log"):
        tensor = torch.log(-tensor.float())

    return tensor


def build_expert_tensors(
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build HF expert tensors from decoded GGUF expert tensors.

    GGUF stores expert tensors in GGML/Fortran order (column-major).
    When decoded with reverse_shape=True, the shape is already in HF layout:
      gate: [num_experts, intermediate_size, hidden_size]
      up:   [num_experts, intermediate_size, hidden_size]
      down: [num_experts, hidden_size, intermediate_size]

    HF layout (Qwen3_5MoeExperts):
      gate_up_proj: [num_experts, intermediate_size*2, hidden_size]
      down_proj:    [num_experts, hidden_size, intermediate_size]

    No permutation is needed — just concatenate gate+up on dim=1.

    Returns (gate_up_proj, down_proj).
    """
    # Concatenate gate and up along intermediate dimension — already in [E, I, H]
    gate_up = torch.cat([gate, up], dim=1).contiguous()  # [E, I*2, H]

    # Down projection is already in [E, H, I]
    down_proj = down.contiguous()

    return gate_up, down_proj
