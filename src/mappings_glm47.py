"""Shared tensor name mappings, constants, and helpers for GLM-4.7-Flash architecture.

Used by both the converter (gguf_to_safetensors_glm47.py) and the
verifier (verify_conversion_glm47.py) to ensure mapping consistency.

GLM-4.7-Flash uses the DeepSeek2 architecture with MLA attention and MoE.
Key features:
  - kv_b_proj reconstruction from separate k_b + v_b tensors
  - 3D expert tensor splitting into per-expert slices
  - Hard-coded architecture constants for GLM-4.7-Flash
"""

from __future__ import annotations

import torch

# --- Architecture constants (GLM-4.7-Flash specific) ---
NUM_LAYERS = 48
NUM_EXPERTS = 64
NUM_ATTENTION_HEADS = 20
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 256
KV_LORA_RANK = 512
Q_LORA_RANK = 768

# --- Global tensor mappings ---
GLOBAL_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output.weight": "lm_head.weight",
    "output_norm.weight": "model.norm.weight",
}

LAYER_SUFFIX_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_a.weight": "self_attn.q_a_proj.weight",
    "attn_q_a_norm.weight": "self_attn.q_a_layernorm.weight",
    "attn_q_b.weight": "self_attn.q_b_proj.weight",
    "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "attn_kv_a_norm.weight": "self_attn.kv_a_layernorm.weight",
}

DENSE_MLP_MAP = {
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}

MOE_SHARED_MAP = {
    "ffn_gate_shexp.weight": "mlp.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_experts.down_proj.weight",
}

MOE_GATE_MAP = {
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.e_score_correction_bias",
}

KV_B_SUFFIXES = {"attn_k_b.weight", "attn_v_b.weight"}

EXPERT_SUFFIXES = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}

EXPERT_HF_MAP = {
    "ffn_gate_exps.weight": "gate_proj.weight",
    "ffn_up_exps.weight": "up_proj.weight",
    "ffn_down_exps.weight": "down_proj.weight",
}

MTP_TENSOR_NAMES = [
    "model.layers.47.eh_proj.weight",
    "model.layers.47.embed_tokens.weight",
    "model.layers.47.enorm.weight",
    "model.layers.47.hnorm.weight",
    "model.layers.47.shared_head.head.weight",
    "model.layers.47.shared_head.norm.weight",
]


def reconstruct_kv_b(
    k_b: torch.Tensor,
    v_b: torch.Tensor,
    num_heads: int = NUM_ATTENTION_HEADS,
    qk_nope_head_dim: int = QK_NOPE_HEAD_DIM,
    v_head_dim: int = V_HEAD_DIM,
    kv_lora_rank: int = KV_LORA_RANK,
) -> torch.Tensor:
    """Reconstruct the full kv_b_proj weight from separate k_b and v_b tensors.

    GGUF stores k_b and v_b separately; HuggingFace combines them into a
    single kv_b_proj weight with interleaved per-head layout.
    """
    per_head_rows = qk_nope_head_dim + v_head_dim
    total_rows = num_heads * per_head_rows
    kv_b_proj = torch.zeros(total_rows, kv_lora_rank, dtype=k_b.dtype, device=k_b.device)

    for h in range(num_heads):
        k_part = k_b[h].T
        v_part = v_b[h]
        offset = h * per_head_rows
        kv_b_proj[offset : offset + qk_nope_head_dim] = k_part
        kv_b_proj[offset + qk_nope_head_dim : offset + per_head_rows] = v_part

    return kv_b_proj
