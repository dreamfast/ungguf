"""Unit tests for mappings_glm47.py — GLM-4.7-Flash tensor name mapping and kv_b reconstruction.

Covers:
  - Architecture constants (NUM_LAYERS, NUM_EXPERTS, etc.)
  - GLOBAL_MAP, LAYER_SUFFIX_MAP, DENSE_MLP_MAP, MOE_SHARED_MAP, MOE_GATE_MAP
  - EXPERT_SUFFIXES, EXPERT_HF_MAP, MTP_TENSOR_NAMES
  - reconstruct_kv_b() interleaved weight reconstruction
"""

from __future__ import annotations

import pytest
import torch

from mappings_glm47 import (
    DENSE_MLP_MAP,
    EXPERT_HF_MAP,
    EXPERT_SUFFIXES,
    GLOBAL_MAP,
    KV_B_SUFFIXES,
    KV_LORA_RANK,
    LAYER_SUFFIX_MAP,
    MOE_GATE_MAP,
    MOE_SHARED_MAP,
    MTP_TENSOR_NAMES,
    NUM_ATTENTION_HEADS,
    NUM_EXPERTS,
    NUM_LAYERS,
    Q_LORA_RANK,
    QK_NOPE_HEAD_DIM,
    QK_ROPE_HEAD_DIM,
    V_HEAD_DIM,
    reconstruct_kv_b,
)


# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
class TestConstants:
    def test_num_layers(self):
        assert NUM_LAYERS == 48

    def test_num_experts(self):
        assert NUM_EXPERTS == 64

    def test_num_attention_heads(self):
        assert NUM_ATTENTION_HEADS == 20

    def test_qk_nope_head_dim(self):
        assert QK_NOPE_HEAD_DIM == 192

    def test_qk_rope_head_dim(self):
        assert QK_ROPE_HEAD_DIM == 64

    def test_v_head_dim(self):
        assert V_HEAD_DIM == 256

    def test_kv_lora_rank(self):
        assert KV_LORA_RANK == 512

    def test_q_lora_rank(self):
        assert Q_LORA_RANK == 768


# ---------------------------------------------------------------------------
# Map entry verification
# ---------------------------------------------------------------------------
class TestMaps:
    @pytest.mark.parametrize(
        "gguf_name, expected_hf",
        list(GLOBAL_MAP.items()),
        ids=list(GLOBAL_MAP.keys()),
    )
    def test_global_map(self, gguf_name: str, expected_hf: str):
        assert GLOBAL_MAP[gguf_name] == expected_hf

    @pytest.mark.parametrize("layer", [0, 5, 47])
    @pytest.mark.parametrize(
        "suffix",
        list(LAYER_SUFFIX_MAP.keys()),
        ids=list(LAYER_SUFFIX_MAP.keys()),
    )
    def test_layer_suffix_map_values_are_valid(self, layer: int, suffix: str):
        """Each LAYER_SUFFIX_MAP value should be a non-empty HF-style dotted path."""
        hf_suffix = LAYER_SUFFIX_MAP[suffix]
        assert isinstance(hf_suffix, str) and len(hf_suffix) > 0
        assert "." in hf_suffix, f"HF suffix '{hf_suffix}' should be a dotted path"

    def test_layer_suffix_map_spot_check(self):
        """Spot-check specific well-known GGUF→HF suffix mappings."""
        assert LAYER_SUFFIX_MAP["attn_norm.weight"] == "input_layernorm.weight"
        assert LAYER_SUFFIX_MAP["ffn_norm.weight"] == "post_attention_layernorm.weight"
        assert LAYER_SUFFIX_MAP["attn_output.weight"] == "self_attn.o_proj.weight"
        assert LAYER_SUFFIX_MAP["attn_q_a.weight"] == "self_attn.q_a_proj.weight"
        assert LAYER_SUFFIX_MAP["attn_kv_a_mqa.weight"] == "self_attn.kv_a_proj_with_mqa.weight"

    def test_dense_mlp_map_entries(self):
        assert "ffn_gate.weight" in DENSE_MLP_MAP
        assert "ffn_up.weight" in DENSE_MLP_MAP
        assert "ffn_down.weight" in DENSE_MLP_MAP

    def test_moe_shared_map_entries(self):
        assert "ffn_gate_shexp.weight" in MOE_SHARED_MAP
        assert "ffn_up_shexp.weight" in MOE_SHARED_MAP
        assert "ffn_down_shexp.weight" in MOE_SHARED_MAP

    def test_moe_gate_map_entries(self):
        assert "ffn_gate_inp.weight" in MOE_GATE_MAP
        assert "exp_probs_b.bias" in MOE_GATE_MAP

    def test_expert_suffixes(self):
        assert {
            "ffn_gate_exps.weight",
            "ffn_up_exps.weight",
            "ffn_down_exps.weight",
        } == EXPERT_SUFFIXES

    def test_expert_hf_map(self):
        assert EXPERT_HF_MAP["ffn_gate_exps.weight"] == "gate_proj.weight"
        assert EXPERT_HF_MAP["ffn_up_exps.weight"] == "up_proj.weight"
        assert EXPERT_HF_MAP["ffn_down_exps.weight"] == "down_proj.weight"

    def test_kv_b_suffixes(self):
        assert {"attn_k_b.weight", "attn_v_b.weight"} == KV_B_SUFFIXES

    def test_mtp_tensor_names(self):
        assert len(MTP_TENSOR_NAMES) == 6
        for name in MTP_TENSOR_NAMES:
            assert name.startswith("model.layers.47."), f"MTP tensor {name} not in layer 47"

    def test_no_duplicate_hf_targets(self):
        """All maps combined should not produce duplicate HF names."""
        all_hf: list[str] = list(GLOBAL_MAP.values())
        for layer in range(3):
            all_hf.extend(
                f"model.layers.{layer}.{hf_suffix}" for hf_suffix in LAYER_SUFFIX_MAP.values()
            )
            all_hf.extend(
                f"model.layers.{layer}.{hf_suffix}" for hf_suffix in DENSE_MLP_MAP.values()
            )
            all_hf.extend(
                f"model.layers.{layer}.{hf_suffix}" for hf_suffix in MOE_SHARED_MAP.values()
            )
            all_hf.extend(
                f"model.layers.{layer}.{hf_suffix}" for hf_suffix in MOE_GATE_MAP.values()
            )
        assert len(all_hf) == len(set(all_hf)), "Duplicate HF target names found across maps"


# ---------------------------------------------------------------------------
# reconstruct_kv_b tests
# ---------------------------------------------------------------------------
class TestReconstructKvB:
    def test_output_shape(self):
        """Output shape should be (num_heads * (qk_nope + v_head), kv_lora_rank)."""
        k_b = torch.randn(NUM_ATTENTION_HEADS, KV_LORA_RANK, QK_NOPE_HEAD_DIM)
        v_b = torch.randn(NUM_ATTENTION_HEADS, V_HEAD_DIM, KV_LORA_RANK)
        result = reconstruct_kv_b(k_b, v_b)
        expected_rows = NUM_ATTENTION_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM)
        assert result.shape == (expected_rows, KV_LORA_RANK)

    def test_interleaving_pattern(self):
        """k_b and v_b should be interleaved per head with known values."""
        num_heads = 2
        qk_dim = 3
        v_dim = 2
        kv_rank = 4

        # Create distinguishable k_b and v_b
        k_b = torch.zeros(num_heads, kv_rank, qk_dim)
        v_b = torch.zeros(num_heads, v_dim, kv_rank)
        for h in range(num_heads):
            k_b[h] = (h + 1) * 10  # Head 0: all 10s, Head 1: all 20s
            v_b[h] = (h + 1) * 100  # Head 0: all 100s, Head 1: all 200s

        result = reconstruct_kv_b(
            k_b,
            v_b,
            num_heads=num_heads,
            qk_nope_head_dim=qk_dim,
            v_head_dim=v_dim,
            kv_lora_rank=kv_rank,
        )

        # Head 0: rows 0:3 = k_b[0].T (all 10s), rows 3:5 = v_b[0] (all 100s)
        assert torch.all(result[0:3] == 10), f"Expected 10s, got {result[0:3]}"
        assert torch.all(result[3:5] == 100), f"Expected 100s, got {result[3:5]}"

        # Head 1: rows 5:8 = k_b[1].T (all 20s), rows 8:10 = v_b[1] (all 200s)
        assert torch.all(result[5:8] == 20), f"Expected 20s, got {result[5:8]}"
        assert torch.all(result[8:10] == 200), f"Expected 200s, got {result[8:10]}"

    def test_k_b_transposed(self):
        """k_b should be transposed in the output (k_b[h].T).

        k_b has shape (num_heads, kv_lora_rank, qk_nope_head_dim).
        In the output, k_b[h].T has shape (qk_nope_head_dim, kv_lora_rank).
        v_b has shape (num_heads, v_head_dim, kv_lora_rank).
        In the output, v_b[h] keeps shape (v_head_dim, kv_lora_rank).
        """
        num_heads = 1
        qk_dim = 3
        v_dim = 1
        kv_rank = 2

        # k_b shape: (1, kv_rank=2, qk_dim=3)
        k_b = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        # v_b shape: (1, v_dim=1, kv_rank=2)
        v_b = torch.tensor([[[7.0, 8.0]]])

        result = reconstruct_kv_b(
            k_b,
            v_b,
            num_heads=num_heads,
            qk_nope_head_dim=qk_dim,
            v_head_dim=v_dim,
            kv_lora_rank=kv_rank,
        )

        # k_b[0].T = shape (3, 2): [[1,4],[2,5],[3,6]]
        expected_k_t = k_b[0].T
        assert torch.equal(result[0:3], expected_k_t)
        # v_b[0] = shape (1, 2): [[7, 8]]
        assert torch.equal(result[3:4], v_b[0])

    def test_custom_params(self):
        """Should work with non-default architecture parameters."""
        k_b = torch.randn(4, 64, 32)
        v_b = torch.randn(4, 16, 64)
        result = reconstruct_kv_b(
            k_b, v_b, num_heads=4, qk_nope_head_dim=32, v_head_dim=16, kv_lora_rank=64
        )
        assert result.shape == (4 * (32 + 16), 64)  # (192, 64)

    def test_output_dtype_matches_input(self):
        """Output dtype should match k_b dtype."""
        k_b = torch.randn(2, 8, 4, dtype=torch.bfloat16)
        v_b = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        result = reconstruct_kv_b(
            k_b, v_b, num_heads=2, qk_nope_head_dim=4, v_head_dim=4, kv_lora_rank=8
        )
        assert result.dtype == torch.bfloat16
