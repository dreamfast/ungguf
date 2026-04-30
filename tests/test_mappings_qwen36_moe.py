"""Unit tests for mappings_qwen36_moe.py — Qwen3.6-35B-A3B MoE tensor mappings.

Covers:
  - GLOBAL_MAP / LAYER_SUFFIX_MAP name mapping
  - EXPERT_SUFFIXES handling (returns None from gguf_name_to_hf)
  - get_expert_layer() / get_expert_suffix() helpers
  - resolve_hf_name() with namespace variants
  - is_missing_tensor() for MTP/vision detection
  - apply_conventions() norm-1.0 and A_log value transforms
  - build_expert_tensors() shape correctness and value integrity
  - get_linear_attn_config() (imported from qwen35 — re-verify for MoE config)
"""

from __future__ import annotations

import pytest
import torch

from mappings_qwen36_moe import (
    EXPERT_SUFFIXES,
    FINAL_NORM_NAMES,
    GLOBAL_MAP,
    LAYER_SUFFIX_MAP,
    NORM_SUFFIXES,
    apply_conventions,
    apply_inverse_v_reorder,
    build_expert_tensors,
    get_expert_layer,
    get_expert_suffix,
    get_linear_attn_config,
    gguf_name_to_hf,
    is_missing_tensor,
    resolve_hf_name,
)


# ---------------------------------------------------------------------------
# Name mapping tests
# ---------------------------------------------------------------------------
class TestGlobalMap:
    @pytest.mark.parametrize(
        "gguf_name, expected_hf",
        list(GLOBAL_MAP.items()),
        ids=list(GLOBAL_MAP.keys()),
    )
    def test_global_map_entry(self, gguf_name: str, expected_hf: str):
        assert gguf_name_to_hf(gguf_name) == expected_hf

    def test_lm_head_direct(self):
        """output.weight maps to lm_head.weight (not model.language_model.lm_head.weight)."""
        assert gguf_name_to_hf("output.weight") == "lm_head.weight"


class TestLayerSuffixMap:
    @pytest.mark.parametrize("layer", [0, 5, 20, 39])
    @pytest.mark.parametrize(
        "suffix",
        list(LAYER_SUFFIX_MAP.keys()),
        ids=list(LAYER_SUFFIX_MAP.keys()),
    )
    def test_layer_mapping(self, layer: int, suffix: str):
        gguf_name = f"blk.{layer}.{suffix}"
        expected = f"model.language_model.layers.{layer}.{LAYER_SUFFIX_MAP[suffix]}"
        assert gguf_name_to_hf(gguf_name) == expected

    def test_unknown_suffix_returns_none(self):
        assert gguf_name_to_hf("blk.0.unknown.weight") is None

    def test_non_layer_name_returns_none(self):
        assert gguf_name_to_hf("totally_fake") is None

    def test_expert_suffix_returns_none(self):
        """Expert suffixes should return None from gguf_name_to_hf."""
        for suffix in EXPERT_SUFFIXES:
            assert gguf_name_to_hf(f"blk.0.{suffix}") is None

    def test_empty_string_returns_none(self):
        assert gguf_name_to_hf("") is None


# ---------------------------------------------------------------------------
# Expert tensor helper tests
# ---------------------------------------------------------------------------
class TestExpertHelpers:
    @pytest.mark.parametrize("layer", [0, 19, 39])
    def test_get_expert_layer_valid(self, layer: int):
        for suffix in EXPERT_SUFFIXES:
            assert get_expert_layer(f"blk.{layer}.{suffix}") == layer

    def test_get_expert_layer_invalid(self):
        assert get_expert_layer("blk.0.attn_q.weight") is None
        assert get_expert_layer("output.weight") is None
        assert get_expert_layer("blk.0.ssm_a") is None
        assert get_expert_layer("") is None

    @pytest.mark.parametrize(
        "suffix",
        list(EXPERT_SUFFIXES),
        ids=list(EXPERT_SUFFIXES),
    )
    def test_get_expert_suffix_valid(self, suffix: str):
        assert get_expert_suffix(f"blk.5.{suffix}") == suffix

    def test_get_expert_suffix_invalid(self):
        assert get_expert_suffix("blk.0.attn_q.weight") is None
        assert get_expert_suffix("blk.0.ffn_gate_shexp.weight") is None
        assert get_expert_suffix("") is None

    def test_all_expert_suffixes_recognized(self):
        """Verify the complete set of expected expert suffixes."""
        expected = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"}
        assert expected == EXPERT_SUFFIXES


# ---------------------------------------------------------------------------
# resolve_hf_name tests
# ---------------------------------------------------------------------------
class TestResolveHfName:
    def test_direct_match(self):
        shapes = {"lm_head.weight": [248320, 2048]}
        assert resolve_hf_name("lm_head.weight", shapes) == "lm_head.weight"

    def test_strip_language_model_prefix(self):
        shapes = {"model.layers.0.q.weight": [512, 2048]}
        assert (
            resolve_hf_name("model.language_model.layers.0.q.weight", shapes)
            == "model.layers.0.q.weight"
        )

    def test_strip_model_prefix(self):
        shapes = {"layers.0.q.weight": [512, 2048]}
        assert (
            resolve_hf_name("model.language_model.layers.0.q.weight", shapes)
            == "layers.0.q.weight"
        )

    def test_no_match_returns_none(self):
        shapes = {"other.weight": [10, 10]}
        assert resolve_hf_name("missing.weight", shapes) is None

    def test_empty_shapes_returns_none(self):
        assert resolve_hf_name("any.weight", {}) is None


# ---------------------------------------------------------------------------
# is_missing_tensor tests
# ---------------------------------------------------------------------------
class TestIsMissingTensor:
    def test_mtp_prefix(self):
        assert is_missing_tensor("mtp.fc.weight") is True
        assert is_missing_tensor("mtp.layers.0.self_attn.q.weight") is True

    def test_visual_prefix(self):
        assert is_missing_tensor("model.visual.blocks.0.attn.weight") is True
        assert is_missing_tensor("model.visual.merger.weight") is True

    def test_language_model_not_missing(self):
        assert is_missing_tensor("model.language_model.layers.0.self_attn.q.weight") is False

    def test_lm_head_not_missing(self):
        assert is_missing_tensor("lm_head.weight") is False

    def test_empty_string_not_missing(self):
        assert is_missing_tensor("") is False


# ---------------------------------------------------------------------------
# get_linear_attn_config tests (imported from qwen35, verify with MoE config)
# ---------------------------------------------------------------------------
class TestGetLinearAttnConfig:
    def _make_moe_config(
        self,
        num_k: int = 16,
        num_v: int = 32,
        head_k: int = 128,
        head_v: int = 128,
    ) -> dict:
        return {
            "text_config": {
                "linear_num_key_heads": num_k,
                "linear_num_value_heads": num_v,
                "linear_key_head_dim": head_k,
                "linear_value_head_dim": head_v,
            }
        }

    def test_moe_config_v_per_k_2(self):
        """Qwen3.6-35B-A3B MoE: 16 K-heads / 32 V-heads → v_per_k=2."""
        cfg = get_linear_attn_config(self._make_moe_config())
        assert cfg is not None
        assert cfg["v_per_k"] == 2
        assert cfg["num_k_heads"] == 16
        assert cfg["num_v_heads"] == 32
        assert cfg["key_dim"] == 2048
        assert cfg["value_dim"] == 4096

    def test_returns_none_when_k_eq_v(self):
        cfg = get_linear_attn_config(self._make_moe_config(num_k=16, num_v=16))
        assert cfg is None

    def test_returns_none_missing_keys(self):
        assert get_linear_attn_config({"text_config": {}}) is None


# ---------------------------------------------------------------------------
# apply_conventions tests
# ---------------------------------------------------------------------------
class TestApplyConventions:
    def test_norm_weight_subtracts_one(self):
        for suffix in NORM_SUFFIXES:
            hf_name = f"model.language_model.layers.0.{suffix}"
            tensor = torch.ones(64, dtype=torch.bfloat16) * 2.0
            result = apply_conventions(hf_name, tensor)
            assert torch.allclose(result.float(), torch.ones(64)), f"Failed for {suffix}"

    def test_final_norm_subtracts_one(self):
        for hf_name in FINAL_NORM_NAMES:
            tensor = torch.ones(64, dtype=torch.bfloat16) * 3.0
            result = apply_conventions(hf_name, tensor)
            assert torch.allclose(result.float(), torch.ones(64) * 2.0), f"Failed for {hf_name}"

    def test_non_norm_non_transpose_passthrough(self):
        hf_name = "model.language_model.layers.0.self_attn.q_proj.weight"
        tensor = torch.randn(32, 16, dtype=torch.bfloat16)
        result = apply_conventions(hf_name, tensor)
        assert torch.equal(result, tensor)

    def test_a_log_transform(self):
        hf_name = "model.language_model.layers.0.linear_attn.A_log"
        tensor = torch.tensor([-1.0, -2.0, -4.0], dtype=torch.bfloat16)
        result = apply_conventions(hf_name, tensor)
        expected = torch.log(-tensor.float())
        assert torch.allclose(result.float(), expected, atol=1e-2)

    def test_preserves_dtype_for_norm(self):
        hf_name = "model.language_model.layers.0.input_layernorm.weight"
        tensor = torch.ones(64, dtype=torch.bfloat16) * 2.0
        result = apply_conventions(hf_name, tensor)
        assert result.dtype == torch.bfloat16

    def test_no_structural_transform(self):
        """apply_conventions only does value transforms; no transpose/reshape."""
        hf_name = "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight"
        tensor = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3)
        result = apply_conventions(hf_name, tensor)
        assert result.shape == tensor.shape
        assert torch.equal(result, tensor)


# ---------------------------------------------------------------------------
# build_expert_tensors tests
# ---------------------------------------------------------------------------
class TestBuildExpertTensors:
    """Test the expert tensor concatenation logic.

    GGUF expert tensors decoded with reverse_shape=True (GGML/Fortran → C-order):
      gate: [E, I, H], up: [E, I, H], down: [E, H, I]
    HF layout: gate_up [E, I*2, H], down [E, H, I]
    """

    @pytest.fixture
    def small_experts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create small deterministic expert tensors in decoded [E, I, H] layout."""
        n_experts, inter, hidden = 3, 2, 4
        gate = torch.arange(n_experts * inter * hidden, dtype=torch.bfloat16).reshape(
            n_experts, inter, hidden
        )
        up = (
            torch.arange(n_experts * inter * hidden, dtype=torch.bfloat16).reshape(
                n_experts, inter, hidden
            )
            + 100
        )
        down = torch.arange(n_experts * hidden * inter, dtype=torch.bfloat16).reshape(
            n_experts, hidden, inter
        )
        return gate, up, down

    def test_output_shapes(self, small_experts):
        gate, up, down = small_experts
        gate_up, down_proj = build_expert_tensors(gate, up, down)
        n_experts, inter, hidden = 3, 2, 4
        assert gate_up.shape == (n_experts, inter * 2, hidden)
        assert down_proj.shape == (n_experts, hidden, inter)

    def test_gate_up_concatenation(self, small_experts):
        """gate_up should contain gate followed by up along the intermediate dim."""
        gate, up, _ = small_experts
        gate_up, _ = build_expert_tensors(gate, up, small_experts[2])
        _n_exp, inter_x2, _hidden = gate_up.shape
        inter = inter_x2 // 2
        # Expert 0: gate_up[0] should be [gate[0], up[0]] along dim=0
        e = 0
        # No permutation — tensors are already in [E, I, H]
        assert torch.equal(gate_up[e, :inter, :], gate[e])
        assert torch.equal(gate_up[e, inter:, :], up[e])

    def test_down_proj_values(self, small_experts):
        """down_proj should equal the down tensor directly (no permutation)."""
        _, _, down = small_experts
        _, down_proj = build_expert_tensors(*small_experts)
        # No permutation — down is already in [E, H, I]
        assert torch.equal(down_proj, down)

    def test_full_pipeline_shapes(self):
        """Test with realistic dimensions matching Qwen3.6-35B-A3B."""
        n_experts, inter, hidden = 256, 512, 2048
        gate = torch.randn(n_experts, inter, hidden, dtype=torch.bfloat16)
        up = torch.randn(n_experts, inter, hidden, dtype=torch.bfloat16)
        down = torch.randn(n_experts, hidden, inter, dtype=torch.bfloat16)
        gate_up, down_proj = build_expert_tensors(gate, up, down)
        assert gate_up.shape == (n_experts, inter * 2, hidden)
        assert down_proj.shape == (n_experts, hidden, inter)

    def test_output_contiguous(self, small_experts):
        gate_up, down_proj = build_expert_tensors(*small_experts)
        assert gate_up.is_contiguous()
        assert down_proj.is_contiguous()

    def test_dtype_preserved(self, small_experts):
        gate_up, down_proj = build_expert_tensors(*small_experts)
        assert gate_up.dtype == torch.bfloat16
        assert down_proj.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Integration: no duplicate HF targets
# ---------------------------------------------------------------------------
class TestMappingIntegrity:
    def test_no_duplicate_hf_targets(self):
        """All GLOBAL_MAP + LAYER_SUFFIX_MAP values should produce unique HF targets."""
        targets = set(GLOBAL_MAP.values())
        for suffix, hf_suffix in LAYER_SUFFIX_MAP.items():
            for layer in range(4):
                target = f"model.language_model.layers.{layer}.{hf_suffix}"
                assert target not in targets, f"Duplicate target: {target} (from suffix {suffix})"
                targets.add(target)

    def test_expert_suffixes_not_in_layer_map(self):
        """Expert suffixes should NOT appear in LAYER_SUFFIX_MAP."""
        for suffix in EXPERT_SUFFIXES:
            assert suffix not in LAYER_SUFFIX_MAP, f"{suffix} should not be in LAYER_SUFFIX_MAP"


# ---------------------------------------------------------------------------
# apply_inverse_v_reorder tests (imported from qwen35, verify for MoE params)
# ---------------------------------------------------------------------------
class TestApplyInverseVReorder:
    """Verify V-head reorder works correctly with MoE model's v_per_k=2 config."""

    @pytest.fixture
    def moe_la_cfg(self) -> dict:
        return {
            "num_k_heads": 16,
            "num_v_heads": 32,
            "head_k_dim": 128,
            "head_v_dim": 128,
            "v_per_k": 2,
            "key_dim": 2048,
            "value_dim": 4096,
        }

    def test_in_proj_z_shape_preserved(self, moe_la_cfg: dict):
        """in_proj_z reorder should preserve tensor shape."""
        value_dim = moe_la_cfg["value_dim"]
        cols = 2048
        tensor = torch.arange(value_dim * cols, dtype=torch.float32).reshape(value_dim, cols)
        result = apply_inverse_v_reorder(
            tensor, "model.language_model.layers.0.linear_attn.in_proj_z.weight", moe_la_cfg
        )
        assert result.shape == tensor.shape

    def test_in_proj_qkv_qk_unchanged(self, moe_la_cfg: dict):
        """in_proj_qkv should only reorder the V portion; Q+K should be unchanged."""
        key_dim = moe_la_cfg["key_dim"]
        value_dim = moe_la_cfg["value_dim"]
        total = key_dim * 2 + value_dim
        cols = 128
        tensor = torch.arange(total * cols, dtype=torch.float32).reshape(total, cols)
        result = apply_inverse_v_reorder(
            tensor, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", moe_la_cfg
        )
        assert torch.equal(result[: key_dim * 2], tensor[: key_dim * 2])
        assert result.shape == tensor.shape

    def test_passthrough_non_linear_attn(self, moe_la_cfg: dict):
        tensor = torch.randn(32, 16, dtype=torch.float32)
        result = apply_inverse_v_reorder(
            tensor, "model.language_model.layers.0.self_attn.q_proj.weight", moe_la_cfg
        )
        assert torch.equal(result, tensor)
