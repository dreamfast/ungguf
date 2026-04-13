"""Unit tests for mappings_qwen35.py — Qwen3.5 tensor name mapping, conventions, and V-head reorder.

Covers:
  - GLOBAL_MAP / LAYER_SUFFIX_MAP name mapping
  - get_linear_attn_config() config extraction
  - apply_conventions() norm-1.0 and A_log transforms
  - apply_inverse_v_reorder() V-head layout reorder
"""

from __future__ import annotations

import pytest
import torch

from mappings_qwen35 import (
    FINAL_NORM_NAMES,
    GLOBAL_MAP,
    LAYER_SUFFIX_MAP,
    NORM_SUFFIXES,
    apply_conventions,
    apply_inverse_v_reorder,
    get_linear_attn_config,
    gguf_name_to_hf,
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


class TestLayerSuffixMap:
    @pytest.mark.parametrize("layer", [0, 5, 20])
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


# ---------------------------------------------------------------------------
# get_linear_attn_config tests
# ---------------------------------------------------------------------------
class TestGetLinearAttnConfig:
    def _make_config(
        self,
        num_k: int = 4,
        num_v: int = 8,
        head_k: int = 32,
        head_v: int = 32,
    ) -> dict:
        return {
            "text_config": {
                "linear_num_key_heads": num_k,
                "linear_num_value_heads": num_v,
                "linear_key_head_dim": head_k,
                "linear_value_head_dim": head_v,
            }
        }

    def test_returns_config_when_v_gt_k(self):
        cfg = get_linear_attn_config(self._make_config(num_k=4, num_v=8))
        assert cfg is not None
        assert cfg["v_per_k"] == 2
        assert cfg["num_k_heads"] == 4
        assert cfg["num_v_heads"] == 8
        assert cfg["key_dim"] == 128
        assert cfg["value_dim"] == 256

    def test_returns_none_when_k_eq_v(self):
        cfg = get_linear_attn_config(self._make_config(num_k=4, num_v=4))
        assert cfg is None

    def test_returns_none_when_k_gt_v(self):
        cfg = get_linear_attn_config(self._make_config(num_k=8, num_v=4))
        assert cfg is None

    def test_returns_none_missing_keys(self):
        assert get_linear_attn_config({"text_config": {}}) is None
        assert get_linear_attn_config({}) is None

    def test_handles_flat_config(self):
        """Config without text_config key should fall back to top-level."""
        flat = {
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 6,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
        }
        cfg = get_linear_attn_config(flat)
        assert cfg is not None
        assert cfg["v_per_k"] == 3


# ---------------------------------------------------------------------------
# apply_conventions tests
# ---------------------------------------------------------------------------
class TestApplyConventions:
    def test_norm_weight_subtracts_one(self):
        """Norm tensors should have 1.0 subtracted."""
        for suffix in NORM_SUFFIXES:
            hf_name = f"model.language_model.layers.0.{suffix}"
            tensor = torch.ones(64, dtype=torch.bfloat16) * 2.0
            result = apply_conventions(hf_name, tensor)
            assert torch.allclose(result.float(), torch.ones(64)), f"Failed for {suffix}"

    def test_final_norm_subtracts_one(self):
        """Final norm tensors should have 1.0 subtracted."""
        for hf_name in FINAL_NORM_NAMES:
            tensor = torch.ones(64, dtype=torch.bfloat16) * 3.0
            result = apply_conventions(hf_name, tensor)
            assert torch.allclose(result.float(), torch.ones(64) * 2.0), f"Failed for {hf_name}"

    def test_non_norm_passthrough(self):
        """Non-norm tensors should pass through unchanged."""
        hf_name = "model.language_model.layers.0.self_attn.q_proj.weight"
        tensor = torch.randn(32, 16, dtype=torch.bfloat16)
        result = apply_conventions(hf_name, tensor)
        assert torch.equal(result, tensor)

    def test_a_log_transform(self):
        """A_log tensors should get log(-x) transform."""
        hf_name = "model.language_model.layers.0.linear_attn.A_log"
        tensor = torch.tensor([-1.0, -2.0, -4.0], dtype=torch.bfloat16)
        result = apply_conventions(hf_name, tensor)
        expected = torch.log(-tensor.float())
        assert torch.allclose(result.float(), expected, atol=1e-2)

    def test_preserves_dtype(self):
        """Convention transforms should preserve the original dtype."""
        hf_name = "model.language_model.layers.0.input_layernorm.weight"
        tensor = torch.ones(64, dtype=torch.bfloat16) * 2.0
        result = apply_conventions(hf_name, tensor)
        assert result.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# apply_inverse_v_reorder tests
# ---------------------------------------------------------------------------
class TestApplyInverseVReorder:
    @pytest.fixture
    def la_cfg(self) -> dict:
        return {
            "num_k_heads": 2,
            "num_v_heads": 4,
            "head_k_dim": 16,
            "head_v_dim": 16,
            "v_per_k": 2,
            "key_dim": 32,
            "value_dim": 64,
        }

    def test_in_proj_qkv_reorders_v_only(self, la_cfg: dict):
        """in_proj_qkv should reorder only the V portion (after Q+K)."""
        key_dim = la_cfg["key_dim"]  # 32
        value_dim = la_cfg["value_dim"]  # 64
        total_rows = key_dim * 2 + value_dim  # 128
        # Use tensor size divisible by v_per_k * k_heads for V reorder
        tensor = torch.arange(total_rows * 8, dtype=torch.float32).reshape(total_rows, 8)
        result = apply_inverse_v_reorder(
            tensor, "model.language_model.layers.0.linear_attn.in_proj_qkv.weight", la_cfg
        )
        # Q+K portion should be unchanged
        assert torch.equal(result[: key_dim * 2], tensor[: key_dim * 2])
        # V portion shape should be the same
        assert result.shape == tensor.shape

    def test_in_proj_z_reorder(self, la_cfg: dict):
        """in_proj_z should be fully reordered — verify permutation with known values.

        With v_per_k=2, k_heads=2, head_v=16: the reorder reshapes dim 0 from
        [v_per_k=2, k_heads=2, head_v=16] and swaps the first two axes to
        [k_heads=2, v_per_k=2, head_v=16], then flattens back.
        """
        v_per_k, k_heads, head_v = la_cfg["v_per_k"], la_cfg["num_k_heads"], la_cfg["head_v_dim"]
        cols = 8
        tensor = torch.arange(la_cfg["value_dim"] * cols, dtype=torch.float32).reshape(
            la_cfg["value_dim"], cols
        )
        result = apply_inverse_v_reorder(
            tensor, "model.language_model.layers.0.linear_attn.in_proj_z.weight", la_cfg
        )
        assert result.shape == tensor.shape
        # Manually compute expected: reshape → permute → flatten
        expected = (
            tensor.reshape(v_per_k, k_heads, head_v, cols)
            .permute(1, 0, 2, 3)
            .contiguous()
            .reshape(la_cfg["value_dim"], cols)
        )
        assert torch.equal(result, expected)

    def test_a_b_reorder(self, la_cfg: dict):
        """in_proj_a and in_proj_b should be reordered with head_dim=1.

        With v_per_k=2, k_heads=2, head_dim=1: dim 0 has 4 elements.
        Reorder: [v0_k0, v0_k1, v1_k0, v1_k1] → [v0_k0, v1_k0, v0_k1, v1_k1]
        (swap v_per_k and k_heads axes).
        """
        for suffix in ["in_proj_a.weight", "in_proj_b.weight"]:
            hf_name = f"model.language_model.layers.0.linear_attn.{suffix}"
            cols = 8
            tensor = torch.arange(4 * cols, dtype=torch.float32).reshape(4, cols)
            result = apply_inverse_v_reorder(tensor, hf_name, la_cfg)
            assert result.shape == tensor.shape
            # Manually compute: reshape(v_per_k=2, k_heads=2, 1, cols) → permute(1,0,2,3) → flatten
            expected = (
                tensor.reshape(2, 2, 1, cols).permute(1, 0, 2, 3).contiguous().reshape(4, cols)
            )
            assert torch.equal(result, expected), f"Value mismatch for {suffix}"

    def test_out_proj_reorder_dim1(self, la_cfg: dict):
        """out_proj should reorder along dim=1, verified with deterministic values."""
        v_per_k, k_heads, head_v = la_cfg["v_per_k"], la_cfg["num_k_heads"], la_cfg["head_v_dim"]
        rows = 8
        tensor = torch.arange(rows * la_cfg["value_dim"], dtype=torch.float32).reshape(
            rows, la_cfg["value_dim"]
        )
        result = apply_inverse_v_reorder(
            tensor, "model.language_model.layers.0.linear_attn.out_proj.weight", la_cfg
        )
        assert result.shape == tensor.shape
        # dim=1 reorder: reshape(rows, v_per_k, k_heads, head_v) → permute(0,2,1,3) → flatten
        expected = (
            tensor.reshape(rows, v_per_k, k_heads, head_v)
            .permute(0, 2, 1, 3)
            .contiguous()
            .reshape(rows, la_cfg["value_dim"])
        )
        assert torch.equal(result, expected)

    def test_conv1d_reorder(self, la_cfg: dict):
        """conv1d should reorder the V portion, verified with deterministic values."""
        key_dim = la_cfg["key_dim"]
        v_per_k, k_heads, head_v = la_cfg["v_per_k"], la_cfg["num_k_heads"], la_cfg["head_v_dim"]
        total = key_dim * 2 + la_cfg["value_dim"]
        cols = 3
        tensor = torch.arange(total * cols, dtype=torch.float32).reshape(total, cols)
        result = apply_inverse_v_reorder(
            tensor, "model.language_model.layers.0.linear_attn.conv1d.weight", la_cfg
        )
        assert result.shape == tensor.shape
        # Q+K portion unchanged
        assert torch.equal(result[: key_dim * 2], tensor[: key_dim * 2])
        # V portion reordered: reshape(v_per_k, k_heads, head_v, cols) → permute(1,0,2,3)
        v_in = tensor[key_dim * 2 :]
        v_expected = (
            v_in.reshape(v_per_k, k_heads, head_v, cols)
            .permute(1, 0, 2, 3)
            .contiguous()
            .reshape(la_cfg["value_dim"], cols)
        )
        assert torch.equal(result[key_dim * 2 :], v_expected)

    def test_dt_bias_reorder(self, la_cfg: dict):
        """dt_bias should be reordered (1D tensor), verified with deterministic values.

        Uses unsqueeze→reorder→squeeze path. With v_per_k=2, k_heads=2, head_dim=1:
        input [a, b, c, d] → unsqueeze → [[a, b, c, d]]
        reshape(1, v_per_k=2, k_heads=2, 1) → permute(0, 2, 1, 3) → squeeze
        Result: [a, c, b, d]
        """
        tensor = torch.tensor([10.0, 20.0, 30.0, 40.0], dtype=torch.float32)
        result = apply_inverse_v_reorder(
            tensor, "model.language_model.layers.0.linear_attn.dt_bias", la_cfg
        )
        assert result.shape == tensor.shape
        expected = torch.tensor([10.0, 30.0, 20.0, 40.0], dtype=torch.float32)
        assert torch.equal(result, expected)

    def test_passthrough_non_linear_attn(self, la_cfg: dict):
        """Non-linear_attn tensors should pass through unchanged."""
        hf_name = "model.language_model.layers.0.self_attn.q_proj.weight"
        tensor = torch.randn(32, 16, dtype=torch.float32)
        result = apply_inverse_v_reorder(tensor, hf_name, la_cfg)
        assert torch.equal(result, tensor)
