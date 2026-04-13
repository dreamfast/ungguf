"""Unit tests for mappings_qwen3.py — Qwen3 tensor name mapping.

Verifies GLOBAL_MAP, LAYER_SUFFIX_MAP, and gguf_name_to_hf() produce the
correct HuggingFace tensor names for every supported GGUF tensor.
"""

from __future__ import annotations

import pytest

from mappings_qwen3 import GLOBAL_MAP, LAYER_SUFFIX_MAP, gguf_name_to_hf


class TestGlobalMap:
    """Tests for the top-level (non-layer) tensor mappings."""

    @pytest.mark.parametrize(
        "gguf_name, expected_hf",
        list(GLOBAL_MAP.items()),
        ids=list(GLOBAL_MAP.keys()),
    )
    def test_global_map_entry(self, gguf_name: str, expected_hf: str):
        assert gguf_name_to_hf(gguf_name) == expected_hf

    def test_global_map_completeness(self):
        for gguf_name in GLOBAL_MAP:
            result = gguf_name_to_hf(gguf_name)
            assert result is not None, f"GLOBAL_MAP entry {gguf_name} returned None"


class TestLayerSuffixMap:
    """Tests for per-layer tensor mappings (blk.N.*)."""

    @pytest.mark.parametrize("layer", [0, 1, 15, 47])
    @pytest.mark.parametrize(
        "suffix",
        list(LAYER_SUFFIX_MAP.keys()),
        ids=list(LAYER_SUFFIX_MAP.keys()),
    )
    def test_layer_mapping(self, layer: int, suffix: str):
        gguf_name = f"blk.{layer}.{suffix}"
        expected = f"model.layers.{layer}.{LAYER_SUFFIX_MAP[suffix]}"
        assert gguf_name_to_hf(gguf_name) == expected

    def test_all_suffixes_are_tested(self):
        assert len(LAYER_SUFFIX_MAP) == 11


class TestEdgeCases:
    """Tests for unknown/invalid tensor names."""

    def test_unknown_global_name(self):
        assert gguf_name_to_hf("totally_fake_tensor") is None

    def test_unknown_layer_suffix(self):
        assert gguf_name_to_hf("blk.0.unknown.weight") is None

    def test_empty_string(self):
        assert gguf_name_to_hf("") is None

    def test_partial_layer_name(self):
        assert gguf_name_to_hf("blk.0.") is None


class TestMappingIntegrity:
    """Verify mapping properties (1:1, no duplicates)."""

    def test_no_duplicate_hf_targets(self):
        """Every GGUF name must map to a unique HF name (1:1 mapping)."""
        all_hf: list[str] = list(GLOBAL_MAP.values())
        for layer in range(3):
            all_hf.extend(
                f"model.layers.{layer}.{hf_suffix}" for hf_suffix in LAYER_SUFFIX_MAP.values()
            )
        assert len(all_hf) == len(set(all_hf)), "Duplicate HF target names found"
