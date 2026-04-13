"""Unit tests for verify_sha256.py — SHA256 fingerprint utilities.

Covers:
  - canonical() triple-prefix name normalization
  - is_lm_key() language model key filtering
  - tensor_bytes() BF16 byte representation
"""

from __future__ import annotations

import torch

from verify_sha256 import canonical, is_lm_key, tensor_bytes


class TestCanonical:
    def test_triple_prefix_normalized(self):
        """Triple model.language_model prefix should be reduced to single."""
        name = "model.language_model.language_model.language_model.layers.0.weight"
        result = canonical(name)
        assert result == "model.language_model.layers.0.weight"
        assert result.count("model.language_model") == 1

    def test_single_prefix_unchanged(self):
        """Already-normalized names should pass through."""
        name = "model.language_model.layers.0.self_attn.q_proj.weight"
        assert canonical(name) == name

    def test_no_prefix_unchanged(self):
        """Names without the prefix should pass through."""
        name = "model.embed_tokens.weight"
        assert canonical(name) == name

    def test_partial_prefix_unchanged(self):
        """Double prefix (not triple) should pass through."""
        name = "model.language_model.language_model.layers.0.weight"
        assert canonical(name) == name


class TestIsLmKey:
    def test_language_model_key(self):
        assert (
            is_lm_key(
                "model.language_model.layers.0.weight", "model.language_model.layers.0.weight"
            )
            is True
        )

    def test_visual_excluded(self):
        assert (
            is_lm_key("visual.encoder.weight", "model.language_model.visual.encoder.weight")
            is False
        )

    def test_mtp_excluded(self):
        assert is_lm_key("model.layers.47.mtp.weight", "model.language_model.mtp.weight") is False

    def test_no_language_model_excluded(self):
        assert is_lm_key("model.embed_tokens.weight", "model.embed_tokens.weight") is False

    def test_language_model_with_visual_excluded(self):
        """Key with both language_model AND visual should be excluded."""
        assert (
            is_lm_key(
                "model.language_model.visual.weight",
                "model.language_model.visual.weight",
            )
            is False
        )

    def test_language_model_with_mtp_excluded(self):
        """Key with both language_model AND mtp should be excluded."""
        assert (
            is_lm_key(
                "model.language_model.layers.47.mtp_proj.weight",
                "model.language_model.layers.47.mtp_proj.weight",
            )
            is False
        )


class TestTensorBytes:
    def test_returns_bytes(self):
        """tensor_bytes should return a bytes object."""
        t = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        result = tensor_bytes(t)
        assert isinstance(result, bytes)

    def test_deterministic(self):
        """Same tensor should always produce the same bytes."""
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        b1 = tensor_bytes(t)
        b2 = tensor_bytes(t)
        assert b1 == b2

    def test_bf16_normalization(self):
        """F32 tensor should be normalized to BF16 before hashing."""
        t_f32 = torch.tensor([1.0, 2.0], dtype=torch.float32)
        t_bf16 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        assert tensor_bytes(t_f32) == tensor_bytes(t_bf16)

    def test_different_values_different_bytes(self):
        """Different tensor values should produce different bytes."""
        t1 = torch.tensor([1.0], dtype=torch.bfloat16)
        t2 = torch.tensor([2.0], dtype=torch.bfloat16)
        assert tensor_bytes(t1) != tensor_bytes(t2)
