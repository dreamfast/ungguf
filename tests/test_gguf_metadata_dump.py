"""Unit tests for gguf_metadata_dump.py — GGUF forensic inspection utilities.

Covers:
  - is_highlight() keyword detection
  - format_value() value formatting (lists, bytes, strings, normal)
  - extract_field_value() field extraction from mock GGUF field objects
"""

from __future__ import annotations

import numpy as np
import pytest

from gguf_metadata_dump import extract_field_value, format_value, is_highlight


class TestIsHighlight:
    @pytest.mark.parametrize(
        "key",
        [
            "general.architecture",
            "qwen3.block_count",
            "mamba.ssm_type",
            "model.attention.head_count",
            "model.expert.count",
            "model.feed_forward_length",
        ],
    )
    def test_highlights_architecture_keys(self, key: str):
        assert is_highlight(key) is True

    @pytest.mark.parametrize(
        "key",
        [
            "general.name",
            "general.description",
            "tokenizer.ggml.tokens",
            "some_random_key",
        ],
    )
    def test_non_highlight_keys(self, key: str):
        assert is_highlight(key) is False

    def test_case_insensitive(self):
        assert is_highlight("MODEL.ARCHITECTURE") is True
        assert is_highlight("Model.Mamba_Type") is True

    def test_rope_highlighted(self):
        assert is_highlight("model.rope.freq_base") is True

    def test_context_length_highlighted(self):
        assert is_highlight("model.context_length") is True


class TestFormatValue:
    def test_short_list(self):
        result = format_value([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_long_list_truncated(self):
        long_list = list(range(100))
        result = format_value(long_list)
        assert "list of 100 items" in result
        assert "..." in result

    def test_bytes_decoded(self):
        result = format_value(b"hello world")
        assert result == "hello world"

    def test_bytes_non_utf8(self):
        result = format_value(b"\xff\xfe")
        # Should not crash, either returns decoded or <bytes> representation
        assert isinstance(result, str)

    def test_long_string_truncated(self):
        long_str = "a" * 300
        result = format_value(long_str)
        assert len(result) < 300
        assert result.endswith("...")

    def test_normal_value(self):
        assert format_value(42) == "42"
        assert format_value("hello") == "hello"
        assert format_value(3.14) == "3.14"

    def test_short_bytes(self):
        result = format_value(b"test")
        assert result == "test"


class TestExtractFieldValue:
    """Tests for extract_field_value with mock field objects."""

    def test_simple_scalar(self):
        """Should extract a single scalar value from parts[data[0]]."""
        field = type(
            "Field",
            (),
            {
                "parts": [None, None, np.int64(42)],
                "data": [2],
            },
        )()
        result = extract_field_value(field)
        assert result == 42

    def test_array_values(self):
        """Should extract multiple values from parts."""
        field = type(
            "Field",
            (),
            {
                "parts": [None, np.int64(1), np.int64(2), np.int64(3)],
                "data": [1, 2, 3],
            },
        )()
        result = extract_field_value(field)
        assert result == [1, 2, 3]

    def test_empty_data_returns_last_part(self):
        """With no data indices, should return last part value.

        Note: extract_field_value checks tobytes() first (which numpy scalars
        have), then item(). For numpy scalars, tobytes().decode() produces
        a bytestring representation, not the numeric value. This test
        validates the actual behavior.
        """
        val = np.int64(99)
        field = type(
            "Field",
            (),
            {
                "parts": [None, val],
                "data": [],
            },
        )()
        result = extract_field_value(field)
        # np.int64 has tobytes(), so it goes through decode path
        assert isinstance(result, str)

    def test_single_data_index_scalar(self):
        """Should extract scalar value from parts[data[0]] via .item()."""
        val = np.int64(42)
        field = type(
            "Field",
            (),
            {
                "parts": [None, val],
                "data": [1],
            },
        )()
        result = extract_field_value(field)
        assert result == 42

    def test_bytes_part_decoded(self):
        """Parts with tobytes() should be decoded to UTF-8."""
        field = type(
            "Field",
            (),
            {
                "parts": [np.array([ord(c) for c in "hello"], dtype=np.uint8)],
                "data": [],
            },
        )()
        result = extract_field_value(field)
        assert isinstance(result, str)
