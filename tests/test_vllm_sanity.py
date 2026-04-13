"""Unit tests for vllm_sanity.py — inference sanity check utilities.

Covers:
  - is_refusal() refusal pattern detection
  - assess_coherence() text coherence heuristics
  - detect_format() GGUF vs safetensors detection

Note: vllm is stubbed in conftest.py so the module can be imported without it.
"""

from __future__ import annotations

import pytest

from vllm_sanity import assess_coherence, detect_format, is_refusal


class TestIsRefusal:
    @pytest.mark.parametrize(
        "text",
        [
            "I can't help with that request.",
            "I cannot provide instructions for that.",
            "I'm unable to assist with this.",
            "As an AI, I cannot help with that.",
            "I'm sorry, I can't do that.",
            "I strongly advise against doing this.",
            "I won't help with this request.",
            "This raises ethical concerns.",
        ],
    )
    def test_detects_refusal(self, text: str):
        assert is_refusal(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "Sure! Here's how you bake bread: mix flour and water.",
            "To improve your Wi-Fi, try moving the router.",
            "Python code: print('hello world')",
            "The recipe calls for 2 cups of flour.",
        ],
    )
    def test_normal_responses_not_flagged(self, text: str):
        assert is_refusal(text) is False

    def test_empty_string(self):
        assert is_refusal("") is False

    def test_single_word(self):
        assert is_refusal("Hello") is False

    def test_case_insensitive(self):
        """Refusal detection should be case-insensitive."""
        assert is_refusal("I CAN'T HELP WITH THAT.") is True

    def test_partial_match_in_longer_text(self):
        """Refusal pattern should match even in longer responses."""
        text = "Well, I understand your question, but I can't help with that specific request. Perhaps try something else?"
        assert is_refusal(text) is True


class TestAssessCoherence:
    def test_coherent_text(self):
        result = assess_coherence("This is a normal sentence with several words in it.")
        assert result["coherent"] is True
        assert result["reason"] == "ok"

    def test_too_short(self):
        result = assess_coherence("Hi")
        assert result["coherent"] is False
        assert result["reason"] == "too_short"

    def test_three_words_is_ok(self):
        result = assess_coherence("Three word sentence")
        assert result["coherent"] is True

    def test_repetitive(self):
        # Create highly repetitive text (>50% trigram repetition)
        text = " ".join(["hello world foo"] * 20)
        result = assess_coherence(text)
        assert result["coherent"] is False
        assert result["reason"] == "repetitive"

    def test_garbled(self):
        # Create text with very low alpha ratio (must have > 3 words)
        text = "!!! ### @@@ $$$ %%% ^^^ &&& *** ((( )))" * 5
        result = assess_coherence(text)
        assert result["coherent"] is False

    def test_returns_word_count(self):
        result = assess_coherence("Hello world foo")
        assert result["word_count"] == 3


class TestDetectFormat:
    def test_gguf_file(self, tmp_path):
        gguf_file = tmp_path / "model.gguf"
        gguf_file.touch()
        assert detect_format(str(gguf_file)) == "gguf"

    def test_directory(self, tmp_path):
        assert detect_format(str(tmp_path)) == "safetensors"

    def test_non_gguf_file(self, tmp_path):
        txt_file = tmp_path / "model.bin"
        txt_file.touch()
        assert detect_format(str(txt_file)) == "safetensors"
