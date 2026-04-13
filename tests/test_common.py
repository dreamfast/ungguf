"""Unit tests for common.py — shared conversion utilities.

Covers:
  - decode_gguf_tensor() for F32, F16, BF16 quantization types
  - write_shards() single/multi shard, naming, index JSON
  - copy_reference_files() file copying
  - load_converted_tensors() round-trip
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from conftest import MockGGUFTensor, MockReaderField, make_f16_tensor, make_f32_tensor, make_tensor
from gguf.constants import GGMLQuantizationType
from safetensors.torch import save_file

from common import (
    copy_reference_files,
    decode_gguf_tensor,
    load_converted_tensors,
    write_shards,
)


# ---------------------------------------------------------------------------
# decode_gguf_tensor tests
# ---------------------------------------------------------------------------
class TestDecodeGgufTensor:
    def test_f32_preserved(self):
        """F32 tensor should be kept as float32 when keep_f32=True."""
        gt = make_f32_tensor("test.weight", (4, 8), reverse_shape=True)
        result = decode_gguf_tensor(gt, keep_f32=True, reverse_shape=True)
        assert result.dtype == torch.float32
        assert list(result.shape) == [4, 8]

    def test_f32_to_bf16(self):
        """F32 tensor should convert to bfloat16 when keep_f32=False."""
        gt = make_f32_tensor("test.weight", (4, 8), reverse_shape=True)
        result = decode_gguf_tensor(gt, keep_f32=False, reverse_shape=True)
        assert result.dtype == torch.bfloat16

    def test_f16_to_bf16(self):
        """F16 tensor should convert to bfloat16 when keep_f16=False."""
        gt = make_f16_tensor("test.weight", (4, 8), reverse_shape=True)
        result = decode_gguf_tensor(gt, keep_f16=False, reverse_shape=True)
        assert result.dtype == torch.bfloat16

    def test_f16_preserved(self):
        """F16 tensor should stay float16 when keep_f16=True."""
        gt = make_f16_tensor("test.weight", (4, 8), reverse_shape=True)
        result = decode_gguf_tensor(gt, keep_f16=True, reverse_shape=True)
        assert result.dtype == torch.float16

    def test_bf16_decode(self):
        """BF16 tensor should decode correctly via uint16 shift trick."""
        # Create known BF16 values
        f32_vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        bf16_t = torch.from_numpy(f32_vals).to(torch.bfloat16)
        uint16_data = bf16_t.view(torch.uint16).numpy().copy()

        gt = MockGGUFTensor(
            name="test.weight",
            tensor_type=GGMLQuantizationType.BF16,
            shape=np.array([4], dtype=np.uint32),
            n_elements=4,
            n_bytes=8,
            data_offset=0,
            data=uint16_data,
            field=MockReaderField(),
        )
        result = decode_gguf_tensor(gt, reverse_shape=False)
        assert result.dtype == torch.bfloat16
        assert torch.allclose(result.float(), torch.tensor([1.0, 2.0, 3.0, 4.0]), atol=1e-2)

    def test_reverse_shape_true(self):
        """Shape should be reversed when reverse_shape=True."""
        gt = make_f32_tensor("test.weight", (4, 8), reverse_shape=True)
        result = decode_gguf_tensor(gt, keep_f32=True, reverse_shape=True)
        assert list(result.shape) == [4, 8]

    def test_reverse_shape_false(self):
        """Shape should NOT be reversed when reverse_shape=False."""
        gt = make_f32_tensor("test.weight", (4, 8), reverse_shape=False)
        result = decode_gguf_tensor(gt, keep_f32=True, reverse_shape=False)
        assert list(result.shape) == [4, 8]


# ---------------------------------------------------------------------------
# write_shards tests
# ---------------------------------------------------------------------------
class TestWriteShards:
    def test_single_shard(self, tmp_path: Path, sample_tensors: dict[str, torch.Tensor]):
        """Small tensors should produce a single model.safetensors file."""
        write_shards(sample_tensors, tmp_path, shard_size_mb=100)
        assert (tmp_path / "model.safetensors").exists()
        assert not (tmp_path / "model.safetensors.index.json").exists()

    def test_multi_shard(self, tmp_path: Path):
        """Large shard_size_mb=0 forces each tensor into its own shard."""
        tensors = {
            "model.a.weight": make_tensor((100, 100)),
            "model.b.weight": make_tensor((100, 100)),
        }
        write_shards(tensors, tmp_path, shard_size_mb=0)
        # Should produce multiple shard files
        shard_files = list(tmp_path.glob("model.safetensors-*.safetensors"))
        assert len(shard_files) == 2
        # Should produce index JSON
        assert (tmp_path / "model.safetensors.index.json").exists()

    def test_index_json_format(self, tmp_path: Path):
        """Multi-shard index JSON should have weight_map and total_size."""
        tensors = {
            "model.a.weight": torch.ones(10, 10, dtype=torch.float32),
            "model.b.weight": torch.ones(5, 5, dtype=torch.float32),
        }
        write_shards(tensors, tmp_path, shard_size_mb=0)

        index_path = tmp_path / "model.safetensors.index.json"
        with index_path.open() as f:
            index = json.load(f)

        assert "metadata" in index
        assert "total_size" in index["metadata"]
        assert "weight_map" in index
        assert set(index["weight_map"].keys()) == {"model.a.weight", "model.b.weight"}

    def test_shard_naming_convention(self, tmp_path: Path):
        """Shard files should follow model.safetensors-NNNNN-of-NNNNN.safetensors."""
        tensors = {f"model.t{i}.weight": make_tensor((50, 50)) for i in range(3)}
        write_shards(tensors, tmp_path, shard_size_mb=0)

        shard_files = sorted(tmp_path.glob("model.safetensors-*.safetensors"))
        for f in shard_files:
            # Pattern: model.safetensors-00001-of-00003.safetensors
            name = f.name
            assert name.startswith("model.safetensors-")
            assert name.endswith(".safetensors")

    def test_sorted_keys(self, tmp_path: Path):
        """Tensors should be written in sorted name order."""
        tensors = {
            "model.z.weight": make_tensor((2, 2)),
            "model.a.weight": make_tensor((2, 2)),
        }
        write_shards(tensors, tmp_path, shard_size_mb=100)
        assert (tmp_path / "model.safetensors").exists()
        # Verify keys in saved file are sorted
        loaded = load_converted_tensors(str(tmp_path))
        assert list(loaded.keys()) == sorted(loaded.keys())


# ---------------------------------------------------------------------------
# copy_reference_files tests
# ---------------------------------------------------------------------------
class TestCopyReferenceFiles:
    def test_copies_config_and_existing_files(self, tmp_path: Path):
        """Should copy config.json and any existing default files."""
        ref = tmp_path / "ref"
        ref.mkdir()
        out = tmp_path / "out"
        out.mkdir()

        (ref / "config.json").write_text('{"model_type": "qwen3"}')
        (ref / "tokenizer.json").write_text('{"version": 1}')
        # Non-existent files should be silently skipped
        copy_reference_files(str(ref), str(out))

        assert (out / "config.json").read_text() == '{"model_type": "qwen3"}'
        assert (out / "tokenizer.json").exists()
        assert not (out / "vocab.json").exists()  # didn't exist in ref

    def test_handles_missing_config(self, tmp_path: Path, capsys):
        """Should print warning and return when config.json is missing."""
        ref = tmp_path / "ref"
        ref.mkdir()
        out = tmp_path / "out"
        out.mkdir()

        copy_reference_files(str(ref), str(out))
        assert not (out / "config.json").exists()
        captured = capsys.readouterr()
        assert "WARNING" in captured.out

    def test_extra_files(self, tmp_path: Path):
        """Extra files list should be copied when present."""
        ref = tmp_path / "ref"
        ref.mkdir()
        out = tmp_path / "out"
        out.mkdir()

        (ref / "config.json").write_text("{}")
        (ref / "custom_file.txt").write_text("custom content")

        copy_reference_files(str(ref), str(out), extra_files=["custom_file.txt"])
        assert (out / "custom_file.txt").read_text() == "custom content"


# ---------------------------------------------------------------------------
# load_converted_tensors tests
# ---------------------------------------------------------------------------
class TestLoadConvertedTensors:
    def test_round_trip(self, tmp_path: Path):
        """Write tensors via save_file, load them back, verify shapes match."""
        tensors = {
            "model.a.weight": torch.randn(10, 5),
            "model.b.bias": torch.randn(5),
        }
        save_file(tensors, str(tmp_path / "model.safetensors"))

        loaded = load_converted_tensors(str(tmp_path))
        assert set(loaded.keys()) == set(tensors.keys())
        for name, value in tensors.items():
            assert list(loaded[name].shape) == list(value.shape)
            assert torch.equal(loaded[name], value)

    def test_multi_file(self, tmp_path: Path):
        """Should load tensors from multiple safetensors files."""
        t1 = {"model.a.weight": torch.randn(3, 3)}
        t2 = {"model.b.weight": torch.randn(4, 4)}
        save_file(t1, str(tmp_path / "model-00001.safetensors"))
        save_file(t2, str(tmp_path / "model-00002.safetensors"))

        loaded = load_converted_tensors(str(tmp_path))
        assert "model.a.weight" in loaded
        assert "model.b.weight" in loaded
