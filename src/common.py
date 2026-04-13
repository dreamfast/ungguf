"""Shared utilities for GGUF-to-safetensors conversion and verification."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from gguf import dequantize
from gguf.constants import GGMLQuantizationType
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def decode_gguf_tensor(
    tensor,
    keep_f32: bool = True,
    keep_f16: bool = False,
    reverse_shape: bool = True,
) -> torch.Tensor:
    """Convert a GGUF tensor to a PyTorch tensor.

    Args:
        tensor: GGUF tensor object from GGUFReader.
        keep_f32: If True and source is F32, preserve float32 dtype.
                  Otherwise convert to bfloat16.
        keep_f16: If True and source is F16, preserve float16 dtype.
                  Otherwise convert to bfloat16.
        reverse_shape: If True, reverse the GGUF-reported shape to convert
                       from Fortran/column-major (GGML) to C-order (NumPy/PyTorch).
    """
    data = tensor.data
    qtype = tensor.tensor_type
    shape = [int(x) for x in tensor.shape]
    if reverse_shape:
        shape = shape[::-1]

    if qtype == GGMLQuantizationType.BF16:
        arr = np.frombuffer(data, dtype=np.uint16).copy()
        arr32 = np.zeros(len(arr), dtype=np.uint32)
        arr32[:] = arr.astype(np.uint32) << 16
        return torch.from_numpy(arr32.view(np.float32)).to(torch.bfloat16).reshape(shape)
    elif qtype == GGMLQuantizationType.F32:
        t = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy()).reshape(shape)
        if not keep_f32:
            t = t.to(torch.bfloat16)
        return t
    elif qtype == GGMLQuantizationType.F16:
        t = torch.from_numpy(np.frombuffer(data, dtype=np.float16).copy()).reshape(shape)
        if not keep_f16:
            t = t.to(torch.bfloat16)
        return t
    else:
        weights = dequantize(data, qtype).copy()
        return torch.from_numpy(weights).to(torch.bfloat16).reshape(shape)


def load_reference_shapes(ref_dir: str) -> dict[str, list[int]]:
    """Load tensor shapes from a reference safetensors model directory.

    Reads only metadata (not full tensor data) to minimise memory usage.
    """
    shapes = {}
    for p in sorted(Path(ref_dir).glob("*.safetensors")):
        with safe_open(str(p), framework="pt") as sf:
            for k in sf.keys():  # noqa: SIM118
                shapes[k] = list(sf.get_tensor(k).shape)
    return shapes


def load_converted_tensors(conv_dir: str) -> dict[str, torch.Tensor]:
    """Load all tensors from a converted safetensors directory."""
    tensors = {}
    for p in sorted(Path(conv_dir).glob("*.safetensors")):
        with safe_open(str(p), framework="pt") as sf:
            for k in sf.keys():  # noqa: SIM118
                tensors[k] = sf.get_tensor(k)
    return tensors


DEFAULT_CONFIG_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
]


def copy_reference_files(
    ref_dir: str,
    output_dir: str | Path,
    extra_files: list[str] | None = None,
) -> None:
    """Copy config and tokenizer files from reference model to output directory."""
    ref_path = Path(ref_dir)
    out_path = Path(output_dir)

    ref_config = ref_path / "config.json"
    if not ref_config.exists():
        print("WARNING: No config.json in reference model")  # noqa: T201
        return

    shutil.copy2(ref_config, out_path / "config.json")

    files = list(DEFAULT_CONFIG_FILES)
    if extra_files:
        files.extend(extra_files)

    for name in files:
        src = ref_path / name
        if src.exists():
            shutil.copy2(src, out_path / name)

    print(f"Copied config + tokenizer files from {ref_dir}")  # noqa: T201


def write_shards(
    hf_tensors: dict[str, torch.Tensor],
    output_path: Path,
    shard_size_mb: int = 4500,
) -> None:
    """Write tensors to sharded safetensors files.

    Uses HuggingFace naming convention:
        model.safetensors                                (single shard)
        model.safetensors-NNNNN-of-NNNNN.safetensors    (multiple shards)

    Writes model.safetensors.index.json for multi-shard models.
    """
    sorted_names = sorted(hf_tensors.keys())
    shard_size_bytes = shard_size_mb * 1024 * 1024
    shards: list[tuple[int, dict[str, torch.Tensor]]] = []
    current_shard: dict[str, torch.Tensor] = {}
    current_size = 0
    shard_index = 1

    for name in sorted_names:
        tensor = hf_tensors[name]
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current_size + tensor_bytes > shard_size_bytes and current_shard:
            shards.append((shard_index, current_shard))
            shard_index += 1
            current_shard = {}
            current_size = 0
        current_shard[name] = tensor
        current_size += tensor_bytes

    if current_shard:
        shards.append((shard_index, current_shard))

    total_shards = len(shards)
    total_size = sum(t.numel() * t.element_size() for t in hf_tensors.values())
    index_dict: dict = {"metadata": {"total_size": total_size}, "weight_map": {}}

    for shard_num, shard_tensors in tqdm(shards, desc="Writing safetensors"):
        if total_shards == 1:
            shard_name = "model.safetensors"
        else:
            shard_name = f"model.safetensors-{shard_num:05d}-of-{total_shards:05d}.safetensors"

        shard_path = output_path / shard_name
        save_file(shard_tensors, str(shard_path))

        for name in shard_tensors:
            index_dict["weight_map"][name] = shard_name

    if total_shards > 1:
        index_path = output_path / "model.safetensors.index.json"
        with index_path.open("w") as f:
            json.dump(index_dict, f, indent=2)

    print(f"\nDone! {len(hf_tensors)} tensors -> {total_shards} shard(s)")  # noqa: T201
    print(f"Output: {output_path}")  # noqa: T201
