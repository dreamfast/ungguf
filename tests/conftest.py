"""Shared fixtures and helpers for the ungguf test suite.

Provides mock GGUF tensor objects, tensor creation helpers, and a sys.modules
stub for the vllm package (so vllm_sanity.py can be imported without vLLM
installed).
"""

from __future__ import annotations

import sys
from collections import namedtuple

import numpy as np
import pytest
import torch
from gguf.constants import GGMLQuantizationType

# ---------------------------------------------------------------------------
# vllm stub — allows importing vllm_sanity.py without vllm installed
# ---------------------------------------------------------------------------
_vllm_stub = type(sys)("vllm")
_vllm_stub.LLM = None  # type: ignore[attr-defined]
_vllm_stub.SamplingParams = None  # type: ignore[attr-defined]
if "vllm" not in sys.modules:
    sys.modules["vllm"] = _vllm_stub  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Mock GGUF tensor type (mirrors gguf.gguf_reader.ReaderTensor)
# ---------------------------------------------------------------------------
MockReaderField = namedtuple("MockReaderField", [])

MockGGUFTensor = namedtuple(
    "MockGGUFTensor",
    ["name", "tensor_type", "shape", "n_elements", "n_bytes", "data_offset", "data", "field"],
)


def make_f32_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    reverse_shape: bool = False,
    fill: float | None = None,
) -> MockGGUFTensor:
    """Create a mock GGUF tensor with F32 data.

    Args:
        name: Tensor name (e.g. ``"blk.0.attn_q.weight"``).
        shape: Logical shape as a tuple. If *reverse_shape* is True the raw
               GGUF shape is stored reversed (column-major convention).
        reverse_shape: Store shape in Fortran order (as GGUF does for most
                       architectures).
        fill: Optional fill value. Defaults to arange-like values.
    """
    gguf_shape = tuple(reversed(shape)) if reverse_shape else shape
    n_elements = int(np.prod(gguf_shape))
    if fill is not None:
        arr = np.full(n_elements, fill, dtype=np.float32)
    else:
        arr = np.arange(n_elements, dtype=np.float32)
    arr = arr.reshape(gguf_shape)
    return MockGGUFTensor(
        name=name,
        tensor_type=GGMLQuantizationType.F32,
        shape=np.array(gguf_shape, dtype=np.uint32),
        n_elements=n_elements,
        n_bytes=n_elements * 4,
        data_offset=0,
        data=arr,
        field=MockReaderField(),
    )


def make_bf16_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    reverse_shape: bool = False,
) -> MockGGUFTensor:
    """Create a mock GGUF tensor with BF16 data (stored as uint16)."""
    gguf_shape = tuple(reversed(shape)) if reverse_shape else shape
    n_elements = int(np.prod(gguf_shape))
    # Generate BF16 values via the encode path: float32 -> bfloat16 -> uint16
    f32_vals = np.arange(n_elements, dtype=np.float32) * 0.01
    bf16_tensor = torch.from_numpy(f32_vals).to(torch.bfloat16)
    uint16_data = bf16_tensor.view(torch.uint16).numpy().copy().reshape(gguf_shape)
    return MockGGUFTensor(
        name=name,
        tensor_type=GGMLQuantizationType.BF16,
        shape=np.array(gguf_shape, dtype=np.uint32),
        n_elements=n_elements,
        n_bytes=n_elements * 2,
        data_offset=0,
        data=uint16_data,
        field=MockReaderField(),
    )


def make_f16_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    reverse_shape: bool = False,
) -> MockGGUFTensor:
    """Create a mock GGUF tensor with F16 data."""
    gguf_shape = tuple(reversed(shape)) if reverse_shape else shape
    n_elements = int(np.prod(gguf_shape))
    f32_vals = np.arange(n_elements, dtype=np.float32) * 0.01
    f16_tensor = torch.from_numpy(f32_vals).to(torch.float16)
    f16_data = f16_tensor.numpy().copy().reshape(gguf_shape)
    return MockGGUFTensor(
        name=name,
        tensor_type=GGMLQuantizationType.F16,
        shape=np.array(gguf_shape, dtype=np.uint32),
        n_elements=n_elements,
        n_bytes=n_elements * 2,
        data_offset=0,
        data=f16_data,
        field=MockReaderField(),
    )


def make_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    fill: float | None = None,
) -> torch.Tensor:
    """Create a small PyTorch tensor with predictable values for assertions."""
    n = int(np.prod(shape))
    if fill is not None:
        t = torch.full((n,), fill, dtype=dtype)
    else:
        t = torch.arange(n, dtype=dtype) * 0.01
    return t.reshape(shape)


@pytest.fixture
def sample_tensors() -> dict[str, torch.Tensor]:
    """Return a small dict of named tensors for shard-writing tests."""
    return {
        "model.embed_tokens.weight": make_tensor((100, 64)),
        "model.norm.weight": make_tensor((100,)),
        "model.layers.0.self_attn.q_proj.weight": make_tensor((128, 100)),
        "model.layers.0.self_attn.k_proj.weight": make_tensor((32, 100)),
    }
