"""Convert GGUF model files to HuggingFace safetensors format for Qwen3 architecture.

Standard GQA transformer with SwiGLU MLP. No MoE, no MLA, no Mamba.
1:1 tensor mapping with GGUF shape reversal fix.

Usage:
    python3 gguf_to_safetensors_qwen3.py \\
        --gguf /path/to/model.gguf \\
        --reference-model /path/to/hf_reference/ \\
        --output /path/to/output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from gguf import GGUFReader

from common import (
    copy_reference_files,
    decode_gguf_tensor,
    load_reference_shapes,
    write_shards,
)
from mappings_qwen3 import gguf_name_to_hf

_MAX_DISPLAY = 30


def convert(
    gguf_path: str,
    ref_dir: str,
    output_dir: str,
    shard_size_mb: int = 4500,
    keep_fp16: bool = False,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    ref_shapes = load_reference_shapes(ref_dir)

    reader = GGUFReader(gguf_path)
    gguf_tensors = reader.tensors
    print(f"GGUF: {len(gguf_tensors)} tensors")

    mapping = {}
    unmatched: list[tuple[str, str | None, str]] = []
    for gt in gguf_tensors:
        hf_name = gguf_name_to_hf(gt.name)
        if hf_name and hf_name in ref_shapes:
            mapping[gt.name] = hf_name
        elif hf_name:
            unmatched.append((gt.name, hf_name, "mapped but not in ref"))
        else:
            unmatched.append((gt.name, None, "no mapping rule"))

    if unmatched:
        print(f"\nERROR: {len(unmatched)} unmatched GGUF tensors:", file=sys.stderr)
        for name, hf, reason in unmatched:
            print(f"  {name} -> {hf} ({reason})", file=sys.stderr)
        sys.exit(1)

    mapped_count = len(mapping)
    print(f"Mapped: {mapped_count}/{len(gguf_tensors)}")

    hf_tensors = {}
    decode_errors = []
    for gt in gguf_tensors:
        if gt.name not in mapping:
            continue
        hf_name = mapping[gt.name]
        try:
            # GGUF stores most tensors in Fortran-order (column-major); reverse_shape=True
            # converts to C-order (row-major) which PyTorch/NumPy expect.
            t = decode_gguf_tensor(gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True)
            expected = ref_shapes[hf_name]
            if list(t.shape) != expected:
                decode_errors.append(
                    f"{gt.name} -> {hf_name}: got {list(t.shape)}, expected {expected}"
                )
                continue
            hf_tensors[hf_name] = t
        except Exception as e:
            decode_errors.append(f"{gt.name}: {e}")

    if decode_errors:
        print(f"\nERROR: {len(decode_errors)} tensor decode/shape errors:", file=sys.stderr)
        for msg in decode_errors:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)

    # Verify coverage: all reference tensors must be present
    missing = [k for k in ref_shapes if k not in hf_tensors]
    if missing:
        print(f"\nERROR: {len(missing)} reference tensors not produced:", file=sys.stderr)
        for name in missing[:_MAX_DISPLAY]:
            print(f"  {name}  {ref_shapes[name]}", file=sys.stderr)
        if len(missing) > _MAX_DISPLAY:
            print(f"  ... and {len(missing) - _MAX_DISPLAY} more", file=sys.stderr)
        sys.exit(1)

    print(f"\nConverted: {len(hf_tensors)} tensors")

    copy_reference_files(ref_dir, output)
    write_shards(hf_tensors, output, shard_size_mb=shard_size_mb)


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF to safetensors (Qwen3 architecture)"
    )
    parser.add_argument("--gguf", required=True)
    parser.add_argument(
        "--reference-model",
        required=True,
        help="Path to base model safetensors directory for name/shape mapping",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-size-mb", type=int, default=4500)
    parser.add_argument("--keep-fp16", action="store_true")
    args = parser.parse_args()
    convert(
        args.gguf,
        args.reference_model,
        args.output,
        shard_size_mb=args.shard_size_mb,
        keep_fp16=args.keep_fp16,
    )


if __name__ == "__main__":
    main()
