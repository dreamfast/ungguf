"""Convert Qwen3.6-27B GGUF model files to HuggingFace safetensors format.

Qwen3.6-27B uses the same Qwen3_5ForConditionalGeneration architecture as Qwen3.5,
sharing identical GGUF tensor names, conventions (norm-1.0, A_log), and V-head reorder
logic (v_per_k=3 for Qwen3.6-27B: 16 K-heads / 48 V-heads).

Key difference from the Qwen3.5 converter: the GGUF file does NOT contain MTP
(Multi-Token Prediction) or vision encoder tensors. These are preserved by copying
them from the reference model's safetensors, ensuring the converted output is a
complete, loadable model.

Architecture: Hybrid Gated DeltaNet + Gated Attention
  - 64 layers, hidden_size=5120, intermediate_size=17408
  - 3 linear attention + 1 full attention per block of 4
  - reverse_shape=False (C-order, same as Qwen3.5)

Usage (via ungguf.sh):
    ungguf convert-qwen36 <gguf_file> <output_dir> <ref_model_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

from common import (
    copy_reference_files,
    decode_gguf_tensor,
    load_converted_tensors,
    load_reference_shapes,
    write_shards,
)
from mappings_qwen35 import (
    apply_conventions,
    apply_inverse_v_reorder,
    get_linear_attn_config,
    gguf_name_to_hf,
)

# Tensor name prefixes that are NOT present in the GGUF but exist in the
# reference model. These tensors are copied verbatim from the reference.
_MISSING_PREFIXES = (
    "mtp.",
    "model.visual.",
)

_MAX_DISPLAY = 30


def _is_missing_tensor(hf_name: str) -> bool:
    """Check if a tensor name belongs to MTP or vision modules (absent from GGUF)."""
    return any(hf_name.startswith(p) for p in _MISSING_PREFIXES)


def convert_gguf_to_safetensors(
    gguf_path: str,
    output_dir: str,
    reference_model: str,
    shard_size_mb: int = 4500,
    keep_fp16: bool = False,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)

    gguf_tensors = reader.tensors
    print(f"GGUF tensors: {len(gguf_tensors)}")

    qtypes: dict[str, int] = {}
    for t in gguf_tensors:
        qt = str(t.tensor_type)
        qtypes[qt] = qtypes.get(qt, 0) + 1
    print(f"Quant types: {qtypes}")

    print(f"Loading reference model shapes from: {reference_model}")
    ref_shapes = load_reference_shapes(reference_model)
    print(f"Reference tensors: {len(ref_shapes)}")

    # Separate reference tensors into GGUF-expected vs. missing (MTP/vision)
    gguf_expected = {k: v for k, v in ref_shapes.items() if not _is_missing_tensor(k)}
    missing_in_gguf = {k: v for k, v in ref_shapes.items() if _is_missing_tensor(k)}
    print(f"GGUF-expected reference tensors: {len(gguf_expected)}")
    print(f"Missing (MTP/vision) tensors to copy from reference: {len(missing_in_gguf)}")

    # Build GGUF → HF name mapping
    mapping: dict[str, str] = {}
    unmapped: list[tuple[str, list[int], str]] = []

    for gt in gguf_tensors:
        hf_name = gguf_name_to_hf(gt.name)
        if hf_name is None:
            unmapped.append((gt.name, [int(x) for x in gt.shape], "no name mapping rule"))
            continue

        # Try direct match, then with prefix variations
        resolved = None
        for candidate in (
            hf_name,
            hf_name.replace("model.language_model.", "model."),
            hf_name.replace("model.language_model.", ""),
        ):
            if candidate in gguf_expected:
                resolved = candidate
                break

        if resolved:
            mapping[gt.name] = resolved
        else:
            unmapped.append(
                (gt.name, [int(x) for x in gt.shape], f"mapped to {hf_name} but not in ref")
            )

    print(f"Mapped {len(mapping)} / {len(gguf_tensors)} GGUF tensors to HF names")

    if unmapped:
        print(f"\nERROR: {len(unmapped)} unmatched GGUF tensors:", file=sys.stderr)
        for name, shape, reason in unmapped[:_MAX_DISPLAY]:
            print(f"  {name}  {shape}  {reason}", file=sys.stderr)
        if len(unmapped) > _MAX_DISPLAY:
            print(f"  ... and {len(unmapped) - _MAX_DISPLAY} more", file=sys.stderr)
        sys.exit(1)

    # Check for duplicate HF targets
    hf_targets = list(mapping.values())
    dupes = [k for k in set(hf_targets) if hf_targets.count(k) > 1]
    if dupes:
        print(
            f"\nERROR: {len(dupes)} duplicate HF target(s) — ambiguous mapping:",
            file=sys.stderr,
        )
        for d in dupes[:10]:
            sources = [g for g, h in mapping.items() if h == d]
            print(f"  {d} <- {sources}", file=sys.stderr)
        sys.exit(1)

    # Load V-head config from reference config.json
    ref_config = Path(reference_model) / "config.json"
    if not ref_config.exists():
        print(
            "ERROR: No config.json in reference model — required for architecture detection",
            file=sys.stderr,
        )
        sys.exit(1)

    copy_reference_files(
        reference_model,
        output_path,
        extra_files=[
            "tokenization_qwen.py",
            "video_preprocessor_config.json",
            "chat_template.jinja",
        ],
    )

    with ref_config.open() as f:
        model_config = json.load(f)

    la_cfg = get_linear_attn_config(model_config)
    if la_cfg:
        print(
            f"V-head reorder enabled: k_heads={la_cfg['num_k_heads']}, "
            f"v_heads={la_cfg['num_v_heads']}, v_per_k={la_cfg['v_per_k']}, "
            f"key_dim={la_cfg['key_dim']}, value_dim={la_cfg['value_dim']}"
        )
    else:
        print("V-head reorder: not needed (num_k_heads == num_v_heads or config N/A)")

    # Decode and transform all GGUF tensors
    hf_tensors: dict[str, torch.Tensor] = {}
    decode_errors: list[str] = []

    for gt in gguf_tensors:
        if gt.name not in mapping:
            continue
        try:
            # Qwen3.6 GGUF tensors are stored in C-order (row-major); do NOT reverse.
            tensor = decode_gguf_tensor(gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=False)
            hf_name = mapping[gt.name]
            target_shape = gguf_expected.get(hf_name)
            if target_shape and list(tensor.shape) != target_shape:
                if tensor.numel() == int(np.prod(target_shape)):
                    tensor = tensor.reshape(target_shape).contiguous()
                else:
                    decode_errors.append(
                        f"{hf_name}: GGUF {list(tensor.shape)} vs ref {target_shape}"
                        " (element count mismatch)"
                    )
                    continue

            tensor = apply_conventions(hf_name, tensor)

            if la_cfg and "linear_attn." in hf_name:
                tensor = apply_inverse_v_reorder(tensor, hf_name, la_cfg)

            hf_tensors[hf_name] = tensor
        except Exception as e:
            decode_errors.append(f"{gt.name}: {e}")

    if decode_errors:
        print(f"\nERROR: {len(decode_errors)} tensor decode/transform errors:", file=sys.stderr)
        for msg in decode_errors:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)

    # Verify GGUF coverage: all GGUF-expected reference tensors must be present
    missing_from_conversion = [k for k in gguf_expected if k not in hf_tensors]
    if missing_from_conversion:
        print(
            f"\nERROR: {len(missing_from_conversion)} GGUF-expected tensors not produced:",
            file=sys.stderr,
        )
        for name in missing_from_conversion[:_MAX_DISPLAY]:
            print(f"  {name}  {gguf_expected[name]}", file=sys.stderr)
        if len(missing_from_conversion) > _MAX_DISPLAY:
            print(f"  ... and {len(missing_from_conversion) - _MAX_DISPLAY} more", file=sys.stderr)
        sys.exit(1)

    # Copy MTP and vision tensors from reference model
    print(f"\nCopying {len(missing_in_gguf)} MTP/vision tensors from reference model...")
    ref_tensors = load_converted_tensors(reference_model)
    copied = 0
    for hf_name in sorted(missing_in_gguf):
        if hf_name in ref_tensors:
            hf_tensors[hf_name] = ref_tensors[hf_name]
            copied += 1
        else:
            print(
                f"  WARNING: reference tensor {hf_name} not found in safetensors", file=sys.stderr
            )
    print(f"Copied {copied} / {len(missing_in_gguf)} missing tensors from reference")

    # Final coverage check
    total_missing = [k for k in ref_shapes if k not in hf_tensors]
    if total_missing:
        print(
            f"\nERROR: {len(total_missing)} reference tensors still missing after copy:",
            file=sys.stderr,
        )
        for name in total_missing[:_MAX_DISPLAY]:
            print(f"  {name}  {ref_shapes[name]}", file=sys.stderr)
        sys.exit(1)

    print(
        f"\nTotal tensors: {len(hf_tensors)} ({len(gguf_tensors)} from GGUF + {copied} from reference)"
    )
    write_shards(hf_tensors, output_path, shard_size_mb=shard_size_mb)

    del hf_tensors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.6 GGUF to safetensors using reference model"
    )
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference-model",
        required=True,
        help="Path to base model safetensors directory for name mapping and MTP/vision tensors",
    )
    parser.add_argument("--shard-size-mb", type=int, default=4500)
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Preserve float16 tensors as-is instead of converting to bfloat16",
    )
    args = parser.parse_args()

    convert_gguf_to_safetensors(
        gguf_path=args.gguf,
        output_dir=args.output,
        reference_model=args.reference_model,
        shard_size_mb=args.shard_size_mb,
        keep_fp16=args.keep_fp16,
    )


if __name__ == "__main__":
    main()
