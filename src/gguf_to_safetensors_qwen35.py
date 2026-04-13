"""Convert GGUF model files to HuggingFace safetensors format.

Uses deterministic name-based mapping for Qwen3.5 hybrid (Mamba2+Transformer)
architecture. Falls back to shape-based matching for unknown architectures.
Handles BF16 GGUF natively and dequantizes Q8_0/etc to bfloat16.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from gguf import GGUFReader

from common import (
    copy_reference_files,
    decode_gguf_tensor,
    load_reference_shapes,
    write_shards,
)
from mappings_qwen35 import (
    apply_conventions,
    apply_inverse_v_reorder,
    get_linear_attn_config,
    gguf_name_to_hf,
)

_MAX_DISPLAY = 30


def build_mapping(
    gguf_tensors: list, ref_shapes: dict[str, list[int]]
) -> tuple[dict[str, str], list]:
    mapping = {}
    unmatched = []

    for gt in gguf_tensors:
        hf_name = gguf_name_to_hf(gt.name)
        if hf_name and hf_name in ref_shapes:
            mapping[gt.name] = hf_name
        elif hf_name:
            alt = hf_name.replace("model.language_model.", "model.")
            alt2 = hf_name.replace("model.language_model.", "")
            if alt in ref_shapes:
                mapping[gt.name] = alt
            elif alt2 in ref_shapes:
                mapping[gt.name] = alt2
            else:
                unmatched.append(
                    (gt.name, [int(x) for x in gt.shape], f"mapped to {hf_name} but not in ref")
                )

    mapped_hf = set(mapping.values())
    remaining = [gt for gt in gguf_tensors if gt.name not in mapping]

    if remaining:
        ref_by_shape = defaultdict(list)
        for name, shape in ref_shapes.items():
            if name not in mapped_hf and "language_model" in name:
                ref_by_shape[tuple(shape)].append(name)
                if len(shape) == 2:  # noqa: PLR2004
                    ref_by_shape[tuple(reversed(shape))].append(name)

        for gt in remaining:
            gshape = tuple(int(x) for x in gt.shape)
            candidates = ref_by_shape.get(gshape, [])
            if candidates:
                best = candidates[0]
                for c in candidates:
                    if _name_similarity(gt.name, c) > _name_similarity(gt.name, best):
                        best = c
                mapping[gt.name] = best
                mapped_hf.add(best)
            else:
                unmatched.append((gt.name, list(gshape), "no shape match"))

    return mapping, unmatched


def _name_similarity(gguf_name: str, hf_name: str) -> int:
    score = 0
    if "norm" in gguf_name and "norm" in hf_name:
        score += 10
    if "embed" in gguf_name and "embed" in hf_name:
        score += 10
    if gguf_name == "output.weight" and "lm_head" in hf_name:
        score += 10
    if "output_norm" in gguf_name and hf_name.endswith("norm.weight"):
        score += 10
    if "ffn_down" in gguf_name and "down_proj" in hf_name:
        score += 10
    if "ffn_gate" in gguf_name and "gate_proj" in hf_name:
        score += 10
    if "ffn_up" in gguf_name and "up_proj" in hf_name:
        score += 10
    if "ssm_out" in gguf_name and "out_proj" in hf_name:
        score += 10
    if "ssm_conv" in gguf_name and "conv1d" in hf_name:
        score += 10
    if "ssm_norm" in gguf_name and "linear_attn.norm" in hf_name:
        score += 10
    if gguf_name.rsplit(".", maxsplit=1)[-1] == "ssm_a" and "A_log" in hf_name:
        score += 10
    if "ssm_dt" in gguf_name and "dt_bias" in hf_name:
        score += 10
    if "attn_q." in gguf_name and "q_proj" in hf_name:
        score += 10
    if "attn_k." in gguf_name and "k_proj" in hf_name:
        score += 10
    if "attn_v." in gguf_name and "v_proj" in hf_name:
        score += 10
    if "attn_output" in gguf_name and "o_proj" in hf_name:
        score += 10
    if "attn_qkv" in gguf_name and "in_proj_qkv" in hf_name:
        score += 10
    if "attn_gate" in gguf_name and "in_proj_z" in hf_name:
        score += 10
    if "ssm_alpha" in gguf_name and "in_proj_a" in hf_name:
        score += 10
    if "ssm_beta" in gguf_name and "in_proj_b" in hf_name:
        score += 10

    g_layer = re.search(r"blk\.(\d+)\.", gguf_name)
    h_layer = re.search(r"layers\.(\d+)\.", hf_name)
    if g_layer and h_layer and g_layer.group(1) == h_layer.group(1):
        score += 50

    return score


def convert_gguf_to_safetensors(
    gguf_path: str,
    output_dir: str,
    reference_model: str,
    shard_size_mb: int = 4500,
    keep_fp16: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Reading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)

    gguf_tensors = reader.tensors
    print(f"GGUF tensors: {len(gguf_tensors)}")

    qtypes = {str(t.tensor_type) for t in gguf_tensors}
    print(f"Quant types: {qtypes}")

    print(f"Loading reference model shapes from: {reference_model}")
    ref_shapes = load_reference_shapes(reference_model)
    print(f"Reference tensors: {len(ref_shapes)}")

    lm_ref = sum(1 for k in ref_shapes if "language_model" in k)
    print(f"Reference language_model tensors: {lm_ref}")

    mapping, unmatched_gguf = build_mapping(gguf_tensors, ref_shapes)
    print(f"Mapped {len(mapping)} / {len(gguf_tensors)} GGUF tensors to HF names")

    if unmatched_gguf:
        print(f"\nERROR: {len(unmatched_gguf)} unmatched GGUF tensors:", file=sys.stderr)
        for item in unmatched_gguf[:_MAX_DISPLAY]:
            name, shape = item[0], item[1]
            reason = item[2] if len(item) > 2 else ""  # noqa: PLR2004
            print(f"  {name}  {shape}  {reason}", file=sys.stderr)
        if len(unmatched_gguf) > _MAX_DISPLAY:
            print(f"  ... and {len(unmatched_gguf) - _MAX_DISPLAY} more", file=sys.stderr)
        sys.exit(1)

    hf_targets = list(mapping.values())
    dupes = [k for k in set(hf_targets) if hf_targets.count(k) > 1]
    if dupes:
        print(
            f"\nERROR: {len(dupes)} duplicate HF target(s) — ambiguous mapping:", file=sys.stderr
        )
        for d in dupes[:10]:
            sources = [g for g, h in mapping.items() if h == d]
            print(f"  {d} <- {sources}", file=sys.stderr)
        sys.exit(1)

    la_cfg = None
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

    hf_tensors = {}
    decode_errors = []
    for gt in gguf_tensors:
        if gt.name not in mapping:
            continue
        try:
            # Qwen3.5 GGUF tensors are stored in C-order (row-major), unlike most
            # GGUF files which use Fortran-order (column-major). Do NOT reverse.
            tensor = decode_gguf_tensor(gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=False)
            hf_name = mapping[gt.name]
            target_shape = ref_shapes.get(hf_name)
            if target_shape and list(tensor.shape) != target_shape:
                ts = list(tensor.shape)
                if tensor.numel() == int(np.prod(target_shape)):
                    tensor = tensor.reshape(target_shape).contiguous()
                else:
                    decode_errors.append(
                        f"{hf_name}: GGUF {ts} vs ref {target_shape} (element count mismatch)"
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

    # Verify coverage: all reference tensors must be present
    missing = [k for k in ref_shapes if k not in hf_tensors]
    if missing:
        print(f"\nERROR: {len(missing)} reference tensors not produced:", file=sys.stderr)
        for name in missing[:_MAX_DISPLAY]:
            print(f"  {name}  {ref_shapes[name]}", file=sys.stderr)
        if len(missing) > _MAX_DISPLAY:
            print(f"  ... and {len(missing) - _MAX_DISPLAY} more", file=sys.stderr)
        sys.exit(1)

    print(f"\nConverted {len(hf_tensors)} tensors")
    write_shards(hf_tensors, output_path, shard_size_mb=shard_size_mb)

    del hf_tensors


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF to safetensors using reference model"
    )
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference-model",
        required=True,
        help="Path to base model safetensors directory for name mapping",
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
