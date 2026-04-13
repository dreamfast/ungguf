"""Convert GGUF model files to HuggingFace safetensors format.

Specific converter for GLM-4.7-Flash (deepseek2 architecture with MLA attention + MoE).
Handles kv_b_proj reconstruction, expert tensor splitting, and MTP layer 47 tensors.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from gguf import GGUFReader

from common import (
    copy_reference_files,
    decode_gguf_tensor,
    load_reference_shapes,
    write_shards,
)
from mappings_glm47 import (
    DENSE_MLP_MAP,
    EXPERT_HF_MAP,
    EXPERT_SUFFIXES,
    GLOBAL_MAP,
    KV_B_SUFFIXES,
    LAYER_SUFFIX_MAP,
    MOE_GATE_MAP,
    MOE_SHARED_MAP,
    MTP_TENSOR_NAMES,
    NUM_LAYERS,
    reconstruct_kv_b,
)

_MAX_DISPLAY = 30


def build_mapping(
    gguf_tensors: list,
    ref_shapes: dict[str, list[int]],
) -> tuple[dict[str, str], dict[int, list[str | None]], list[tuple[str, int, str]], list]:
    mapping: dict[str, str] = {}
    kv_b_pairs: dict[int, list[str | None]] = {}
    expert_tensors: list[tuple[str, int, str]] = []
    unmatched: list = []

    for gt in gguf_tensors:
        name = gt.name
        shape = [int(x) for x in gt.shape]

        if name in GLOBAL_MAP:
            hf_name = GLOBAL_MAP[name]
            if hf_name in ref_shapes:
                mapping[name] = hf_name
            else:
                unmatched.append((name, shape, f"mapped to {hf_name} but not in ref"))
            continue

        m = re.match(r"blk\.(\d+)\.(.*)", name)
        if not m:
            unmatched.append((name, shape, "not a layer tensor"))
            continue

        layer_num = int(m.group(1))
        suffix = m.group(2)

        if suffix in KV_B_SUFFIXES:
            if layer_num not in kv_b_pairs:
                kv_b_pairs[layer_num] = [None, None]
            if suffix == "attn_k_b.weight":
                kv_b_pairs[layer_num][0] = name
            else:
                kv_b_pairs[layer_num][1] = name
            continue

        if suffix in EXPERT_SUFFIXES:
            expert_tensors.append((name, layer_num, suffix))
            continue

        if suffix in LAYER_SUFFIX_MAP:
            hf_name = f"model.layers.{layer_num}.{LAYER_SUFFIX_MAP[suffix]}"
            if hf_name in ref_shapes:
                mapping[name] = hf_name
            else:
                unmatched.append((name, shape, f"mapped to {hf_name} but not in ref"))
            continue

        if layer_num == 0 and suffix in DENSE_MLP_MAP:
            hf_name = f"model.layers.0.{DENSE_MLP_MAP[suffix]}"
            if hf_name in ref_shapes:
                mapping[name] = hf_name
            else:
                unmatched.append((name, shape, f"mapped to {hf_name} but not in ref"))
            continue

        if layer_num >= 1 and suffix in MOE_SHARED_MAP:
            hf_name = f"model.layers.{layer_num}.{MOE_SHARED_MAP[suffix]}"
            if hf_name in ref_shapes:
                mapping[name] = hf_name
            else:
                unmatched.append((name, shape, f"mapped to {hf_name} but not in ref"))
            continue

        if layer_num >= 1 and suffix in MOE_GATE_MAP:
            hf_name = f"model.layers.{layer_num}.{MOE_GATE_MAP[suffix]}"
            if hf_name in ref_shapes:
                mapping[name] = hf_name
            else:
                unmatched.append((name, shape, f"mapped to {hf_name} but not in ref"))
            continue

        unmatched.append((name, shape, "unknown suffix"))

    return mapping, kv_b_pairs, expert_tensors, unmatched


def split_expert_tensor(
    expert_3d: torch.Tensor,
    layer_num: int,
    gguf_suffix: str,
    ref_shapes: dict[str, list[int]],
) -> list[tuple[str, torch.Tensor]]:
    hf_proj_name = EXPERT_HF_MAP[gguf_suffix]
    results = []
    num_experts = expert_3d.shape[0]

    for e in range(num_experts):
        hf_name = f"model.layers.{layer_num}.mlp.experts.{e}.{hf_proj_name}"
        expert_slice = expert_3d[e]

        target_shape = ref_shapes.get(hf_name)
        if target_shape and list(expert_slice.shape) != target_shape:
            if expert_slice.numel() == int(np.prod(target_shape)):
                expert_slice = expert_slice.reshape(target_shape).contiguous()
            else:
                raise ValueError(
                    f"Expert {hf_name}: element count mismatch "
                    f"(got {list(expert_slice.shape)}, expected {target_shape})"
                )
        results.append((hf_name, expert_slice.contiguous()))

    return results


def convert_gguf_to_safetensors_glm47(
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

    gguf_by_name = {t.name: t for t in gguf_tensors}

    mapping, kv_b_pairs, expert_tensors, unmatched = build_mapping(gguf_tensors, ref_shapes)
    print(f"Mapped {len(mapping)} / {len(gguf_tensors)} GGUF tensors to HF names (1:1)")
    print(f"KV-B pairs: {len(kv_b_pairs)} layers")
    print(f"Expert tensors: {len(expert_tensors)} (expect {3 * (NUM_LAYERS - 1)} = 3 * 47 layers)")

    if unmatched:
        print(f"\nERROR: {len(unmatched)} unmatched GGUF tensors:", file=sys.stderr)
        for item in unmatched[:_MAX_DISPLAY]:
            name, shape = item[0], item[1]
            reason = item[2] if len(item) > 2 else ""  # noqa: PLR2004
            print(f"  {name}  {shape}  {reason}", file=sys.stderr)
        if len(unmatched) > _MAX_DISPLAY:
            print(f"  ... and {len(unmatched) - _MAX_DISPLAY} more", file=sys.stderr)
        sys.exit(1)

    copy_reference_files(reference_model, output_path)

    hf_tensors = {}

    # Phase 1: 1:1 mapped tensors
    decode_errors = []
    for gt in gguf_tensors:
        if gt.name not in mapping:
            continue
        try:
            # GGUF stores most tensors in Fortran-order (column-major); reverse_shape=True
            # converts to C-order (row-major) which PyTorch/NumPy expect.
            tensor = decode_gguf_tensor(gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True)
            hf_name = mapping[gt.name]
            target_shape = ref_shapes.get(hf_name)
            if target_shape and list(tensor.shape) != target_shape:
                if tensor.numel() == int(np.prod(target_shape)):
                    tensor = tensor.reshape(target_shape).contiguous()
                else:
                    decode_errors.append(
                        f"shape mismatch {hf_name}: GGUF {list(tensor.shape)} "
                        f"vs ref {target_shape} (element count mismatch)"
                    )
                    continue
            hf_tensors[hf_name] = tensor
        except Exception as e:
            decode_errors.append(f"{gt.name}: {e}")

    # Phase 2: KV-B reconstruction
    print(f"\nReconstructing kv_b_proj for {len(kv_b_pairs)} layers...")
    for layer_num, pair in kv_b_pairs.items():
        k_b_name, v_b_name = pair[0], pair[1]
        if k_b_name is None or v_b_name is None:
            decode_errors.append(
                f"Layer {layer_num}: missing k_b or v_b tensor (k={k_b_name}, v={v_b_name})"
            )
            continue

        hf_name = f"model.layers.{layer_num}.self_attn.kv_b_proj.weight"
        if hf_name not in ref_shapes:
            decode_errors.append(f"Layer {layer_num}: {hf_name} not in reference shapes")
            continue

        try:
            k_b_gt = gguf_by_name[k_b_name]
            v_b_gt = gguf_by_name[v_b_name]
            k_b_t = decode_gguf_tensor(
                k_b_gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True
            )
            v_b_t = decode_gguf_tensor(
                v_b_gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True
            )

            kv_b_proj = reconstruct_kv_b(k_b_t, v_b_t)
            target_shape = ref_shapes[hf_name]
            if list(kv_b_proj.shape) != target_shape:
                if kv_b_proj.numel() == int(np.prod(target_shape)):
                    kv_b_proj = kv_b_proj.reshape(target_shape).contiguous()
                else:
                    decode_errors.append(
                        f"kv_b shape mismatch {hf_name}: got {list(kv_b_proj.shape)} vs ref {target_shape}"
                    )
                    continue
            hf_tensors[hf_name] = kv_b_proj
        except Exception as e:
            decode_errors.append(f"kv_b layer {layer_num}: {e}")

    # Phase 3: Expert tensor splitting
    print(f"\nSplitting {len(expert_tensors)} expert tensors...")
    for gguf_name, layer_num, suffix in expert_tensors:
        try:
            gt = gguf_by_name[gguf_name]
            expert_3d = decode_gguf_tensor(
                gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True
            )

            expert_pairs = split_expert_tensor(expert_3d, layer_num, suffix, ref_shapes)
            hf_tensors.update(dict(expert_pairs))
        except Exception as e:  # noqa: PERF203
            decode_errors.append(f"expert {gguf_name}: {e}")

    # Phase 4: MTP (layer 47) tensors are expected to be missing from GGUF
    missing_mtp = [
        mtp_name
        for mtp_name in MTP_TENSOR_NAMES
        if mtp_name in ref_shapes and mtp_name not in hf_tensors
    ]
    if missing_mtp:
        print(
            f"Note: Layer 47 MTP tensors not in GGUF (expected): "
            f"{', '.join(t.split('.')[-1] for t in missing_mtp)}"
        )

    if decode_errors:
        print(f"\nERROR: {len(decode_errors)} tensor decode/transform errors:", file=sys.stderr)
        for msg in decode_errors:
            print(f"  {msg}", file=sys.stderr)
        sys.exit(1)

    print(f"\nConverted {len(hf_tensors)} tensors total")

    # Verify coverage (excluding MTP layer 47 which is expected to be missing)
    mtp_set = set(MTP_TENSOR_NAMES)
    missing_ref = [name for name in ref_shapes if name not in hf_tensors and name not in mtp_set]
    if missing_ref:
        print(f"\nERROR: {len(missing_ref)} reference tensors not produced:", file=sys.stderr)
        for name in missing_ref[:_MAX_DISPLAY]:
            print(f"  {name}  {ref_shapes[name]}", file=sys.stderr)
        if len(missing_ref) > _MAX_DISPLAY:
            print(f"  ... and {len(missing_ref) - _MAX_DISPLAY} more", file=sys.stderr)
        sys.exit(1)

    write_shards(hf_tensors, output_path, shard_size_mb=shard_size_mb)

    del hf_tensors


def main():
    parser = argparse.ArgumentParser(
        description="Convert GGUF to safetensors for GLM-4.7-Flash (deepseek2 architecture)"
    )
    parser.add_argument("--gguf", required=True, help="Path to input GGUF file")
    parser.add_argument("--output", required=True, help="Output directory for safetensors")
    parser.add_argument(
        "--reference-model",
        required=True,
        help="Path to reference HF model directory (for name mapping and shapes)",
    )
    parser.add_argument("--shard-size-mb", type=int, default=4500)
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Preserve float16 tensors as-is instead of converting to bfloat16",
    )
    args = parser.parse_args()

    convert_gguf_to_safetensors_glm47(
        gguf_path=args.gguf,
        output_dir=args.output,
        reference_model=args.reference_model,
        shard_size_mb=args.shard_size_mb,
        keep_fp16=args.keep_fp16,
    )


if __name__ == "__main__":
    main()
