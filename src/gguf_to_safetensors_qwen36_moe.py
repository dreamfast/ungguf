"""Convert Qwen3.6-35B-A3B GGUF model files to HuggingFace safetensors format.

Qwen3.6-35B-A3B is a Mixture-of-Experts model (Qwen3_5MoeForConditionalGeneration)
with 256 experts per layer, 40 layers, hidden_size=2048, 3B active params.

Key differences from the dense Qwen3.6-27B converter:
  - MoE MLP: expert tensors are stacked in GGUF (all 256 in one tensor) and need
    concatenation (gate+up -> gate_up_proj) and permutation to match HF layout
  - Shared expert projections need transposition (GGUF [H,I] vs HF [I,H])
  - Router gate needs transposition
  - V-head reorder uses v_per_k=2 (16 K-heads / 32 V-heads)
  - MTP and vision tensors are NOT in the GGUF — copied from reference model

Architecture: Hybrid Gated DeltaNet + Gated Attention with MoE
  - 40 layers, hidden_size=2048, moe_intermediate_size=512
  - 256 experts, 8 active per token
  - 3 linear attention + 1 full attention per block of 4
  - reverse_shape=True (GGML F-order → C-order, same as 27B converter)

Usage (via ungguf.sh):
    ungguf convert-qwen36-moe <gguf_file> <output_dir> <ref_model_dir>
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
from mappings_qwen36_moe import (
    apply_conventions,
    apply_inverse_v_reorder,
    build_expert_tensors,
    get_expert_layer,
    get_expert_suffix,
    get_linear_attn_config,
    gguf_name_to_hf,
    is_missing_tensor,
    resolve_hf_name,
)

_MAX_DISPLAY = 30


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
    gguf_expected = {k: v for k, v in ref_shapes.items() if not is_missing_tensor(k)}
    missing_in_gguf = {k: v for k, v in ref_shapes.items() if is_missing_tensor(k)}
    print(f"GGUF-expected reference tensors: {len(gguf_expected)}")
    print(f"Missing (MTP/vision) tensors to copy from reference: {len(missing_in_gguf)}")

    # Build GGUF -> HF name mapping for standard tensors
    mapping: dict[str, str] = {}
    expert_tensors: dict[int, dict[str, object]] = {}  # layer -> {suffix: gguf_tensor}
    unmapped: list[tuple[str, list[int], str]] = []

    for gt in gguf_tensors:
        # Check if this is an expert tensor
        exp_layer = get_expert_layer(gt.name)
        if exp_layer is not None:
            exp_suffix = get_expert_suffix(gt.name)
            if exp_suffix:
                if exp_layer not in expert_tensors:
                    expert_tensors[exp_layer] = {}
                expert_tensors[exp_layer][exp_suffix] = gt
                continue

        hf_name = gguf_name_to_hf(gt.name)
        if hf_name is None:
            unmapped.append((gt.name, [int(x) for x in gt.shape], "no name mapping rule"))
            continue

        resolved = resolve_hf_name(hf_name, gguf_expected)
        if resolved:
            mapping[gt.name] = resolved
        else:
            unmapped.append(
                (gt.name, [int(x) for x in gt.shape], f"mapped to {hf_name} but not in ref")
            )

    print(f"Mapped {len(mapping)} / {len(gguf_tensors)} standard GGUF tensors to HF names")
    print(f"Expert tensor groups: {len(expert_tensors)} layers")

    if unmapped:
        print(f"\nERROR: {len(unmapped)} unmatched GGUF tensors:", file=sys.stderr)
        for name, shape, reason in unmapped[:_MAX_DISPLAY]:
            print(f"  {name}  {shape}  {reason}", file=sys.stderr)
        if len(unmapped) > _MAX_DISPLAY:
            print(f"  ... and {len(unmapped) - _MAX_DISPLAY} more", file=sys.stderr)
        sys.exit(1)

    # Check for duplicate HF targets
    hf_targets = list(mapping.values())
    seen_targets: set[str] = set()
    dupes: list[str] = []
    for ht in hf_targets:
        if ht in seen_targets:
            dupes.append(ht)
        seen_targets.add(ht)
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

    # Decode and transform all standard GGUF tensors
    hf_tensors: dict[str, torch.Tensor] = {}
    decode_errors: list[str] = []

    for gt in gguf_tensors:
        if gt.name not in mapping:
            continue
        try:
            # All tensors use reverse_shape=True (GGML F-order → C-order).
            # This matches the 27B converter and gives correct data layout directly.
            tensor = decode_gguf_tensor(gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True)
            hf_name = mapping[gt.name]

            # Align shape to match HF reference — only reshape, never transpose.
            # With reverse_shape=True, 2D weights already have correct [out, in] layout.
            # Some tensors need minor reshapes (e.g. 1D→2D, 2D→3D for conv1d).
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

            # Apply value-only conventions (norm-1.0, A_log)
            tensor = apply_conventions(hf_name, tensor)

            # Apply V-head reorder for linear attention tensors
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

    # Process expert tensors (stacked GGUF -> individual HF tensors)
    print(f"\nProcessing expert tensors for {len(expert_tensors)} layers...")
    expert_errors: list[str] = []
    for layer_num in sorted(expert_tensors.keys()):
        layer_tensors = expert_tensors[layer_num]
        needed = ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight")
        if not all(s in layer_tensors for s in needed):
            expert_errors.append(
                f"Layer {layer_num}: missing expert tensors: "
                f"{[s for s in needed if s not in layer_tensors]}"
            )
            continue

        try:
            # Expert tensors use GGML/Fortran order — must reverse shape
            gate = decode_gguf_tensor(
                layer_tensors["ffn_gate_exps.weight"],
                keep_f32=True,
                keep_f16=keep_fp16,
                reverse_shape=True,
            )
            up = decode_gguf_tensor(
                layer_tensors["ffn_up_exps.weight"],
                keep_f32=True,
                keep_f16=keep_fp16,
                reverse_shape=True,
            )
            down = decode_gguf_tensor(
                layer_tensors["ffn_down_exps.weight"],
                keep_f32=True,
                keep_f16=keep_fp16,
                reverse_shape=True,
            )

            gate_up, down_proj = build_expert_tensors(gate, up, down)

            # Free decoded GGUF expert tensors to reduce peak memory
            del gate, up, down

            # Resolve expert tensor names via shared helper
            gate_up_raw = f"model.language_model.layers.{layer_num}.mlp.experts.gate_up_proj"
            down_raw = f"model.language_model.layers.{layer_num}.mlp.experts.down_proj"
            gate_up_name = resolve_hf_name(gate_up_raw, gguf_expected) or gate_up_raw
            down_name = resolve_hf_name(down_raw, gguf_expected) or down_raw

            # Validate shapes against reference
            for hf_name, tensor in [(gate_up_name, gate_up), (down_name, down_proj)]:
                target = gguf_expected.get(hf_name)
                if target and list(tensor.shape) != target:
                    expert_errors.append(
                        f"{hf_name}: expert shape {list(tensor.shape)} vs ref {target}"
                    )
                    continue
                hf_tensors[hf_name] = tensor

        except Exception as e:
            expert_errors.append(f"Layer {layer_num} expert tensors: {e}")

    if expert_errors:
        print(f"\nERROR: {len(expert_errors)} expert tensor errors:", file=sys.stderr)
        for msg in expert_errors:
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

    total_gguf = len(gguf_tensors)
    print(f"\nTotal tensors: {len(hf_tensors)} ({total_gguf} from GGUF + {copied} from reference)")
    write_shards(hf_tensors, output_path, shard_size_mb=shard_size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.6-35B-A3B MoE GGUF to safetensors using reference model"
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
