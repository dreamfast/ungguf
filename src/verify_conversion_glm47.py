"""Verify GGUF-to-safetensors conversion correctness for GLM-4.7-Flash.

Self-contained verifier for deepseek2 architecture (MLA + MoE).
Reconstructs kv_b_proj from GGUF k_b + v_b, splits expert 3D tensors,
and compares every tensor against the converted safetensors bit-for-bit.

No convention transforms needed for GLM-4.7 (unlike Qwen3.5).

Usage (inside Docker):
    python3 verify_conversion_glm47.py \\
        --gguf /input/model.gguf \\
        --converted /converted/ \\
        --reference /ref/ \\
        --output /results/glm47_verification.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader

from common import decode_gguf_tensor, load_converted_tensors, load_reference_shapes
from mappings_glm47 import (
    DENSE_MLP_MAP,
    EXPERT_HF_MAP,
    EXPERT_SUFFIXES,
    GLOBAL_MAP,
    KV_B_SUFFIXES,
    LAYER_SUFFIX_MAP,
    MOE_GATE_MAP,
    MOE_SHARED_MAP,
    reconstruct_kv_b,
)

_MAX_DISPLAY_MISMATCH = 50
_MAX_DISPLAY_OTHER = 20


def compare_tensors(
    expected: torch.Tensor,
    actual: torch.Tensor,
    gguf_label: str,
    hf_name: str,
    quant_type: str = "",
) -> dict[str, Any]:
    if list(expected.shape) != list(actual.shape):
        return {
            "gguf_name": gguf_label,
            "hf_name": hf_name,
            "error": f"shape mismatch: expected {list(expected.shape)} vs actual {list(actual.shape)}",
            "quant_type": quant_type,
        }

    dtype_note = None
    if expected.dtype != actual.dtype:
        dtype_note = f"dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
        match = torch.equal(expected.float(), actual.float())
    else:
        match = torch.equal(expected, actual)

    if match:
        entry: dict[str, Any] = {
            "gguf_name": gguf_label,
            "hf_name": hf_name,
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "quant_type": quant_type,
            "status": "MATCH",
        }
        if dtype_note:
            entry["dtype_note"] = dtype_note
        return entry

    diff = (expected.float() - actual.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    num_different = (diff > 0).sum().item()
    total_elements = diff.numel()

    entry = {
        "gguf_name": gguf_label,
        "hf_name": hf_name,
        "shape": list(actual.shape),
        "dtype_expected": str(expected.dtype),
        "dtype_actual": str(actual.dtype),
        "quant_type": quant_type,
        "status": "MISMATCH",
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "num_different_elements": num_different,
        "total_elements": total_elements,
        "pct_different": round(100.0 * num_different / total_elements, 4),
    }
    if dtype_note:
        entry["dtype_note"] = dtype_note

    if num_different > 0:
        flat_diff = diff.flatten()
        worst_idx = int(flat_diff.argmax().item())
        entry["sample_worst"] = {
            "index": worst_idx,
            "expected": expected.flatten()[worst_idx].float().item(),
            "actual": actual.flatten()[worst_idx].float().item(),
        }
        diff_indices = (flat_diff > 0).nonzero(as_tuple=True)[0][:5]
        entry["sample_diffs"] = [
            {
                "index": int(idx.item()),
                "expected": expected.flatten()[int(idx.item())].float().item(),
                "actual": actual.flatten()[int(idx.item())].float().item(),
            }
            for idx in diff_indices
        ]

    return entry


def verify(
    gguf_path: str,
    conv_dir: str,
    ref_dir: str,
    output_path: str | None,
    keep_fp16: bool = False,
):
    print(f"Loading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)
    gguf_tensors = reader.tensors
    print(f"  GGUF tensors: {len(gguf_tensors)}")

    qtypes: dict[str, int] = {}
    for t in gguf_tensors:
        qt = str(t.tensor_type)
        qtypes[qt] = qtypes.get(qt, 0) + 1
    print(f"  Quant types: {qtypes}")

    print(f"\nLoading reference shapes: {ref_dir}")
    ref_shapes = load_reference_shapes(ref_dir)
    print(f"  Reference tensors: {len(ref_shapes)}")

    print(f"\nLoading converted safetensors: {conv_dir}")
    conv_tensors = load_converted_tensors(conv_dir)
    print(f"  Converted tensors: {len(conv_tensors)}")

    gguf_by_name = {t.name: t for t in gguf_tensors}

    results: dict[str, Any] = {
        "gguf_path": gguf_path,
        "converted_dir": conv_dir,
        "reference_dir": ref_dir,
        "gguf_tensor_count": len(gguf_tensors),
        "converted_tensor_count": len(conv_tensors),
        "quant_types": qtypes,
        "matched": [],
        "mismatched": [],
        "unmapped_gguf": [],
        "missing_in_converted": [],
        "extra_in_converted": [],
    }

    verified_hf_names = set()

    # Phase 1: 1:1 mapped tensors
    print("\nPhase 1: Verifying 1:1 mapped tensors...")
    phase1_count = 0

    for gt in gguf_tensors:
        name = gt.name
        shape = [int(x) for x in gt.shape]

        hf_name = None

        if name in GLOBAL_MAP:
            hf_name = GLOBAL_MAP[name]
        else:
            m = re.match(r"blk\.(\d+)\.(.*)", name)
            if m:
                layer_num = int(m.group(1))
                suffix = m.group(2)

                if suffix in KV_B_SUFFIXES or suffix in EXPERT_SUFFIXES:
                    continue

                if suffix in LAYER_SUFFIX_MAP:
                    hf_name = f"model.layers.{layer_num}.{LAYER_SUFFIX_MAP[suffix]}"
                elif layer_num == 0 and suffix in DENSE_MLP_MAP:
                    hf_name = f"model.layers.0.{DENSE_MLP_MAP[suffix]}"
                elif layer_num >= 1 and suffix in MOE_SHARED_MAP:
                    hf_name = f"model.layers.{layer_num}.{MOE_SHARED_MAP[suffix]}"
                elif layer_num >= 1 and suffix in MOE_GATE_MAP:
                    hf_name = f"model.layers.{layer_num}.{MOE_GATE_MAP[suffix]}"
                else:
                    results["unmapped_gguf"].append(
                        {
                            "gguf_name": name,
                            "shape": shape,
                            "reason": "unknown suffix",
                        }
                    )
                    continue
            else:
                results["unmapped_gguf"].append(
                    {
                        "gguf_name": name,
                        "shape": shape,
                        "reason": "not global or layer tensor",
                    }
                )
                continue

        if hf_name is None:
            continue

        if hf_name not in ref_shapes:
            results["unmapped_gguf"].append(
                {
                    "gguf_name": name,
                    "shape": shape,
                    "reason": f"mapped to {hf_name} but not in reference",
                }
            )
            continue

        verified_hf_names.add(hf_name)

        if hf_name not in conv_tensors:
            results["missing_in_converted"].append(
                {
                    "gguf_name": name,
                    "hf_name": hf_name,
                    "shape": shape,
                }
            )
            continue

        try:
            decoded = decode_gguf_tensor(gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True)
        except Exception as e:
            results["mismatched"].append(
                {
                    "gguf_name": name,
                    "hf_name": hf_name,
                    "error": f"GGUF decode failed: {e}",
                }
            )
            continue

        target_shape = ref_shapes.get(hf_name)
        if target_shape and list(decoded.shape) != target_shape:
            if decoded.numel() == int(np.prod(target_shape)):
                decoded = decoded.reshape(target_shape).contiguous()
            else:
                results["mismatched"].append(
                    {
                        "gguf_name": name,
                        "hf_name": hf_name,
                        "error": f"element count mismatch: GGUF {decoded.numel()} vs ref {int(np.prod(target_shape))}",
                    }
                )
                continue

        actual = conv_tensors[hf_name]
        result = compare_tensors(decoded, actual, name, hf_name, str(gt.tensor_type))

        if result.get("status") == "MATCH":
            results["matched"].append(result)
        else:
            results["mismatched"].append(result)

        phase1_count += 1

    print(f"  Verified {phase1_count} direct 1:1 tensors")

    # Phase 2: kv_b_proj reconstruction
    print("\nPhase 2: Verifying kv_b_proj reconstruction...")
    phase2_count = 0

    kv_b_pairs: dict[int, list] = {}
    for gt in gguf_tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", gt.name)
        if not m:
            continue
        layer_num = int(m.group(1))
        suffix = m.group(2)
        if suffix in KV_B_SUFFIXES:
            if layer_num not in kv_b_pairs:
                kv_b_pairs[layer_num] = [None, None]
            if suffix == "attn_k_b.weight":
                kv_b_pairs[layer_num][0] = gt.name
            else:
                kv_b_pairs[layer_num][1] = gt.name

    for layer_num in sorted(kv_b_pairs.keys()):
        k_b_name, v_b_name = kv_b_pairs[layer_num]
        hf_name = f"model.layers.{layer_num}.self_attn.kv_b_proj.weight"
        verified_hf_names.add(hf_name)

        if k_b_name is None or v_b_name is None:
            results["mismatched"].append(
                {
                    "gguf_name": f"blk.{layer_num}.attn_{{k,v}}_b.weight",
                    "hf_name": hf_name,
                    "error": f"incomplete pair: k_b={k_b_name}, v_b={v_b_name}",
                }
            )
            continue

        if hf_name not in conv_tensors:
            results["missing_in_converted"].append(
                {
                    "gguf_name": f"{k_b_name} + {v_b_name}",
                    "hf_name": hf_name,
                }
            )
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

            reconstructed = reconstruct_kv_b(k_b_t, v_b_t)

            target_shape = ref_shapes.get(hf_name)
            if (
                target_shape
                and list(reconstructed.shape) != target_shape
                and reconstructed.numel() == int(np.prod(target_shape))
            ):
                reconstructed = reconstructed.reshape(target_shape).contiguous()

            actual = conv_tensors[hf_name]
            result = compare_tensors(
                reconstructed,
                actual,
                f"{k_b_name} + {v_b_name}",
                hf_name,
                f"{k_b_gt.tensor_type} + {v_b_gt.tensor_type}",
            )

            if result.get("status") == "MATCH":
                results["matched"].append(result)
            else:
                results["mismatched"].append(result)

        except Exception as e:
            results["mismatched"].append(
                {
                    "gguf_name": f"{k_b_name} + {v_b_name}",
                    "hf_name": hf_name,
                    "error": f"reconstruction failed: {e}",
                }
            )

        phase2_count += 1

    print(f"  Verified {phase2_count} kv_b_proj reconstructions")

    # Phase 3: Expert tensor splitting
    print("\nPhase 3: Verifying expert tensor splits...")
    phase3_count = 0
    phase3_experts = 0

    expert_tensors = []
    for gt in gguf_tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", gt.name)
        if not m:
            continue
        layer_num = int(m.group(1))
        suffix = m.group(2)
        if suffix in EXPERT_SUFFIXES:
            expert_tensors.append((gt.name, layer_num, suffix))

    for gguf_name, layer_num, suffix in expert_tensors:
        gt = gguf_by_name[gguf_name]
        hf_proj_name = EXPERT_HF_MAP[suffix]

        try:
            expert_3d = decode_gguf_tensor(
                gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=True
            )
            num_experts = expert_3d.shape[0]

            for expert_i in range(num_experts):
                hf_name = f"model.layers.{layer_num}.mlp.experts.{expert_i}.{hf_proj_name}"
                verified_hf_names.add(hf_name)

                if hf_name not in conv_tensors:
                    results["missing_in_converted"].append(
                        {
                            "gguf_name": gguf_name,
                            "hf_name": hf_name,
                            "expert_index": expert_i,
                        }
                    )
                    continue

                expert_slice = expert_3d[expert_i]

                target_shape = ref_shapes.get(hf_name)
                if target_shape and expert_slice.numel() == int(np.prod(target_shape)):
                    expert_slice = expert_slice.reshape(target_shape).contiguous()

                actual = conv_tensors[hf_name]
                result = compare_tensors(
                    expert_slice,
                    actual,
                    f"{gguf_name}[..., {expert_i}]",
                    hf_name,
                    str(gt.tensor_type),
                )

                if result.get("status") == "MATCH":
                    results["matched"].append(result)
                else:
                    results["mismatched"].append(result)

                phase3_experts += 1

        except Exception as e:
            results["mismatched"].append(
                {
                    "gguf_name": gguf_name,
                    "hf_name": f"model.layers.{layer_num}.mlp.experts.*.{hf_proj_name}",
                    "error": f"expert split failed: {e}",
                }
            )

        phase3_count += 1

    print(f"  Verified {phase3_count} expert 3D tensors -> {phase3_experts} individual experts")

    # Phase 4: Check for extra/missing tensors in converted
    for hf_name in conv_tensors:
        if hf_name not in verified_hf_names:
            results["extra_in_converted"].append(
                {
                    "hf_name": hf_name,
                    "shape": list(conv_tensors[hf_name].shape),
                    "dtype": str(conv_tensors[hf_name].dtype),
                }
            )

    summary = {
        "total_gguf_tensors": len(gguf_tensors),
        "total_converted_tensors": len(conv_tensors),
        "mapped_and_matched": len(results["matched"]),
        "mapped_and_mismatched": len(results["mismatched"]),
        "unmapped_gguf": len(results["unmapped_gguf"]),
        "missing_in_converted": len(results["missing_in_converted"]),
        "extra_in_converted": len(results["extra_in_converted"]),
        "conversion_correct": (
            len(results["mismatched"]) == 0 and len(results["missing_in_converted"]) == 0
        ),
    }
    results["summary"] = summary

    print(f"\n{'=' * 60}")
    print("GLM-4.7 VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  GGUF tensors:           {summary['total_gguf_tensors']}")
    print(f"  Converted tensors:      {summary['total_converted_tensors']}")
    print(f"  Matched (bit-exact):    {summary['mapped_and_matched']}")
    print(f"  MISMATCHED:             {summary['mapped_and_mismatched']}")
    print(f"  Unmapped GGUF:          {summary['unmapped_gguf']}")
    print(f"  Missing in converted:   {summary['missing_in_converted']}")
    print(f"  Extra in converted:     {summary['extra_in_converted']}")
    print(f"  CONVERSION CORRECT:     {'YES' if summary['conversion_correct'] else '*** NO ***'}")
    print(f"{'=' * 60}")

    if results["mismatched"]:
        print(f"\nMISMATCHED TENSORS ({len(results['mismatched'])}):")
        for m in results["mismatched"][:_MAX_DISPLAY_MISMATCH]:
            print(f"  {m.get('hf_name', m.get('gguf_name', '???'))}")
            if "error" in m:
                print(f"    Error: {m['error']}")
            else:
                print(
                    f"    max_diff={m.get('max_abs_diff', '?'):.6e}, "
                    f"mean_diff={m.get('mean_abs_diff', '?'):.6e}"
                )
                print(
                    f"    {m.get('num_different_elements', '?')}/{m.get('total_elements', '?')} "
                    f"elements differ ({m.get('pct_different', '?')}%)"
                )
                if "sample_worst" in m:
                    s = m["sample_worst"]
                    print(f"    worst: expected={s['expected']:.8f}, actual={s['actual']:.8f}")
        if len(results["mismatched"]) > _MAX_DISPLAY_MISMATCH:
            print(f"  ... and {len(results['mismatched']) - _MAX_DISPLAY_MISMATCH} more")

    if results["unmapped_gguf"]:
        print(f"\nUNMAPPED GGUF TENSORS ({len(results['unmapped_gguf'])}):")
        for u in results["unmapped_gguf"]:
            print(f"  {u['gguf_name']}  shape={u.get('shape')}  reason={u.get('reason')}")

    if results["missing_in_converted"]:
        print(f"\nMISSING IN CONVERTED ({len(results['missing_in_converted'])}):")
        for m in results["missing_in_converted"][:_MAX_DISPLAY_OTHER]:
            print(f"  {m['hf_name']}")

    if results["extra_in_converted"]:
        print(f"\nEXTRA IN CONVERTED ({len(results['extra_in_converted'])}):")
        for entry in results["extra_in_converted"][:_MAX_DISPLAY_OTHER]:
            print(f"  {entry['hf_name']}  shape={entry['shape']}")

    if output_path:
        output_results = dict(results)
        output_results["matched"] = [
            {
                "gguf_name": m["gguf_name"],
                "hf_name": m["hf_name"],
                "quant_type": m.get("quant_type", ""),
            }
            for m in results["matched"]
        ]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return summary["conversion_correct"]


def main():
    parser = argparse.ArgumentParser(
        description="Verify GGUF-to-safetensors conversion for GLM-4.7-Flash (deepseek2)"
    )
    parser.add_argument("--gguf", required=True, help="Path to original GGUF file")
    parser.add_argument(
        "--converted", required=True, help="Path to converted safetensors directory"
    )
    parser.add_argument("--reference", required=True, help="Path to reference HF model directory")
    parser.add_argument("--output", help="Path to save JSON results")
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Expect float16 tensors preserved (not converted to bfloat16)",
    )
    args = parser.parse_args()

    ok = verify(args.gguf, args.converted, args.reference, args.output, keep_fp16=args.keep_fp16)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
