"""Verify Qwen3.6-27B GGUF-to-safetensors conversion correctness.

Loads the original GGUF file and the converted safetensors side by side,
applies the same conventions (norm-1.0, A_log, V-head inverse reorder),
and verifies every GGUF-derived tensor matches bit-for-bit.

MTP and vision tensors are verified by checking they exist in the converted
output and match the reference model (since they were copied verbatim).

This is the definitive test for conversion integrity — if every tensor
matches, the converted model is a faithful representation of the GGUF.

Usage (inside Docker):
    python3 verify_conversion_qwen36.py \\
        --gguf /input/model.gguf \\
        --converted /converted/ \\
        --reference /ref/ \\
        --output /results/conversion_verification.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from gguf import GGUFReader

from common import decode_gguf_tensor, load_converted_tensors, load_reference_shapes
from mappings_qwen35 import (
    apply_conventions,
    apply_inverse_v_reorder,
    get_linear_attn_config,
    gguf_name_to_hf,
)

_MISSING_PREFIXES = (
    "mtp.",
    "model.visual.",
)


def _is_missing_tensor(hf_name: str) -> bool:
    """Check if a tensor name belongs to MTP or vision modules (absent from GGUF)."""
    return any(hf_name.startswith(p) for p in _MISSING_PREFIXES)


def resolve_hf_name(hf_name: str, ref_shapes: dict[str, list[int]]) -> str | None:
    for candidate in (
        hf_name,
        hf_name.replace("model.language_model.", "model."),
        hf_name.replace("model.language_model.", ""),
    ):
        if candidate in ref_shapes:
            return candidate
    return None


def verify(
    gguf_path: str,
    conv_dir: str,
    ref_dir: str,
    output_path: str | None,
    keep_fp16: bool = False,
) -> bool:
    print(f"Loading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)
    gguf_tensors = reader.tensors
    print(f"  GGUF tensors: {len(gguf_tensors)}")

    qtypes: dict[str, int] = {}
    for t in gguf_tensors:
        qt = str(t.tensor_type)
        qtypes[qt] = qtypes.get(qt, 0) + 1
    print(f"  Quant types: {qtypes}")

    config_path = Path(ref_dir) / "config.json"
    if config_path.exists():
        with config_path.open() as f:
            model_config = json.load(f)
        v_cfg = get_linear_attn_config(model_config)
    else:
        v_cfg = None
    if v_cfg:
        print(
            f"  V-head reorder: ACTIVE (v_per_k={v_cfg['v_per_k']}, "
            f"k_heads={v_cfg['num_k_heads']}, v_heads={v_cfg['num_v_heads']})"
        )
    else:
        print("  V-head reorder: not needed (v_per_k=1 or config not found)")

    print(f"\nLoading reference shapes: {ref_dir}")
    ref_shapes = load_reference_shapes(ref_dir)
    print(f"  Reference tensors: {len(ref_shapes)}")

    print(f"\nLoading converted safetensors: {conv_dir}")
    conv_tensors = load_converted_tensors(conv_dir)
    print(f"  Converted tensors: {len(conv_tensors)}")

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
        "copied_matched": [],
        "copied_mismatched": [],
    }

    mapped_hf_names: set[str] = set()

    # --- Part 1: Verify GGUF-derived tensors ---
    print("\nVerifying GGUF-derived tensors...")
    for gt in gguf_tensors:
        raw_hf = gguf_name_to_hf(gt.name)
        if raw_hf is None:
            results["unmapped_gguf"].append(
                {
                    "gguf_name": gt.name,
                    "shape": [int(x) for x in gt.shape],
                    "reason": "no name mapping rule",
                }
            )
            continue

        hf_name = resolve_hf_name(raw_hf, ref_shapes)
        if hf_name is None:
            results["unmapped_gguf"].append(
                {
                    "gguf_name": gt.name,
                    "mapped_to": raw_hf,
                    "shape": [int(x) for x in gt.shape],
                    "reason": "mapped name not in reference model",
                }
            )
            continue

        mapped_hf_names.add(hf_name)

        if hf_name not in conv_tensors:
            results["missing_in_converted"].append(
                {
                    "gguf_name": gt.name,
                    "hf_name": hf_name,
                    "shape": [int(x) for x in gt.shape],
                }
            )
            continue

        try:
            # Qwen3.6 GGUF tensors are stored in C-order (row-major); do NOT reverse.
            decoded = decode_gguf_tensor(
                gt, keep_f32=True, keep_f16=keep_fp16, reverse_shape=False
            )
        except Exception as e:
            results["mismatched"].append(
                {
                    "gguf_name": gt.name,
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
                        "gguf_name": gt.name,
                        "hf_name": hf_name,
                        "error": (
                            f"element count mismatch: GGUF {decoded.numel()} "
                            f"vs ref {int(np.prod(target_shape))}"
                        ),
                    }
                )
                continue

        expected = apply_conventions(hf_name, decoded)

        if v_cfg:
            expected = apply_inverse_v_reorder(expected, hf_name, v_cfg)

        actual = conv_tensors[hf_name]

        if expected.dtype != actual.dtype:
            dtype_note = f"dtype mismatch: expected {expected.dtype}, got {actual.dtype}"
            match = torch.equal(expected.float(), actual.float())
        else:
            dtype_note = None
            match = torch.equal(expected, actual)

        if match:
            entry: dict[str, Any] = {
                "gguf_name": gt.name,
                "hf_name": hf_name,
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "quant_type": str(gt.tensor_type),
            }
            if dtype_note:
                entry["dtype_note"] = dtype_note
            results["matched"].append(entry)
        else:
            diff = (expected.float() - actual.float()).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            num_different = (diff > 0).sum().item()
            total_elements = diff.numel()

            raw_match = torch.equal(decoded.float(), actual.float())
            raw_vs_conv_diff = (decoded.float() - actual.float()).abs()

            entry = {
                "gguf_name": gt.name,
                "hf_name": hf_name,
                "shape": list(actual.shape),
                "dtype_expected": str(expected.dtype),
                "dtype_actual": str(actual.dtype),
                "quant_type": str(gt.tensor_type),
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "num_different_elements": num_different,
                "total_elements": total_elements,
                "pct_different": round(100.0 * num_different / total_elements, 4),
                "raw_gguf_matches_converted": raw_match,
                "raw_vs_conv_max_diff": raw_vs_conv_diff.max().item(),
            }
            if dtype_note:
                entry["dtype_note"] = dtype_note

            if num_different > 0:
                flat_diff = diff.flatten()
                worst_idx = flat_diff.argmax().item()
                entry["sample_worst"] = {
                    "index": worst_idx,
                    "expected": expected.flatten()[worst_idx].float().item(),
                    "actual": actual.flatten()[worst_idx].float().item(),
                }
                diff_indices = (flat_diff > 0).nonzero(as_tuple=True)[0][:5]
                entry["sample_diffs"] = [
                    {
                        "index": idx.item(),
                        "expected": expected.flatten()[idx].float().item(),
                        "actual": actual.flatten()[idx].float().item(),
                    }
                    for idx in diff_indices
                ]

            results["mismatched"].append(entry)

    # --- Part 2: Verify copied (MTP/vision) tensors against reference ---
    print("Verifying copied (MTP/vision) tensors against reference...")
    ref_tensors = load_converted_tensors(ref_dir)
    for hf_name in conv_tensors:
        if hf_name in mapped_hf_names:
            continue
        if not _is_missing_tensor(hf_name):
            results["extra_in_converted"].append(
                {
                    "hf_name": hf_name,
                    "shape": list(conv_tensors[hf_name].shape),
                    "dtype": str(conv_tensors[hf_name].dtype),
                }
            )
            continue

        # This is a copied tensor — verify against reference
        if hf_name not in ref_tensors:
            results["extra_in_converted"].append(
                {
                    "hf_name": hf_name,
                    "shape": list(conv_tensors[hf_name].shape),
                    "dtype": str(conv_tensors[hf_name].dtype),
                    "note": "not in reference model either",
                }
            )
            continue

        actual = conv_tensors[hf_name]
        ref = ref_tensors[hf_name]
        if torch.equal(actual, ref):
            results["copied_matched"].append(
                {"hf_name": hf_name, "shape": list(actual.shape), "dtype": str(actual.dtype)}
            )
        else:
            diff = (actual.float() - ref.float()).abs()
            results["copied_mismatched"].append(
                {
                    "hf_name": hf_name,
                    "shape": list(actual.shape),
                    "dtype_actual": str(actual.dtype),
                    "dtype_ref": str(ref.dtype),
                    "max_abs_diff": diff.max().item(),
                }
            )

    # Also check for missing copied tensors
    for hf_name in ref_shapes:
        if _is_missing_tensor(hf_name) and hf_name not in conv_tensors:
            results["missing_in_converted"].append(
                {"hf_name": hf_name, "shape": ref_shapes[hf_name], "note": "copied tensor missing"}
            )

    # --- Summary ---
    conversion_correct: bool = (
        len(results["mismatched"]) == 0
        and len(results["missing_in_converted"]) == 0
        and len(results["copied_mismatched"]) == 0
    )
    summary: dict[str, Any] = {
        "total_gguf_tensors": len(gguf_tensors),
        "total_converted_tensors": len(conv_tensors),
        "gguf_matched": len(results["matched"]),
        "gguf_mismatched": len(results["mismatched"]),
        "copied_matched": len(results["copied_matched"]),
        "copied_mismatched": len(results["copied_mismatched"]),
        "unmapped_gguf": len(results["unmapped_gguf"]),
        "missing_in_converted": len(results["missing_in_converted"]),
        "extra_in_converted": len(results["extra_in_converted"]),
        "conversion_correct": conversion_correct,
    }
    results["summary"] = summary

    print(f"\n{'=' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  GGUF tensors:                {summary['total_gguf_tensors']}")
    print(f"  Converted tensors:           {summary['total_converted_tensors']}")
    print(f"  GGUF matched (bit-exact):    {summary['gguf_matched']}")
    print(f"  GGUF MISMATCHED:             {summary['gguf_mismatched']}")
    print(f"  Copied matched (bit-exact):  {summary['copied_matched']}")
    print(f"  Copied MISMATCHED:           {summary['copied_mismatched']}")
    print(f"  Unmapped GGUF:               {summary['unmapped_gguf']}")
    print(f"  Missing in converted:        {summary['missing_in_converted']}")
    print(f"  Extra in converted:          {summary['extra_in_converted']}")
    print(
        f"  CONVERSION CORRECT:          {'YES' if summary['conversion_correct'] else '*** NO ***'}"
    )
    print(f"{'=' * 60}")

    if results["mismatched"]:
        print("\nGGUF MISMATCHED TENSORS:")
        for m in results["mismatched"]:
            print(f"  {m['hf_name']}")
            if "error" in m:
                print(f"    Error: {m['error']}")
            else:
                print(f"    max_diff={m['max_abs_diff']:.6e}, mean_diff={m['mean_abs_diff']:.6e}")
                print(
                    f"    {m['num_different_elements']}/{m['total_elements']}"
                    f" elements differ ({m['pct_different']}%)"
                )

    if results["copied_mismatched"]:
        print("\nCOPIED (MTP/VISION) MISMATCHED TENSORS:")
        for m in results["copied_mismatched"]:
            print(f"  {m['hf_name']}: max_diff={m['max_abs_diff']:.6e}")

    if results["unmapped_gguf"]:
        print("\nUNMAPPED GGUF TENSORS:")
        for u in results["unmapped_gguf"]:
            print(f"  {u['gguf_name']}  shape={u.get('shape')}  reason={u.get('reason')}")

    if output_path:
        output_results = dict(results)
        output_results["matched"] = [
            {"gguf_name": m["gguf_name"], "hf_name": m["hf_name"], "quant_type": m["quant_type"]}
            for m in results["matched"]
        ]
        output_results["copied_matched"] = [
            {"hf_name": m["hf_name"]} for m in results["copied_matched"]
        ]
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    return conversion_correct


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Qwen3.6 GGUF-to-safetensors conversion")
    parser.add_argument("--gguf", required=True, help="Path to original GGUF file")
    parser.add_argument(
        "--converted", required=True, help="Path to converted safetensors directory"
    )
    parser.add_argument(
        "--reference", required=True, help="Path to base model safetensors directory"
    )
    parser.add_argument("--output", help="Path to save JSON results")
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Expect float16 tensors to be preserved (not converted to bfloat16)",
    )
    args = parser.parse_args()

    ok = verify(args.gguf, args.converted, args.reference, args.output, keep_fp16=args.keep_fp16)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
