"""Dump ALL metadata and tensor names from GGUF files for forensic analysis."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from gguf import GGUFReader

_MAX_LIST_PREVIEW = 50
_MAX_STR_PREVIEW = 200

HIGHLIGHT_PATTERNS = [
    "mamba",
    "ssm",
    "hybrid",
    "layer_type",
    "recurrent",
    "qwen3moe",
    "qwen3",
    "attention",
    "architecture",
    "block_count",
    "head_count",
    "embedding_length",
    "feed_forward",
    "context_length",
    "rope",
    "expert",
    "state_size",
    "conv_kernel",
    "inner_size",
    "time_step",
    "dt_rank",
    "chunk_size",
    "group",
    "n_head",
    "n_embd",
    "n_layer",
]


def is_highlight(key: str) -> bool:
    kl = key.lower()
    return any(p in kl for p in HIGHLIGHT_PATTERNS)


def format_value(val):  # noqa: PLR0911
    if isinstance(val, (list, tuple)):
        if len(val) > _MAX_LIST_PREVIEW:
            return f"[list of {len(val)} items: {val[:10]}...]"
        return str(val)
    if isinstance(val, bytes):
        try:
            s = val.decode("utf-8", errors="replace")
            if len(s) > _MAX_STR_PREVIEW:
                return s[:_MAX_STR_PREVIEW] + "..."
            return s
        except Exception:
            return f"<bytes len={len(val)}>"
    if isinstance(val, str) and len(val) > _MAX_STR_PREVIEW:
        return val[:_MAX_STR_PREVIEW] + "..."
    return str(val)


def extract_field_value(field):  # noqa: PLR0911
    try:
        parts = field.parts
        data_indices = field.data
        if len(data_indices) == 0:
            if len(parts) > 0:
                val = parts[-1]
                if hasattr(val, "tobytes"):
                    try:
                        return val.tobytes().decode("utf-8")
                    except Exception:  # noqa: S110
                        pass
                if hasattr(val, "item"):
                    return val.item()
                if hasattr(val, "tolist"):
                    return val.tolist()
                return val
            return "<empty>"

        if len(data_indices) == 1:
            idx = data_indices[0]
            val = parts[idx]
            if hasattr(val, "item"):
                return val.item()
            if hasattr(val, "tobytes"):
                try:
                    return val.tobytes().decode("utf-8")
                except Exception:
                    return val.tolist() if hasattr(val, "tolist") else val
            return val
        else:
            values = []
            for idx in data_indices:
                v = parts[idx]
                if hasattr(v, "item"):
                    values.append(v.item())
                elif hasattr(v, "tobytes"):
                    try:
                        values.append(v.tobytes().decode("utf-8"))
                    except Exception:
                        values.append(v.tolist() if hasattr(v, "tolist") else v)
                else:
                    values.append(v)
            return values
    except Exception as e:
        return f"<error extracting: {e}>"


def dump_gguf(gguf_path: str):
    path = Path(gguf_path)
    if not path.exists():
        print(f"  FILE NOT FOUND: {gguf_path}")
        return

    print(f"\n{'=' * 100}")
    print(f"  GGUF FILE: {path.name}")
    print(f"  SIZE: {path.stat().st_size / 1024**3:.2f} GB")
    print(f"{'=' * 100}")

    reader = GGUFReader(gguf_path)

    fields = reader.fields
    print(f"\n--- ALL METADATA ({len(fields)} keys) ---\n")

    highlighted = []
    all_keys = []

    for key, field in sorted(fields.items()):
        val = extract_field_value(field)
        val_str = format_value(val)
        marker = ""
        if is_highlight(key):
            marker = "  <<<< ARCHITECTURE KEY"
            highlighted.append((key, val_str))
        print(f"  {key} = {val_str}{marker}")
        all_keys.append(key)

    if highlighted:
        print("\n--- ARCHITECTURE-RELATED KEYS (highlighted) ---\n")
        for key, val in highlighted:
            print(f"  {key} = {val}")

    tensors = reader.tensors
    print(f"\n--- ALL TENSOR NAMES ({len(tensors)} tensors) ---\n")

    ssm_tensors = []
    attn_tensors = []
    other_tensors = []

    for t in tensors:
        name = t.name
        shape = [int(x) for x in t.shape]
        qtype = str(t.tensor_type)
        line = f"  {name:60s}  shape={shape!s:30s}  qtype={qtype}"

        if any(k in name for k in ["ssm_", "ssm.", "mamba"]):
            ssm_tensors.append(line)
        elif any(
            k in name
            for k in ["attn_q.", "attn_k.", "attn_v.", "attn_output", "attn_qkv", "attn_gate"]
        ):
            attn_tensors.append(line)
        else:
            other_tensors.append(line)

    print(f"  --- SSM/Mamba tensors ({len(ssm_tensors)}) ---")
    for line in ssm_tensors:
        print(line)

    print(f"\n  --- Attention tensors ({len(attn_tensors)}) ---")
    for line in attn_tensors:
        print(line)

    print(f"\n  --- Other tensors ({len(other_tensors)}) ---")
    for line in other_tensors:
        print(line)

    print("\n--- LAYER TYPE ANALYSIS ---\n")
    layer_types: dict[int, dict[str, list[str]]] = {}
    for t in tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", t.name)
        if m:
            layer_num = int(m.group(1))
            suffix = m.group(2)
            if layer_num not in layer_types:
                layer_types[layer_num] = {"ssm_keys": [], "attn_keys": [], "other_keys": []}
            if any(k in suffix for k in ["ssm_", "ssm."]):
                layer_types[layer_num]["ssm_keys"].append(suffix)
            elif any(
                k in suffix for k in ["attn_q.", "attn_k.", "attn_v.", "attn_output"]
            ) or suffix in ("attn_qkv.weight", "attn_gate.weight"):
                layer_types[layer_num]["attn_keys"].append(suffix)
            else:
                layer_types[layer_num]["other_keys"].append(suffix)

    for layer_num in sorted(layer_types.keys()):
        info = layer_types[layer_num]
        has_ssm = len(info["ssm_keys"]) > 0
        has_attn = len(info["attn_keys"]) > 0

        if has_ssm and not any(
            k in info["attn_keys"]
            for k in ["attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"]
        ):
            layer_type = "MAMBA/SSM"
        elif has_ssm:
            layer_type = "HYBRID (SSM+ATTN)"
        elif has_attn:
            layer_type = "ATTENTION"
        else:
            layer_type = "OTHER"

        ssm_str = ", ".join(info["ssm_keys"][:5])
        attn_str = ", ".join(info["attn_keys"][:5])
        print(
            f"  Layer {layer_num:3d}: {layer_type:20s}  ssm=[{ssm_str}]  attn=[{attn_str}]  other={info['other_keys']}"
        )

    mamba_layers = [
        ln
        for ln, info in layer_types.items()
        if len(info["ssm_keys"]) > 0
        and not any(k in info["attn_keys"] for k in ["attn_q.weight", "attn_k.weight"])
    ]
    attn_layers = [
        ln
        for ln, info in layer_types.items()
        if len(info["ssm_keys"]) == 0 and len(info["attn_keys"]) > 0
    ]
    hybrid_layers = [
        ln
        for ln, info in layer_types.items()
        if len(info["ssm_keys"]) > 0
        and any(k in info["attn_keys"] for k in ["attn_q.weight", "attn_k.weight"])
    ]

    print("\n  SUMMARY:")
    print(f"    Total layers: {len(layer_types)}")
    print(f"    Mamba/SSM layers ({len(mamba_layers)}): {mamba_layers}")
    print(f"    Attention layers ({len(attn_layers)}): {attn_layers}")
    print(f"    Hybrid layers ({len(hybrid_layers)}): {hybrid_layers}")

    return {
        "total_layers": len(layer_types),
        "mamba_layers": mamba_layers,
        "attn_layers": attn_layers,
        "hybrid_layers": hybrid_layers,
        "metadata_keys": all_keys,
    }


def main():
    parser = argparse.ArgumentParser(description="Dump metadata and tensor names from GGUF files")
    parser.add_argument(
        "gguf_files",
        nargs="+",
        help="Path(s) to GGUF file(s) to inspect",
    )
    args = parser.parse_args()

    results = {}
    for gguf_path in args.gguf_files:
        label = Path(gguf_path).stem
        results[label] = dump_gguf(gguf_path)

    if len(results) > 1:
        print(f"\n\n{'=' * 100}")
        print("  CROSS-MODEL COMPARISON")
        print(f"{'=' * 100}\n")

        for label, info in results.items():
            if info:
                print(f"  {label}:")
                print(f"    Total layers: {info['total_layers']}")
                print(f"    Mamba layers: {info['mamba_layers']}")
                print(f"    Attention layers: {info['attn_layers']}")
                print(f"    Hybrid layers: {info['hybrid_layers']}")
                print()


if __name__ == "__main__":
    main()
