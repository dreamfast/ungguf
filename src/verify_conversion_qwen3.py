"""Full verification: compare every GGUF tensor against converted safetensors output.

Decodes each GGUF tensor using the same approach as the converter (reverse shape,
C-order reshape), then checks bit-exact equality against the converted output.

Handles --keep-fp16 mode: if converted is fp16, compare in fp16.
If converted is bf16, cast GGUF fp16 to bf16 before comparing.
"""

from __future__ import annotations

import argparse
import sys

import torch
from gguf import GGUFReader

from common import decode_gguf_tensor, load_converted_tensors
from mappings_qwen3 import gguf_name_to_hf


def main():
    parser = argparse.ArgumentParser(
        description="Verify Qwen3 GGUF-to-safetensors conversion is bit-exact"
    )
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--converted", required=True)
    parser.add_argument(
        "--keep-fp16",
        action="store_true",
        help="Expect float16 tensors to be preserved (not converted to bfloat16)",
    )
    args = parser.parse_args()

    reader = GGUFReader(args.gguf)
    conv = load_converted_tensors(args.converted)

    print(f"GGUF: {len(reader.tensors)} tensors")
    print(f"Converted: {len(conv)} tensors")

    exact = 0
    mismatch = 0
    unmapped = 0
    missing = 0

    mismatched_list = []

    for gt in reader.tensors:
        hf_name = gguf_name_to_hf(gt.name)
        if hf_name is None:
            unmapped += 1
            continue
        if hf_name not in conv:
            missing += 1
            print(f"  MISSING: {gt.name} -> {hf_name}")
            continue

        conv_t = conv[hf_name]
        decoded = decode_gguf_tensor(
            gt, keep_f32=True, keep_f16=args.keep_fp16, reverse_shape=True
        )
        if decoded.dtype != conv_t.dtype:
            decoded = decoded.to(conv_t.dtype)

        if list(decoded.shape) != list(conv_t.shape):
            mismatch += 1
            mismatched_list.append(
                (gt.name, hf_name, f"shape {list(decoded.shape)} vs {list(conv_t.shape)}")
            )
            continue

        if torch.equal(decoded, conv_t):
            exact += 1
        else:
            diff = (decoded.float() - conv_t.float()).abs()
            mismatch += 1
            mismatched_list.append(
                (
                    gt.name,
                    hf_name,
                    f"max_diff={diff.max().item():.6e} mean={diff.mean().item():.6e}",
                )
            )

    print(f"\n{'=' * 50}")
    print(f"BIT-EXACT:  {exact}/{len(reader.tensors)}")
    print(f"MISMATCH:   {mismatch}")
    print(f"UNMAPPED:   {unmapped}")
    print(f"MISSING:    {missing}")
    print(f"{'=' * 50}")

    if mismatched_list:
        print("\nMISMATCHED:")
        for gn, hn, reason in mismatched_list:
            print(f"  {gn} -> {hn}: {reason}")

    if mismatch == 0 and missing == 0:
        print(f"\nALL {exact} TENSORS VERIFIED BIT-EXACT")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
