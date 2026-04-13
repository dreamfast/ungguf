"""
Standalone SHA256 verification tool for safetensors models.

Computes a deterministic SHA256 fingerprint over the language-model tensors
of a safetensors model directory.  Used to verify that two model directories
contain byte-identical language-model weights.

TWO IMPLEMENTATIONS:
  1. load_all  -- loads the entire model into RAM then hashes.
                  Fast but needs ~50 GB for 27B.
  2. streaming -- uses safetensors.safe_open to load one tensor at a time.
                  Uses only a few hundred MB.  Required for 27B.

CONVENTIONS:
  - BF16 normalization: All tensors are cast to torch.bfloat16 before hashing
    so that models stored as F32 and BF16 produce the same hash for the same
    values.
  - Visual/MTP exclusion: Keys containing "visual" or "mtp" are excluded.
  - Canonical key mapping: Triple-prefix naming
    `model.language_model.language_model.language_model.*` is normalised to
    `model.language_model.*` before hashing.
  - Only keys whose canonical form contains "language_model" are included.

Usage:
    python3 verify_sha256.py /models/a /models/b              # load-all mode
    python3 verify_sha256.py /models/a /models/b --streaming   # streaming mode
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

_LOG_INTERVAL = 200
_PAIR_COUNT = 2

TRIPLE_PREFIX = "model.language_model.language_model.language_model."


def canonical(name: str) -> str:
    """Normalise DavidAU's triple-prefix naming to standard form."""
    if name.startswith(TRIPLE_PREFIX):
        return "model.language_model." + name[len(TRIPLE_PREFIX) :]
    return name


def is_lm_key(raw_key: str, canonical_key: str) -> bool:
    """Return True if this tensor belongs to the language model (not visual/mtp)."""
    return "language_model" in canonical_key and "visual" not in raw_key and "mtp" not in raw_key


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Normalise tensor to BF16 and return raw bytes for hashing."""
    return t.to(torch.bfloat16).contiguous().view(torch.uint8).numpy().tobytes()


# ---------------------------------------------------------------------------
# Method 1: Load-all (fast, high memory)
# ---------------------------------------------------------------------------
def sha256_load_all(model_dir: str) -> tuple[str, int]:
    """Load entire model into RAM, filter to LM tensors, hash."""
    weights: dict[str, torch.Tensor] = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        weights.update(load_file(f))

    lm: dict[str, torch.Tensor] = {}
    for k, v in weights.items():
        ck = canonical(k)
        if is_lm_key(k, ck):
            lm[ck] = v

    h = hashlib.sha256()
    for k in sorted(lm):
        h.update(k.encode())
        h.update(tensor_bytes(lm[k]))
    return h.hexdigest(), len(lm)


# ---------------------------------------------------------------------------
# Method 2: Streaming (low memory, slower)
# ---------------------------------------------------------------------------
def sha256_streaming(model_dir: str) -> tuple[str, int]:
    """Stream-hash LM tensors one at a time via safe_open (memory efficient)."""
    # First pass: build canonical -> (raw_key, shard_file) map
    lm_keys: dict[str, tuple[str, str]] = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), framework="pt", device="cpu") as sf:
            for raw in sf.keys():  # noqa: SIM118 - safe_open is not a dict
                ck = canonical(raw)
                if is_lm_key(raw, ck):
                    lm_keys[ck] = (raw, str(f))

    # Second pass: hash in sorted canonical key order
    h = hashlib.sha256()
    for i, ck in enumerate(sorted(lm_keys.keys())):
        raw, fpath = lm_keys[ck]
        with safe_open(fpath, framework="pt", device="cpu") as sf:
            t = sf.get_tensor(raw)
        h.update(ck.encode())
        h.update(tensor_bytes(t))
        del t
        if (i + 1) % _LOG_INTERVAL == 0:
            gc.collect()
            print(f"    hashed {i + 1}/{len(lm_keys)}")
    return h.hexdigest(), len(lm_keys)


def main():
    parser = argparse.ArgumentParser(
        description="SHA256 fingerprint of language-model tensors in safetensors"
    )
    parser.add_argument("model_dirs", nargs="+", help="One or more model directories")
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode (low memory, required for 27B)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON results",
    )
    args = parser.parse_args()

    hash_fn = sha256_streaming if args.streaming else sha256_load_all
    mode = "streaming" if args.streaming else "load-all"

    results = []
    for d in args.model_dirs:
        print(f"Hashing {d} [{mode}]...")
        digest, count = hash_fn(d)
        print(f"  SHA256: {digest}  ({count} LM tensors)")
        results.append({"model_dir": d, "sha256": digest, "lm_tensor_count": count})

    if len(results) == _PAIR_COUNT:
        match = results[0]["sha256"] == results[1]["sha256"]
        print(f"\nMATCH: {match}")
        for r in results:
            r["match"] = match

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(results, indent=2))
        print(f"Saved: {args.json_out}")

    # Exit with 0 if match (or single model), 1 if mismatch
    if len(results) == _PAIR_COUNT and not match:
        sys.exit(1)


if __name__ == "__main__":
    main()
