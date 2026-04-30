<p align="center">
  <img src="https://murmur.dreamfast.solutions/ungguf.webp" alt="ungguf logo" width="256">
</p>

# ungguf

Convert GGUF model files back to HuggingFace safetensors format.

Reverses the typical HF-to-GGUF conversion pipeline, recovering the original
safetensors layout, tensor names, and sharding convention so that converted
models can be loaded by HuggingFace-compatible inference engines such as vLLM.

> **Important limitation:** ungguf requires the **original base safetensors model** as a
> reference. It cannot convert a GGUF file on its own — it needs a HuggingFace-format model
> directory (with `config.json`, tokenizer files, and the original tensor names/shapes) to
> map GGUF tensor names back to HuggingFace convention and validate the output. If you don't
> have the original safetensors, you'll need to download them from HuggingFace first.

## Supported architectures

| Architecture | Converter | Quant types | Notes |
|---|---|---|---|
| **Qwen3** | `convert-qwen3` | BF16, F16, F32, Q8_0, etc. | Standard GQA transformer with SwiGLU MLP. 1:1 tensor mapping. |
| **Qwen3.5** | `convert-qwen35` | BF16, F16, F32, Q8_0, etc. | Hybrid Mamba2 + Transformer. Handles V-head reorder, norm-1.0 convention, A_log convention. |
| **Qwen3.6** | `convert-qwen36` | BF16, F16, F32, Q8_0, etc. | Same hybrid architecture as Qwen3.5. Handles GGUF files that exclude MTP/vision tensors (copies from reference). |
| **Qwen3.6-MoE** | `convert-qwen36-moe` | BF16, F16, F32, Q8_0, etc. | MoE hybrid (256 experts, 8 active). Expert 3D tensor reshape, GGML F-order reversal, V-head reorder. |
| **GLM-4.7-Flash** | `convert-glm47` | BF16, F16, F32, Q8_0, etc. | DeepSeek2 architecture with MLA attention + MoE. Handles kv_b_proj reconstruction, 3D expert tensor splitting. |

Quantized formats (Q8_0, Q4_K_M, etc.) are dequantized to bfloat16 during conversion. Use `--keep-fp16` to preserve float16 tensors when the source GGUF is FP16.

## Prerequisites

- **Docker** with Compose plugin (BuildKit required — the Dockerfile uses `--mount=type=cache`)
- **NVIDIA GPU** + NVIDIA Container Toolkit (`nvidia-container-toolkit`)
- **NVIDIA driver** compatible with the base image (CUDA 12.x)
- ~16 GB RAM minimum (more for larger models)

## What is a "reference model"?

Each converter requires a **reference model directory** — a HuggingFace-format safetensors
model containing `config.json`, tokenizer files, and the original tensor shapes. The converter
uses this to:

1. Map GGUF tensor names (e.g. `blk.0.attn_q.weight`) to HuggingFace names (e.g. `model.layers.0.self_attn.q_proj.weight`)
2. Validate tensor shapes against the known-good reference
3. Copy tokenizer and config files into the output directory

You can download reference models from HuggingFace. The reference doesn't need to be the same
quantization — it just needs to be the same architecture with matching tensor names and shapes.

## Quick start

### 1. Build the Docker image

```bash
./ungguf.sh build
```

### 2. Convert a GGUF model

```bash
# Qwen3.5 example
./ungguf.sh convert-qwen35 model.gguf ./output ./reference_model

# Qwen3.6 (dense) example
./ungguf.sh convert-qwen36 model.gguf ./output ./reference_model

# Qwen3 example
./ungguf.sh convert-qwen3 model.gguf ./output ./reference_model

# Qwen3.6 MoE example
./ungguf.sh convert-qwen36-moe model.gguf ./output ./reference_model

# GLM-4.7 example
./ungguf.sh convert-glm47 model.gguf ./output ./reference_model

# With --keep-fp16 (preserves FP16 tensors for FP16 GGUF sources)
./ungguf.sh convert-glm47 --keep-fp16 model.gguf ./output ./reference_model
```

### 3. Verify the conversion

Bit-exact verification that every tensor in the GGUF matches the safetensors output:

```bash
# Qwen3.5
./ungguf.sh verify model.gguf ./output ./reference_model

# Qwen3.6 (dense)
./ungguf.sh verify-qwen36 model.gguf ./output ./reference_model

# Qwen3
./ungguf.sh verify-qwen3 model.gguf ./output

# Qwen3.6 MoE
./ungguf.sh verify-qwen36-moe model.gguf ./output ./reference_model

# GLM-4.7
./ungguf.sh verify-glm47 model.gguf ./output ./reference_model
```

The verifier decodes every GGUF tensor independently and compares it against the converted
output. If every tensor matches bit-for-bit, the conversion is verified correct.

### 4. Inspect a GGUF file

```bash
./ungguf.sh inspect model.gguf
# Multiple files for cross-model comparison:
./ungguf.sh inspect model-a.gguf model-b.gguf
```

### 5. Run an inference sanity check

Requires the vLLM inference image (separate build):

```bash
./ungguf.sh build inference

# Test a safetensors directory (results saved to ./results/<label>.json)
./ungguf.sh sanity ./output --label "test"

# Test a raw GGUF file
./ungguf.sh sanity model.gguf --label "gguf_test"

# With FP8 quantization and tensor parallelism
./ungguf.sh sanity ./output --label "fp8_test" --quantize fp8 --tp 2
```

**Note:** The `sanity` command runs 10 harmful + 10 benign prompts to test refusal behavior
and coherence. Results are saved to `./results/<label>.json`. This tool is intended for testing
uncensored / abliterated models.

## CLI reference

```
ungguf <command> [args...]

Commands:
  build [convert|inference]          Build Docker images (default: convert)
  convert-qwen35 [--keep-fp16] <g> <o> <r>  Convert Qwen3.5 GGUF to safetensors
  convert-qwen36 [--keep-fp16] <g> <o> <r>  Convert Qwen3.6 (dense) GGUF to safetensors
  convert-qwen36-moe [--keep-fp16] <g> <o> <r>  Convert Qwen3.6 MoE GGUF to safetensors
  convert-glm47  [--keep-fp16] <g> <o> <r>  Convert GLM-4.7 GGUF to safetensors
  convert-qwen3  [--keep-fp16] <g> <o> <r>  Convert Qwen3 GGUF to safetensors
  verify         [--keep-fp16] <g> <c> <r>  Verify Qwen3.5 conversion (bit-exact)
  verify-qwen36  [--keep-fp16] <g> <c> <r>  Verify Qwen3.6 (dense) conversion (bit-exact)
  verify-qwen36-moe [--keep-fp16] <g> <c> <r>  Verify Qwen3.6 MoE conversion (bit-exact)
  verify-glm47   [--keep-fp16] <g> <c> <r>  Verify GLM-4.7 conversion (bit-exact)
  verify-qwen3   [--keep-fp16] <g> <c>      Verify Qwen3 conversion (bit-exact)
  inspect        <gguf> [gguf2 ...]          Dump GGUF metadata and tensor layout
  sanity         <model|gguf> [opts]         Run vLLM inference sanity check
  test           [pytest-args...]            Run unit tests inside Docker
  lint                                       Run ruff + mypy checks (read-only)
  format                                     Auto-fix lint issues and format code
  help                                   Show help message

Options (convert/verify):
  --keep-fp16    Preserve float16 tensors instead of converting to bfloat16

Options (sanity):
  --label X                Label for this test run (required)
  --quantize fp8|awq|gptq  On-the-fly quantization
  --tp N                   Tensor parallel size
  --tokenizer /path        External tokenizer path (for GGUF without chat template)
  --max-model-len N        Max context length (default: 4096)

Environment variables:
  GPU             GPU device(s) — default: 'all' for sanity, '0' for convert/verify
                  Examples: GPU=0  GPU=1  GPU=0,1  GPU=all
  SHARD_SIZE_MB   Shard size in MB for convert (default: 4500)
```

## How it works

Each converter performs the following steps:

1. **Read the GGUF file** using the `gguf` Python library
2. **Load reference shapes** from a HuggingFace safetensors model to get the target tensor names and dimensions
3. **Map GGUF tensor names** to HuggingFace names using architecture-specific mapping tables
4. **Decode tensors** from GGUF format (handles BF16, F16, F32, and quantized formats)
5. **Apply architecture-specific transforms** (e.g., V-head reorder reversal for Qwen3.5, kv_b_proj reconstruction for GLM-4.7)
6. **Write sharded safetensors** in HuggingFace convention with an index JSON

The reference model is needed because GGUF uses different tensor names than HuggingFace, and the reference provides the canonical naming and shapes.

### Error handling

All converters run in **strict mode**: if any tensor cannot be mapped, decoded, or has a shape
mismatch, the conversion fails with a clear error message to stderr. Partial/invalid output is
never written. The reference model's `config.json` is required for architectures that need
architecture-specific transforms (e.g., Qwen3.5 V-head reorder).

## Development

### Quality gates

All Python runs inside Docker — never on the host. The project uses Ruff (lint + format) and
mypy (type checking), configured in `pyproject.toml`.

```bash
# Run all linters (ruff check + ruff format --check + mypy)
./ungguf.sh lint

# Auto-fix lint issues and format code
./ungguf.sh format

# Run unit tests (265+ tests, CPU-only)
./ungguf.sh test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Project structure

```
ungguf.sh                      CLI entry point
src/
  common.py                    Shared utilities (tensor decode, sharding, file copy)
  mappings_qwen3.py            Qwen3 tensor name mappings
  mappings_qwen35.py           Qwen3.5 tensor name mappings + V-head reorder
  mappings_qwen36_moe.py       Qwen3.6 MoE tensor name mappings + expert builders
  mappings_glm47.py            GLM-4.7 tensor name mappings + kv_b reconstruction
  gguf_to_safetensors_qwen3.py    Qwen3 converter
  gguf_to_safetensors_qwen35.py   Qwen3.5 converter
  gguf_to_safetensors_qwen36.py   Qwen3.6 (dense) converter
  gguf_to_safetensors_qwen36_moe.py  Qwen3.6 MoE converter
  gguf_to_safetensors_glm47.py    GLM-4.7 converter
  verify_conversion_qwen3.py      Qwen3 bit-exact verifier
  verify_conversion_qwen35.py     Qwen3.5 bit-exact verifier
  verify_conversion_qwen36.py     Qwen3.6 (dense) bit-exact verifier
  verify_conversion_qwen36_moe.py Qwen3.6 MoE bit-exact verifier
  verify_conversion_glm47.py      GLM-4.7 bit-exact verifier
  verify_sha256.py                 SHA256 fingerprint comparator
  vllm_sanity.py                   vLLM inference sanity checker
gguf_metadata_dump.py          GGUF forensic inspection tool
Dockerfile                     Conversion image (NVIDIA PyTorch base)
Dockerfile.inference           Inference image (adds vLLM)
docker-compose.yml             Service definitions
pyproject.toml                 Ruff, mypy, and pytest configuration
requirements.txt               Python dependencies (conversion)
requirements-dev.txt           Dev dependencies (ruff, mypy, pytest)
requirements-inference.txt     Python dependencies (inference)
```

## Troubleshooting

**"ERROR: N unmatched GGUF tensors"** — The GGUF file contains tensors that don't match any
mapping rule. This usually means the GGUF architecture doesn't match the selected converter.

**"ERROR: N reference tensors not produced"** — Some tensors from the reference model weren't
found in the GGUF file. The GGUF may be from a different model variant.

**"ERROR: No config.json in reference model"** — The reference model directory must contain a
`config.json` for architecture detection. Download a complete HuggingFace model directory.

**Docker build fails with "mount type cache not supported"** — Your Docker installation needs
BuildKit. Enable it with `export DOCKER_BUILDKIT=1` or upgrade Docker.

## License

MIT
