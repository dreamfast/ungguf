# Security Policy

## Reporting vulnerabilities

If you discover a security vulnerability in ungguf, please report it by opening a GitHub issue with the `security` label. For sensitive vulnerabilities, contact the maintainers directly.

## Known security considerations

### Binary GGUF file parsing

ungguf reads and parses binary GGUF files using the `gguf` Python library. While the library is well-maintained, parsing untrusted binary files always carries inherent risk. **Only convert GGUF files from sources you trust.**

### `trust_remote_code=True` in vLLM

The `vllm_sanity.py` inference checker uses `trust_remote_code=True` when loading models with vLLM. This is required for many HuggingFace models but allows arbitrary code execution during model loading. **Only run sanity checks against models from trusted sources.**

### Docker GPU access

All conversion and inference runs inside Docker containers with NVIDIA GPU access (`--runtime=nvidia`). Ensure your Docker installation is properly configured and that GPU access is restricted to trusted users.

### No cryptographic verification of GGUF files

ungguf does not verify the integrity or authenticity of GGUF input files. A corrupted or maliciously modified GGUF file could produce incorrect safetensors output. Always verify conversion results using the built-in bit-exact verifier.
