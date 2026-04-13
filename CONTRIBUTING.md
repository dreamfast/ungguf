# Contributing to ungguf

Contributions are welcome! This guide covers how to add new architecture converters and the code conventions to follow.

## Adding a new architecture converter

Each architecture requires four files:

1. **Mapping module** (`src/mappings_<arch>.py`) — tensor name maps, constants, and shared helpers
2. **Converter** (`src/gguf_to_safetensors_<arch>.py`) — the main conversion logic
3. **Verifier** (`src/verify_conversion_<arch>.py`) — bit-exact verification
4. **Docker service** in `docker-compose.yml` + CLI command in `ungguf.sh`

### Step 1: Create the mapping module

Define `GLOBAL_MAP`, `LAYER_SUFFIX_MAP`, and `gguf_name_to_hf()` in a new `mappings_<arch>.py`.
See `mappings_qwen3.py` for the simplest example.

### Step 2: Create the converter

Follow the pattern in an existing converter (Qwen3 is the simplest). Key requirements:

- Use `argparse` with `if __name__ == "__main__": main()` pattern
- Import shared utilities from `common.py`
- Import mappings from your `mappings_<arch>.py`
- Run in **strict mode**: fail with `sys.exit(1)` and stderr messages on any error
- Use `write_shards()` from common.py for output

### Step 3: Create the verifier

Import the same mappings from the shared module (never duplicate them). The verifier should:

- Decode each GGUF tensor independently using the same approach as the converter
- Compare against the converted safetensors bit-for-bit
- Report matched/mismatched/missing counts
- Exit 0 on success, 1 on failure

### Step 4: Wire up the CLI

Add a `cmd_convert_<arch>()` and `cmd_verify_<arch>()` function in `ungguf.sh`, and add
a corresponding Docker Compose service in `docker-compose.yml`.

## Code style

- **Python version:** 3.10+ (use `str | None`, `dict[str, list[int]]`, etc.)
- **Imports:** `from __future__ import annotations` at the top of every file
- **Naming:** `snake_case` for functions/variables, `UPPER_CASE` for module constants
- **Tensor maps:** `GLOBAL_MAP` + `LAYER_SUFFIX_MAP` dicts at module level in mapping modules
- **Output:** Use `print()` for progress, `print(..., file=sys.stderr)` for errors
- **Error handling:** Strict mode — always `sys.exit(1)` on errors, never produce partial output
- **No `__init__.py`** files — modules are imported via `PYTHONPATH=/app` in Docker
- **Module docstrings:** Include purpose, architecture notes, and usage at the top of each file
- **Formatting:** All code is formatted with Ruff (configured in `pyproject.toml`)

## Quality gates

All quality checks run inside Docker. **Never** run Python, ruff, or mypy directly on the host.

### Linting and formatting (Ruff)

```bash
# Check for lint errors (read-only)
./ungguf.sh lint

# Auto-fix lint issues and format code (writes to files)
./ungguf.sh format
```

Or run individual checks:
```bash
docker compose run --rm lint ruff check .           # Lint only
docker compose run --rm lint ruff format --check .   # Format check only
```

The Ruff configuration in `pyproject.toml` enables these rule sets: E, W, F, I, B, C4, UP, SIM, RUF, N, PL, PERF, PTH, T20, S. Some rules have per-file exceptions (e.g., `T201` print is allowed in converters/verifiers, `S101` assert is allowed in tests).

### Type checking (mypy)

```bash
docker compose run --rm lint mypy src/ gguf_metadata_dump.py
```

All new functions should have type annotations. The mypy configuration uses gradual adoption (`check_untyped_defs = true`, `disallow_untyped_defs = false`).

### Unit tests (pytest)

```bash
./ungguf.sh test
# or
docker compose run --rm test
```

265+ tests covering tensor mapping, conversion logic, shared utilities, and sanity checks. All tests run CPU-only in Docker.

### Full quality gate (all at once)

```bash
docker compose run --rm lint   # Runs: ruff check + ruff format --check + mypy
docker compose run --rm test   # Runs: pytest
```

Before submitting a PR, verify your converter works end-to-end:

```bash
# Lint + type check
./ungguf.sh lint

# Run tests
./ungguf.sh test

# Convert
./ungguf.sh convert-<arch> model.gguf ./output ./reference_model

# Verify bit-exact
./ungguf.sh verify-<arch> model.gguf ./output ./reference_model

# Test inference (optional)
./ungguf.sh sanity ./output --label "test"
```

## Pull requests

- One architecture per PR (or one logical change per PR)
- Include the mapping module, converter, and verifier together
- Verify that `./ungguf.sh lint` passes with zero errors
- Verify that `./ungguf.sh test` passes all tests
- If adding a new architecture, update the README supported architectures table
