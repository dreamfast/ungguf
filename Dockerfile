# syntax=docker/dockerfile:1.7
FROM nvcr.io/nvidia/pytorch:26.02-py3

ARG HOST_UID=1000
ARG HOST_GID=1000

WORKDIR /app

ENV PIP_CACHE_DIR=/root/.cache/pip

# Core conversion dependencies (always installed)
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip3 install -r requirements.txt

# Optional inference dependencies (for sanity checks / model loading)
ARG INSTALL_INFERENCE=false
COPY requirements-inference.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_INFERENCE" = "true" ]; then \
        pip3 install -r requirements-inference.txt; \
    fi

# Optional dev dependencies (pytest, ruff, mypy)
ARG INSTALL_DEV=false
COPY requirements-dev.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$INSTALL_DEV" = "true" ]; then \
        pip3 install -r requirements-dev.txt; \
    fi

# Create non-root user matching host UID/GID
RUN groupadd -g "$HOST_GID" appuser 2>/dev/null || true && \
    useradd -m -u "$HOST_UID" -g "$HOST_GID" -s /bin/bash appuser 2>/dev/null || true

COPY --chown=${HOST_UID}:${HOST_GID} pyproject.toml .
COPY --chown=${HOST_UID}:${HOST_GID} src/ ./src/
COPY --chown=${HOST_UID}:${HOST_GID} tests/ ./tests/
COPY --chown=${HOST_UID}:${HOST_GID} gguf_metadata_dump.py .

ENV PYTHONPATH=/app
WORKDIR /app/src

USER ${HOST_UID}:${HOST_GID}

CMD ["python3", "-c", "print('ungguf – use ungguf.sh for commands')"]
