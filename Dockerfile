# Training / inference image (docs/Architecture.md §6).
# Multi-stage: wheels are resolved once in `builder`, the runtime layer stays lean and
# runs as a non-root user.
#
# Defaults to CPU-only torch (~2 GB smaller). For a GPU image, build with a CUDA wheel
# index matching the host driver and run with the NVIDIA container toolkit:
#   docker compose build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
#   docker compose --profile gpu run --rm vulnloc-gpu
ARG PYTHON_VERSION=3.11
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

# --- builder -----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# git is needed at build time: utils/config.py stamps runs with the commit SHA,
# and setuptools reads the tree while building the editable install.
RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src

ARG TORCH_INDEX_URL
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --extra-index-url "${TORCH_INDEX_URL}" .

# --- runtime -----------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_TRACKING_URI=sqlite:////workspace/artifacts/mlflow.db

RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 vulnloc

COPY --from=builder /opt/venv /opt/venv

WORKDIR /workspace
COPY --chown=vulnloc:vulnloc pyproject.toml README.md Makefile ./
COPY --chown=vulnloc:vulnloc configs ./configs
COPY --chown=vulnloc:vulnloc scripts ./scripts
COPY --chown=vulnloc:vulnloc src ./src
COPY --chown=vulnloc:vulnloc tests ./tests
COPY --chown=vulnloc:vulnloc docs ./docs

RUN mkdir -p /workspace/artifacts && chown -R vulnloc:vulnloc /workspace/artifacts
USER vulnloc

# Phase 0 exit criterion: the image runs the same command as the host.
CMD ["python", "scripts/train.py", "train=debug"]
