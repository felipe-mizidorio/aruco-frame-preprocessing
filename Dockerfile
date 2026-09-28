# GPU image for the aruco pipeline (targets RTX 5090 / Blackwell sm_120).
# torch>=2.11 PyPI wheels ship their own CUDA 13 runtime libs, so the base image
# only needs the NVIDIA container runtime hooks; host driver must be >= 580.
FROM nvidia/cuda:13.0.1-base-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# libgl1/libglib2.0-0: runtime deps of opencv-python (non-headless wheel).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HOME=/cache/huggingface \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

COPY pyproject.toml uv.lock README.md ./
COPY configs ./configs
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/opt/venv/bin:$PATH"

# Runs non-root; compose sets the UID to the host user's. /data (sessions) and
# /cache (HF weights) are mounted volumes, writable by any UID.
# configs/ is read from /app via the editable install (config.py parents[2]).
RUN mkdir -p /data /cache/huggingface && chmod -R 1777 /data /cache
# The host UID has no passwd entry in the image; torch's inductor cache calls
# getpass.getuser(), which needs USER set or it raises OSError.
ENV HOME=/cache XDG_CACHE_HOME=/cache USER=aruco
USER 1000
WORKDIR /data

CMD ["aruco-mask", "--help"]
