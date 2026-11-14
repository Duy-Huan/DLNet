# DLNet in a container (Podman or Docker), CPU.
#
#   podman build -t dlnet -f Containerfile .
#   podman run --rm -v ./work:/workspace:Z dlnet design --data data.csv --target y --budget 10m --out runs/a
#   podman run --rm -p 127.0.0.1:8766:8766 -v ./work:/workspace:Z dlnet ui          # web UI
#   podman run --rm -p 127.0.0.1:8765:8765 -v ./work:/workspace:Z -e DLNET_TOKEN=... \
#       dlnet serve --host 0.0.0.0 --workspace /workspace                              # MCP server
#
# Rootless Podman maps root in the container to your own user, so the files written in
# /workspace are yours. With Docker add --user "$(id -u):$(id -g)". Tested by
# scripts/test_container.py (CI job "container"). Details: doc/install.md.

FROM docker.io/library/python:3.12-slim

LABEL org.opencontainers.image.title="DLNet" \
      org.opencontainers.image.description="Turns a dataset into Top-K PyTorch baselines (CPU)" \
      org.opencontainers.image.source="https://github.com/Duy-Huan/DLNet"

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/dlnet \
    DLNET_UI_BIND=0.0.0.0

WORKDIR /opt/dlnet

# dependencies first (a code change does not download torch again); torch from the CPU index:
# the CUDA wheels are several GB and a container on a laptop has no GPU by default
COPY requirements.txt ./
RUN pip install "torch==2.15.0" --index-url "${TORCH_INDEX}" \
 && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY dlnet ./dlnet
RUN pip install --no-deps . \
 && rm -rf build ./*.egg-info \
 && mkdir -p /workspace "${HOME}" \
 && chmod 1777 /workspace "${HOME}" \
 && dlnet --version

WORKDIR /workspace
VOLUME ["/workspace"]
# 8766: dlnet ui, 8765: dlnet serve
EXPOSE 8766 8765

ENTRYPOINT ["dlnet"]
CMD ["--help"]
