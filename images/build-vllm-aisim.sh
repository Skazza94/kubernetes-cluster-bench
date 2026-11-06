#!/bin/sh
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
VLLM_SOURCE="$REPO_ROOT/vllm"
CPU_IMAGE=$(awk '$1 == "FROM" { print $2; exit }' "$VLLM_SOURCE/docker/Dockerfile.aisim")

docker build --platform linux/amd64 \
    --target vllm-openai \
    --build-arg VLLM_CPU_X86=true \
    --build-arg ENABLE_AMX_FP8=0 \
    --build-arg max_jobs=4 \
    -f "$VLLM_SOURCE/docker/Dockerfile.cpu" \
    -t "$CPU_IMAGE" "$VLLM_SOURCE"

docker build --platform linux/amd64 \
    -f "$VLLM_SOURCE/docker/Dockerfile.aisim" \
    -t "${1:-kubernetes-cluster-bench:vllm-aisim}" "$VLLM_SOURCE"
