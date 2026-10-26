#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker build --platform linux/arm64 -t vllm-cpu-api:0.27.0 -f Dockerfile .
docker build --platform linux/arm64 -t vllm-model-init:0.27.0 -f downloader/Dockerfile .
