#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Used to produce the committed locks: uv 0.12.10.
uv pip compile requirements.in --python-version 3.12 \
  --python-platform aarch64-manylinux_2_34 --torch-backend cpu \
  --index-strategy unsafe-best-match --generate-hashes --emit-index-url \
  --no-annotate --no-header --no-build -o requirements.txt
uv pip compile downloader/requirements.in --python-version 3.12 \
  --python-platform aarch64-manylinux_2_34 --generate-hashes \
  --no-annotate --no-header --no-build -o downloader/requirements.txt
