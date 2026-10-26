# syntax=docker/dockerfile:1
# The lock and CPU wheel below target Linux ARM64 (native on Apple Silicon VMs).
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt /tmp/requirements.txt
RUN test "$(uname -m)" = aarch64 \
    && python -m venv --without-pip /opt/venv \
    && python -m pip --python /opt/venv install --no-deps --require-hashes \
       --only-binary=:all: -r /tmp/requirements.txt
RUN python -m pip --python /opt/venv check

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
RUN apt-get update \
    && apt-get install -y --no-install-recommends libnuma1 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home app
COPY --from=dependencies /opt/venv /opt/venv
COPY --chown=10001:10001 app /app/app
ENV PATH="/opt/venv/bin:$PATH" PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp/cache \
    VLLM_CONFIG_ROOT=/tmp/vllm-config VLLM_CACHE_ROOT=/tmp/vllm-cache \
    VLLM_LOGGING_CONFIG_PATH=/app/app/logging.json VLLM_TARGET_DEVICE=cpu \
    VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 CUDA_VISIBLE_DEVICES=""
WORKDIR /app
USER 10001:10001
EXPOSE 8000
ENTRYPOINT ["python", "-m", "app.launch"]
