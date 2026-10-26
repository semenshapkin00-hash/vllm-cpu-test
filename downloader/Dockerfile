# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c AS dependencies
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
COPY downloader/requirements.txt /tmp/requirements.txt
RUN python -m venv --without-pip /opt/venv \
    && python -m pip --python /opt/venv install --no-deps --require-hashes \
       --only-binary=:all: -r /tmp/requirements.txt
RUN python -m pip --python /opt/venv check

FROM python:3.12.12-slim-bookworm@sha256:593bd06efe90efa80dc4eee3948be7c0fde4134606dd40d8dd8dbcade98e669c
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home app
COPY --from=dependencies /opt/venv /opt/venv
COPY --chown=10001:10001 downloader/download_model.py /app/download_model.py
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HOME=/tmp/huggingface XDG_CACHE_HOME=/tmp/cache
USER 10001:10001
ENTRYPOINT ["python", "/app/download_model.py"]
