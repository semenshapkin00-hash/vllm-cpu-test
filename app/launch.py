"""Read non-secret parameters from ConfigMap and enforce the CPU/privacy settings."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def arguments(config):
    allowed = {
        "model", "served_model_name", "dtype", "max_model_len", "max_num_seqs",
        "max_num_batched_tokens", "enable_chunked_prefill", "enable_prefix_caching",
    }
    if set(config) - allowed:
        raise ValueError("Unsupported server configuration key")
    model = Path(config["model"])
    if not model.is_absolute() or not (model / ".ready.json").is_file():
        raise ValueError("A complete local model snapshot is required")
    if not os.environ.get("VLLM_API_KEY"):
        raise ValueError("VLLM_API_KEY must come from a Kubernetes Secret")
    args = [
        "--host", "0.0.0.0", "--port", "8000",
        "--no-enable-log-requests", "--no-enable-log-outputs",
        "--disable-uvicorn-access-log",
        "--log-config-file", "/app/app/logging.json",
        "--middleware", "app.privacy.RequestAuditMiddleware",
        "--enforce-eager", "--generation-config", "vllm",
    ]
    for key, value in config.items():
        option = key.replace("_", "-")
        if isinstance(value, bool):
            args.append("--" + ("" if value else "no-") + option)
        else:
            args.extend(["--" + option, str(value)])
    return args


def main():
    import torch
    from vllm.platforms import current_platform
    if torch.version.cuda is not None or not current_platform.is_cpu():
        raise RuntimeError("Only the CPU build is permitted")
    if importlib.metadata.version("vllm").split("+")[0] != "0.27.0":
        raise RuntimeError("Unexpected vLLM version")
    config = json.loads(Path("/config/server.json").read_text())
    args = arguments(config)
    os.execv(sys.executable, [sys.executable, "-m", "vllm.entrypoints.openai.api_server", *args])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Configuration and import errors must not dump tokens/configuration.
        print('{"event":"startup_failed"}', file=sys.stderr)
        sys.exit(1)
