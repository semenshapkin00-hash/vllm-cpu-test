#!/usr/bin/env python3
"""Restart the deployment with HF offline, require a cache hit, then restore env."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="vllm-cpu")
    parser.add_argument("--output", type=Path, default=Path("results/offline-restart.json"))
    args = parser.parse_args()
    k = ["kubectl", "--context", args.context, "-n", "vllm-cpu"]

    def run(*cmd):
        return subprocess.check_output(k + list(cmd), text=True)

    def patch_env(env):
        patch = {"spec": {"template": {"spec": {"initContainers": [
            {"name": "download-model", "env": env}
        ]}}}}
        run("patch", "deployment", "vllm", "--type=strategic", "-p", json.dumps(patch))

    before = run("get", "pods", "-l", "app=vllm-cpu", "-o", "jsonpath={.items[0].metadata.uid}")
    try:
        patch_env([
            {"name": "HF_HUB_OFFLINE", "value": "1"},
            {"name": "HF_ENDPOINT", "value": "http://127.0.0.1:9"},
        ])
        run("rollout", "status", "deployment/vllm", "--timeout=10m")
        after = run("get", "pods", "-l", "app=vllm-cpu", "-o", "jsonpath={.items[0].metadata.uid}")
        logs = run("logs", "deployment/vllm", "-c", "download-model")
        assert before != after and '"event":"model_cache_hit"' in logs
        result = {
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "pod_recreated": True, "hf_hub_offline": True,
            "hf_endpoint": "http://127.0.0.1:9", "init_result": "model_cache_hit",
            "rollout_ready": True,
        }
    finally:
        patch_env([
            {"name": "HF_HUB_OFFLINE", "$patch": "delete"},
            {"name": "HF_ENDPOINT", "$patch": "delete"},
        ])
        run("rollout", "status", "deployment/vllm", "--timeout=10m")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: a new pod started from the PVC with HF offline; original environment restored")


if __name__ == "__main__":
    main()
