#!/usr/bin/env python3
"""Record non-secret hardware and deployed configuration used for the benchmark."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="vllm-cpu")
    parser.add_argument("--output", type=Path, default=Path("results/environment.json"))
    args = parser.parse_args()

    def read(*cmd):
        return subprocess.check_output(cmd, text=True).strip()

    k = ["kubectl", "--context", args.context, "-n", "vllm-cpu"]
    deployment = json.loads(read(*k, "get", "deployment", "vllm", "-o", "json"))
    spec = deployment["spec"]["template"]["spec"]
    config = json.loads(read(*k, "get", "configmap", "vllm-config", "-o", "json"))["data"]
    nodes = json.loads(read("kubectl", "--context", args.context, "get", "nodes", "-o", "json"))
    docker = json.loads(read("docker", "info", "--format", "{{json .}}"))
    minikube_limits = json.loads(read(
        "docker", "inspect", args.context, "--format",
        '{"cpu_quota_nanocpus":{{.HostConfig.NanoCpus}},"memory_limit_bytes":{{.HostConfig.Memory}}}',
    ))
    runtime = json.loads(read(*k, "exec", "deployment/vllm", "-c", "api", "--", "python", "-c",
        "import os,json,platform,torch,importlib.metadata; from vllm.platforms import current_platform; "
        "print(json.dumps(dict(uid=os.getuid(),python=platform.python_version(),"
        "vllm=importlib.metadata.version('vllm'),torch=importlib.metadata.version('torch'),"
        "cuda=torch.version.cuda,cpu=current_platform.is_cpu())))"
    ).splitlines()[-1])
    cpu = read("sysctl", "-n", "machdep.cpu.brand_string") if platform.system() == "Darwin" else platform.processor()
    host_ram = int(read("sysctl", "-n", "hw.memsize")) if platform.system() == "Darwin" else None
    result = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"os": platform.system(), "architecture": platform.machine(), "cpu": cpu,
                 "ram_bytes": host_ram},
        "docker": {key: docker.get(key) for key in ["ServerVersion", "Architecture", "NCPU", "MemTotal"]},
        "minikube_container_limits": minikube_limits,
        "kubernetes": nodes["items"][0]["status"]["nodeInfo"],
        "node_allocatable": nodes["items"][0]["status"]["allocatable"],
        "runtime": runtime, "model_id": config["MODEL_ID"], "model_revision": config["MODEL_REVISION"],
        "server_config": json.loads(config["server.json"]),
        "cpu_env": {key: config[key] for key in ["VLLM_CPU_KVCACHE_SPACE", "VLLM_CPU_OMP_THREADS_BIND", "OMP_NUM_THREADS"]},
        "api_resources": spec["containers"][0]["resources"],
        "images": {c["name"]: c["image"] for c in spec["containers"] + spec["initContainers"]},
        "benchmark_transport": "localhost HTTP through kubectl port-forward",
    }
    # Omit node boot/machine/system identifiers.
    for key in ["machineID", "systemUUID", "bootID"]:
        result["kubernetes"].pop(key, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("Saved non-secret benchmark environment metadata")


if __name__ == "__main__":
    main()
