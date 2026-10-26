"""Atomic, revision-pinned HF snapshot download with a network-free cache-hit path."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def valid_snapshot(target, repo, revision):
    try:
        manifest = json.loads((target / ".ready.json").read_text())
        if manifest["repo"] != repo or manifest["revision"] != revision:
            return False
        entries = manifest["files"]
        if not entries or "config.json" not in entries or "tokenizer_config.json" not in entries:
            return False
        if not any(name.endswith(".safetensors") for name in entries):
            return False
        for name, expected in entries.items():
            path = target / name
            if not path.is_relative_to(target) or ".." in Path(name).parts:
                return False
            if path.stat().st_size != expected["size"] or digest(path) != expected["sha256"]:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def download(root, repo, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("MODEL_REVISION must be an immutable commit SHA")
    root.mkdir(parents=True, exist_ok=True)
    target = root / revision
    with (root / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if valid_snapshot(target, repo, revision):
            print('{"event":"model_cache_hit"}', flush=True)
            return target
        # Deliberately import HF only after the entire local-cache check.
        from huggingface_hub import snapshot_download
        staging = root / (".partial-" + revision)
        if staging.exists():
            shutil.rmtree(staging)
        snapshot_download(
            repo_id=repo, revision=revision, local_dir=staging,
            token=os.environ.get("HF_TOKEN") or False,
            allow_patterns=[
                "config.json", "generation_config.json", "tokenizer*", "vocab.json",
                "merges.txt", "special_tokens_map.json", "chat_template.jinja",
                "*.safetensors", "*.safetensors.index.json",
            ],
            max_workers=2,
        )
        shutil.rmtree(staging / ".cache", ignore_errors=True)
        files = {
            str(p.relative_to(staging)): {"size": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(staging.rglob("*")) if p.is_file()
        }
        (staging / ".ready.json").write_text(json.dumps({
            "repo": repo, "revision": revision, "files": files,
        }, sort_keys=True))
        if not valid_snapshot(staging, repo, revision):
            raise RuntimeError("Incomplete snapshot")
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
        print('{"event":"model_download_complete"}', flush=True)
        return target


if __name__ == "__main__":
    os.umask(0o027)
    try:
        download(Path(os.environ.get("MODEL_ROOT", "/models")),
                 os.environ["MODEL_ID"], os.environ["MODEL_REVISION"])
    except Exception:
        # Avoid HTTP exception details, which could include authentication data.
        print('{"event":"model_download_failed"}', file=sys.stderr, flush=True)
        sys.exit(1)
