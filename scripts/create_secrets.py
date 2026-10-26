#!/usr/bin/env python3
"""Create local secret files without printing secrets or putting them in argv."""
import os
from pathlib import Path
import secrets

root = Path(__file__).resolve().parents[1] / ".secrets"
os.umask(0o077)
root.mkdir(mode=0o700, exist_ok=True)
for name, value in (
    ("api-token", os.environ.get("API_TOKEN") or secrets.token_urlsafe(32)),
    ("hf-token", os.environ.get("HF_TOKEN", "")),
):
    path = root / name
    if not path.exists():
        path.write_text(value)
        path.chmod(0o600)
print("Secret files are ready in .secrets (values are not printed).")
