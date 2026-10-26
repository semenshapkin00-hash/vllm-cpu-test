#!/usr/bin/env python3
"""Check the real Kubernetes endpoint, authentication, UID and log privacy."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="vllm-cpu")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", type=Path, default=Path(".secrets/api-token"))
    parser.add_argument("--output", type=Path, default=Path("results/verification.json"))
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    k = ["kubectl", "--context", args.context, "-n", "vllm-cpu"]

    def kubectl(*cmd):
        return subprocess.check_output(k + list(cmd), text=True)

    def call(key, payload, path="/v1/chat/completions"):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = "Bearer " + key
        req = Request(args.base_url + path, json.dumps(payload).encode(), headers)
        try:
            with urlopen(req, timeout=300) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, None

    def audit_records(logs):
        items = []
        for line in logs.splitlines():
            # vLLM workers can prefix stdout with a process label.
            start = line.find('{"time":')
            if start < 0:
                continue
            try:
                item = json.loads(line[start:])
            except ValueError:
                continue
            if item.get("event") == "request_end":
                items.append(item)
        return items

    initial_ids = {r["request_id"] for r in audit_records(kubectl("logs", "deployment/vllm", "-c", "api"))}

    marker = "PRIVATE_" + uuid.uuid4().hex
    payload = {"model": "qwen-cpu", "messages": [{"role": "user", "content": "Repeat exactly " + marker}],
               "max_tokens": 24, "temperature": 0}
    missing, _ = call(None, payload)
    invalid, _ = call("invalid-test-token", payload)
    valid, body = call(token, payload)
    bad_payload = dict(payload, max_tokens="INVALID_" + marker)
    bad_status, _ = call(token, bad_payload)
    assert (missing, invalid, valid) == (401, 401, 200), "Bearer authentication failed"
    assert bad_status in (400, 422), "Invalid input was not rejected"
    answer = body["choices"][0]["message"]["content"]
    assert answer and len(answer) > 8, "No generated answer"

    stream_marker = "PRIVATE_STREAM_" + uuid.uuid4().hex
    streaming_payload = dict(payload, stream=True, messages=[{
        "role": "user", "content": "Repeat exactly " + stream_marker,
    }])
    req = Request(args.base_url + "/v1/chat/completions", json.dumps(streaming_payload).encode(), {
        "Content-Type": "application/json", "Authorization": "Bearer " + token,
    })
    fragments = []
    done = False
    with urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                done = True
                break
            item = json.loads(raw)
            assert "error" not in item, "Streaming inference failed"
            for choice in item.get("choices", []):
                fragments.append(choice.get("delta", {}).get("content") or "")
    streamed_answer = "".join(fragments)
    assert done and len(streamed_answer) > 8, "Incomplete streaming answer"

    tokenize_payload = {"model": "qwen-cpu", "prompt": marker}
    tokenize_missing, _ = call(None, tokenize_payload, "/tokenize")
    tokenize_valid, _ = call(token, tokenize_payload, "/tokenize")
    assert (tokenize_missing, tokenize_valid) == (401, 200), "Tokenizer authentication failed"
    unknown_path = "/" + marker + "?prompt=" + marker
    unknown_missing, _ = call(None, {}, unknown_path)
    unknown_valid, _ = call(token, {}, unknown_path)
    assert (unknown_missing, unknown_valid) == (401, 404), "Unexpected unknown-route behavior"

    for _ in range(20):
        logs = kubectl("logs", "deployment/vllm", "-c", "api")
        audit = [r for r in audit_records(logs) if r["request_id"] not in initial_ids]
        if len(audit) >= 9:
            break
        time.sleep(0.1)
    for private in [marker, stream_marker, answer, streamed_answer, token]:
        assert private not in logs, "Private data leaked to logs"
    assert len(audit) >= 9, "New requests were not audited"
    assert sum(r["status"] == 401 for r in audit) >= 4
    assert sum(r["status"] == 200 for r in audit) >= 3
    assert any(r["status"] == 404 and r["route"] == "other" for r in audit)
    assert all(r["route"] in {"/v1/chat/completions", "/tokenize", "other"} for r in audit)
    runtime = json.loads(kubectl("exec", "deployment/vllm", "-c", "api", "--", "python", "-c",
        "import os,json,torch,importlib.metadata; from vllm.platforms import current_platform; "
        "print(json.dumps(dict(uid=os.getuid(),vllm=importlib.metadata.version('vllm'),"
        "torch=importlib.metadata.version('torch'),cuda=torch.version.cuda,cpu=current_platform.is_cpu())))"
    ).strip().splitlines()[-1])
    assert runtime["uid"] == 10001 and runtime["cuda"] is None and runtime["cpu"]
    init_logs = kubectl("logs", "deployment/vllm", "-c", "download-model")
    assert '"event":"model_' in init_logs
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "runtime": runtime,
        "missing_token_status": missing, "invalid_token_status": invalid,
        "valid_token_status": valid, "invalid_payload_status": bad_status,
        "prompt_answer_and_token_absent_from_logs": True,
        "streaming_completed": done,
        "tokenize_missing_token_status": tokenize_missing,
        "tokenize_valid_status": tokenize_valid,
        "unknown_path_and_query_absent_from_logs": True,
        "new_audit_records_found": len(audit), "audit_example": audit[-1],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("PASS: CPU, non-root, Bearer authentication, request audit and log privacy")


if __name__ == "__main__":
    main()
