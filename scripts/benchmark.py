#!/usr/bin/env python3
"""Streaming HTTP benchmark. Stores timings/counts, never prompts or generated text."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen


def request_json(base, token, path, payload):
    req = Request(base + path, json.dumps(payload).encode(), {
        "Authorization": "Bearer " + token, "Content-Type": "application/json",
    })
    with urlopen(req, timeout=300) as response:
        return json.load(response)


def prompt_ids(base, token, model, length):
    text = ("A database stores records. Explain how indexes affect read and write operations. "
            * (length // 8 + 20))
    ids = request_json(base, token, "/tokenize", {"model": model, "prompt": text})["tokens"]
    if len(ids) < length:
        raise RuntimeError("Not enough input tokens")
    return ids[:length]


def sample(base, token, model, prompt, output_tokens):
    payload = {
        "model": model, "prompt": prompt, "max_tokens": output_tokens,
        "temperature": 0, "seed": 42, "ignore_eos": True,
        "stream": True, "stream_options": {"include_usage": True},
    }
    req = Request(base + "/v1/completions", json.dumps(payload).encode(), {
        "Authorization": "Bearer " + token, "Content-Type": "application/json",
        "Accept": "text/event-stream",
    })
    start = time.perf_counter()
    first = None
    usage = None
    completed = False
    with urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            raw = line[6:].strip()
            if raw == b"[DONE]":
                completed = True
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError("Streaming endpoint returned an error")
            if first is None and any(c.get("text", "") for c in event.get("choices", [])):
                first = time.perf_counter()
            if event.get("usage"):
                usage = event["usage"]
    end = time.perf_counter()
    if not completed or first is None or usage is None:
        raise RuntimeError("Incomplete stream or missing token usage")
    if usage["prompt_tokens"] != len(prompt) or usage["completion_tokens"] != output_tokens:
        raise RuntimeError("Unexpected token counts")
    return {
        "ttft_ms": (first - start) * 1000,
        "latency_ms": (end - start) * 1000,
        "prompt_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
    }


def percentile(values, q):
    values = sorted(values)
    position = (len(values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def markdown(results):
    lines = [
        "| Вход токенов | Параллельность | Запросов | TTFT mean мс | TTFT p95 мс | Latency mean мс | Latency p95 мс | Запросов/с | Выходных токенов/с |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['input_tokens']} | {r['concurrency']} | {r['requests']} | "
            f"{r['ttft_mean_ms']:.2f} | {r['ttft_p95_ms']:.2f} | "
            f"{r['latency_mean_ms']:.2f} | {r['latency_p95_ms']:.2f} | "
            f"{r['requests_per_second']:.3f} | {r['output_tokens_per_second']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", type=Path, default=Path(".secrets/api-token"))
    parser.add_argument("--model", default="qwen-cpu")
    parser.add_argument("--input-tokens", type=int, nargs="+", default=[32, 1024])
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    args = parser.parse_args()
    if min(args.requests, args.output_tokens, *args.input_tokens, *args.concurrency) < 1:
        parser.error("Counts must be positive")
    if args.requests < max(args.concurrency) or args.warmups < 0:
        parser.error("Requests must be >= concurrency; warmups must be >= 0")
    token = args.token_file.read_text().strip()
    results = []
    for length in args.input_tokens:
        prompt = prompt_ids(args.base_url, token, args.model, length)
        for concurrency in args.concurrency:
            for _ in range(args.warmups):
                sample(args.base_url, token, args.model, prompt, args.output_tokens)
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                runs = list(pool.map(
                    lambda _: sample(args.base_url, token, args.model, prompt, args.output_tokens),
                    range(args.requests),
                ))
            elapsed = time.perf_counter() - started
            ttft = [r["ttft_ms"] for r in runs]
            latency = [r["latency_ms"] for r in runs]
            result = {
                "input_tokens": length, "concurrency": concurrency,
                "requests": args.requests, "errors": 0, "wall_seconds": elapsed,
                "ttft_mean_ms": statistics.mean(ttft), "ttft_p50_ms": percentile(ttft, .5),
                "ttft_p95_ms": percentile(ttft, .95),
                "latency_mean_ms": statistics.mean(latency), "latency_p50_ms": percentile(latency, .5),
                "latency_p95_ms": percentile(latency, .95),
                "requests_per_second": len(runs) / elapsed,
                "output_tokens_per_second": sum(r["output_tokens"] for r in runs) / elapsed,
                "samples": runs,
            }
            results.append(result)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": args.model, "output_tokens_per_request": args.output_tokens,
                "warmups_per_scenario": args.warmups, "results": results,
            }, indent=2) + "\n")
            args.output.with_suffix(".md").write_text(markdown(results))
            print(f"input={length} concurrency={concurrency} completed={len(runs)}", flush=True)
    print(markdown(results))


if __name__ == "__main__":
    main()
