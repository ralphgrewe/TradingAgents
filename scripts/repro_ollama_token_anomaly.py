"""Controlled reproduction for issue #148's two input_tokens anomalies.

Sends prompts of known, increasing size to a live local Ollama server via the
OpenAI-compatible endpoint (the same wire format `tradingagents/llm_clients/
openai_client.py`'s `ollama` provider uses) and records what the server's
`usage.prompt_tokens` (== `usage_metadata.input_tokens` once through
LangChain) actually reports, compared to a real tiktoken count of what was
sent. See docs/analysis/prompt-truncation-diagnosis.md for the write-up this
script's output feeds.

This is a standalone diagnostic script (not part of the application) --
it talks to Ollama directly over HTTP so results aren't affected by anything
in this codebase's own instrumentation. It requires a running local Ollama
server (`ollama serve`) with the target model pulled.

Usage:
    ./venv/bin/python scripts/repro_ollama_token_anomaly.py [--model MODEL] [--fast]

--fast runs a smaller/quicker subset (useful on a loaded/shared GPU).
"""
from __future__ import annotations

import argparse
import json
import random
import string

import requests
import tiktoken

BASE_URL = "http://localhost:11434/v1"
ENC = tiktoken.get_encoding("o200k_base")

WORDS = [
    "market", "signal", "volatility", "trend", "reversal", "momentum",
    "earnings", "guidance", "liquidity", "spread", "volume", "sector",
    "macro", "yield", "curve", "inflation", "catalyst", "resistance",
    "support", "breakout", "drawdown", "position", "hedge", "risk",
]


def make_filler(n_tokens: int) -> str:
    """Build filler text whose tiktoken (o200k_base) count is close to n_tokens."""
    approx_tokens_per_word = len(ENC.encode(" ".join(WORDS))) / len(WORDS)
    approx_word_count = max(1, int(n_tokens / approx_tokens_per_word) + 5)
    out = [WORDS[i % len(WORDS)] for i in range(approx_word_count)]
    text = " ".join(out)
    ids = ENC.encode(text)
    while len(ids) < n_tokens:
        out.append(WORDS[len(out) % len(WORDS)])
        text = " ".join(out)
        ids = ENC.encode(text)
    if len(ids) > n_tokens:
        ids = ids[:n_tokens]
    return ENC.decode(ids)


def real_token_count(text: str) -> int:
    return len(ENC.encode(text))


def call(model: str, messages: list[dict], max_tokens: int = 8, timeout: int = 150,
         options: dict | None = None) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "max_tokens": max_tokens}
    if options:
        body["options"] = options
    resp = requests.post(f"{BASE_URL}/chat/completions", json=body, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return {
        "usage": data.get("usage"),
        "content": data["choices"][0]["message"].get("content", ""),
    }


def log(row: dict) -> None:
    print(json.dumps(row), flush=True)


def experiment_scaling(model: str, sizes: list[int]) -> None:
    print("\n=== Scaling: isolated single-message prompts of increasing size ===", flush=True)
    for n in sizes:
        body = make_filler(n)
        text = f"[scale-{n}] {body}\n\nReply with a single word."
        real_tokens = real_token_count(text)
        try:
            out = call(model, [{"role": "user", "content": text}])
            log({
                "experiment": "scaling",
                "requested_filler_tokens": n,
                "real_tiktoken_count": real_tokens,
                "usage": out["usage"],
            })
        except requests.exceptions.RequestException as e:
            log({
                "experiment": "scaling",
                "requested_filler_tokens": n,
                "real_tiktoken_count": real_tokens,
                "error": str(e),
            })


def experiment_num_ctx_override(model: str, n: int) -> None:
    print(f"\n=== num_ctx override at n={n}: does an explicit large num_ctx change the reported constant? ===", flush=True)
    body = make_filler(n)
    text = f"[numctx] {body}\n\nReply with a single word."
    try:
        out_default = call(model, [{"role": "user", "content": text}])
        log({"experiment": "num_ctx_default", "requested_tokens": n, "usage": out_default["usage"]})
    except requests.exceptions.RequestException as e:
        log({"experiment": "num_ctx_default", "requested_tokens": n, "error": str(e)})
    try:
        out_override = call(model, [{"role": "user", "content": text}], options={"num_ctx": 16384})
        log({"experiment": "num_ctx_16384_override", "requested_tokens": n, "usage": out_override["usage"]})
    except requests.exceptions.RequestException as e:
        log({"experiment": "num_ctx_16384_override", "requested_tokens": n, "error": str(e)})


def experiment_marker_truncation(model: str, sizes: list[int]) -> None:
    print("\n=== Marker survival: which end of an oversize prompt does the model actually see? ===", flush=True)

    def rand() -> str:
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=8))

    for n in sizes:
        start_marker, end_marker = rand(), rand()
        body = make_filler(n)
        text = (
            f"BEGIN:{start_marker} {body} END:{end_marker}\n\n"
            "Output exactly two lines, nothing else:\n"
            "LINE1=<the 8-character code that appeared immediately after 'BEGIN:'>\n"
            "LINE2=<the 8-character code that appeared immediately before ' END:'>"
        )
        real_tokens = real_token_count(text)
        try:
            out = call(model, [{"role": "user", "content": text}], max_tokens=40)
            content = out["content"]
            log({
                "experiment": "marker_truncation",
                "requested_tokens": n,
                "real_tiktoken_count": real_tokens,
                "start_marker_seen": start_marker in content,
                "end_marker_seen": end_marker in content,
                "model_output": content,
                "usage": out["usage"],
            })
        except requests.exceptions.RequestException as e:
            log({
                "experiment": "marker_truncation",
                "requested_tokens": n,
                "real_tiktoken_count": real_tokens,
                "error": str(e),
            })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="ministral-3:3b")
    parser.add_argument("--fast", action="store_true", help="Smaller/quicker subset of sizes.")
    args = parser.parse_args()

    sizes = [200, 1000, 3000] if args.fast else [200, 1000, 3000, 6000, 9000]
    experiment_scaling(args.model, sizes)
    experiment_num_ctx_override(args.model, 6000)
    marker_sizes = [4000] if args.fast else [4000, 9000, 16000]
    experiment_marker_truncation(args.model, marker_sizes)


if __name__ == "__main__":
    main()
