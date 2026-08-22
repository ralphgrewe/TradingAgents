# Running TradingAgents with Local Models (Ollama/llama.cpp)

When running the TradingAgents pipeline against a local Ollama or llama.cpp server on a GPU with limited VRAM (e.g., 12 GB), generation may silently fall back to CPU inference, causing dramatic slowdowns. This guide explains the memory mechanism, the tuning knobs you have, and how to diagnose and optimize your setup.

## The Memory Offload Mechanism

The llama.cpp inference engine (which powers Ollama) holds two things in memory:

1. **Model weights** — the neural network parameters (fixed size for a given model/quantization)
2. **KV cache** — the Key-Value pairs accumulated during generation, grows with:
   - **Configured context length** (`num_ctx`)
   - **Number of layers** in the model
   - **Batch size** (usually 1 for single-turn inference)

If the sum of weights + KV cache exceeds available VRAM, llama.cpp automatically offloads layers to CPU, while leaving others on GPU. This mixed GPU/CPU split is transparent but has severe performance penalties.

### Symptoms of Partial CPU Offload

When this happens, you'll observe:

- **VRAM near-full** (e.g., 10–11 GB of 12 GB used)
- **Low GPU utilization** (e.g., 6–10%)
- **High `llama-server` CPU usage** (e.g., 300%+)
- **High system RAM usage** by `llama-server` (several GB)
- **Dramatic drop in tokens/sec** — generation becomes 5–20× slower than when fully GPU-resident

This is not a bug — it is the system responding correctly to resource constraints. The solution is to adjust the resource constraints themselves.

### Verify with `ollama ps`

At any time while a model is loaded, run:

```bash
ollama ps
```

The `PROCESSOR` column shows the split. Examples:

- `100% GPU` — fully GPU-resident (ideal)
- `X% GPU / Y% CPU` — mixed offload (e.g., `52% GPU / 48% CPU`)
- `100% CPU` — fully CPU-resident (slowest)

If you see a CPU/GPU split and your run is slow, your model is too large for your VRAM at the configured context length.

## Context-Length Knobs

### Ollama Server Default

By default, **Ollama uses a context window of 4,096 tokens** for all models. This is typically the KV cache budget you get unless you override it.

### How to Configure Context Length

Ollama provides three ways to set context length, in order of precedence:

#### 1. Model-Specific Modelfile Parameter (highest precedence)

If the model's Modelfile includes `PARAMETER num_ctx`, that value wins:

```
PARAMETER num_ctx 8192
```

To check or override, create a custom Modelfile:

```dockerfile
FROM llama2
PARAMETER num_ctx 8192
```

Then run:

```bash
ollama create my-llama2 -f Modelfile
ollama run my-llama2
```

#### 2. Server Environment Variable

Set `OLLAMA_CONTEXT_LENGTH` when starting the Ollama server. This becomes the default for all models that don't have a Modelfile-baked `num_ctx`:

```bash
OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

Or set it persistently in your shell profile or systemd service.

#### 3. API Request Parameter (lowest precedence, per-call)

This repo talks to Ollama via the **OpenAI-compatible endpoint** at `http://localhost:11434/v1` (overridable via `OLLAMA_BASE_URL`). The context length **comes from the Ollama server or model configuration**, not from TradingAgents' config.

If you need per-call context overrides, they would be set at the Ollama API level:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "llama2", "messages": [...], "options": {"num_ctx": 8192}}'
```

This repo does not expose such per-call overrides — context length is a server/model property.

## Practical Guidance for ~12 GB VRAM

The rule of thumb is: **fit the entire model + KV cache on GPU, or accept CPU offload penalties.**

### The Trade-Off Triangle

You have three levers:

1. **Model size** — use a smaller model (e.g., `mistral` 7B instead of `llama2` 13B)
2. **Quantization** — use a lower-bit version (Q4 vs. Q5, etc.) to shrink weights
3. **Context length** — reduce `num_ctx` to shrink the KV cache budget

### Example: RTX 3060 (12 GB)

- **7B model at Q4 quantization** (~4 GB weights) + **8K context** (~4 GB KV cache at 8K tokens) = ~8 GB total → **fits, fast**
- **13B model at Q4 quantization** (~7 GB weights) + **8K context** (~4 GB KV cache) = ~11 GB total → **barely fits, but risky**
- **13B model at Q4 quantization** + **32K context** (~16 GB KV cache) = **does not fit** → CPU offload

### Recommended Workflow

1. **Pick a model and quantization.** Start with a 7–8B model at Q4 for 12 GB VRAM.
2. **Set a modest context length** (e.g., 8,192 tokens):
   ```bash
   OLLAMA_CONTEXT_LENGTH=8192 ollama serve
   ```
3. **Run a test batch** through TradingAgents:
   ```bash
   ./venv/bin/python run_trading_agents.py stocks.json --show-summary
   ```
4. **While running, open another terminal and check:**
   ```bash
   ollama ps
   ```
   If you see `100% GPU`, you are good. If you see a CPU/GPU split, reduce context length and try again.

## Measuring Your Actual Context Sizes

The TradingAgents pipeline writes contexts of varying sizes depending on the analysis stage (analyst reports, research synthesis, risk debate, etc.). To understand which agents consume the most tokens:

1. **Per-call LLM log**: every `run_trading_agents.py` run generates a per-call JSONL log at `reports/<run_dir>/llm_calls.jsonl` (when the feature is enabled — see issue #138). This log includes:
   - Calling agent/graph node
   - Model name
   - Prompt token count and character count
   - Reported input/output tokens
   - Duration

2. **Full prompt dumps** (opt-in, see issue #139): set `prompt_dump_enabled=true` in config or `TRADINGAGENTS_PROMPT_DUMP_ENABLED=true` to write the complete prompt text of each call to disk, enabling offline analysis.

3. **End-of-run summary**: when the per-call log is enabled, TradingAgents prints aggregate per-agent statistics at the end of the run (total calls, total/max prompt tokens per agent).

### Find Your Bottleneck

Once you have the per-call log, sort by `prompt_tokens` (descending) to find which agent consumes the most context. The largest consumers are typically:

- **Researcher** — synthesis of analyst reports + evidence pack (can be 10–15K tokens)
- **Risk Debate** — repeated re-reading of the trader's plan across multiple speakers
- **Portfolio Manager** — aggregate of all prior stage outputs

Use this data to inform which optimizations matter most for your setup (see "Candidate optimizations" below).

## Candidate Optimizations (Follow-Up Issues)

Once you've measured where the tokens go (via the per-call log), these optimizations can reduce context burden. **Do not implement these yet** — file separate issues once you have data:

1. **Truncate/summarize analyst reports** before later stages — keep the key findings but drop verbose reasoning. (Impact: saves 20–40% on post-analyst context.)
2. **Lower `research_evidence_token_budget`** — reduce the evidence pack from web search in researcher mode. (Config key: `research_evidence_token_budget`, default 3000.) (Impact: saves 1–2K tokens in researcher stage.)
3. **Reduce debate rounds** — `max_debate_rounds` (research debate) and `max_risk_discuss_rounds` (risk debate) re-read the same plans repeatedly. Setting both to 0 removes debate entirely. (Impact: saves 3–8K tokens per debate run.)
4. **Run fewer analysts** — `selected_analysts` (default: market, social, news, fundamentals) controls which analysts run. Remove slower or redundant ones. (Impact: saves 5–15K tokens, reduces run time.)
5. **Sequential chunked processing** — for large texts (e.g., news articles in the fundamental analyst), process them in chunks and merge results instead of feeding the full text inline. (Complex; defers to a dedicated issue.)
6. **Flash Attention for KV cache** — if using llama.cpp 0.3.0+, Flash Attention can reduce KV cache memory by 40–60% without changing context length. Check your Ollama/llama.cpp version and enable if available.

**Do not guess which optimization to try** — start with a measurement run to see which agents burn the most tokens, then file an issue for a concrete optimization targeting that agent.

## See Also

- **Issue #137** — the parent tracking issue for LLM context instrumentation and this guide
- **Issue #138** — per-call LLM call log implementation
- **Issue #139** — opt-in full prompt dumps
- **CLAUDE.md** — project architecture and LLM provider configuration
- **Ollama Official Docs** — [FAQ](https://docs.ollama.com/faq) for complete context-length configuration and troubleshooting
