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

> **See also**: [`docs/analysis/prompt-truncation-diagnosis.md`](analysis/prompt-truncation-diagnosis.md)
> (issue #148) diagnoses two token-accounting anomalies observed against a live local Ollama server — a
> provider-reported `input_tokens` figure that becomes a wrong constant (2051) above a size threshold, and
> a ~4096-token ceiling below what this codebase's own prompts can reach. It also recommended the
> `ollama_num_ctx` knob and the oversize-prompt check documented below (issue #149), both now implemented.

### Ollama Server Default

Ollama does **not** use a fixed context window by default. With `OLLAMA_CONTEXT_LENGTH` unset, the server
auto-fits context length to free VRAM at model-load time — `ollama serve --help` (verified against Ollama
0.32.3) documents this as **"4k/32k/256k based on VRAM"**. This was confirmed live during the #148
investigation: the same model's context varied across separate loads on one machine purely as a function
of how much VRAM a concurrent process had claimed at that moment (observed values: 11677, 10887, 9622,
9596, and — under real memory pressure — exactly **4096**). Do not assume "4096" as a safe floor or a
predictable value; see the diagnosis linked above for the full picture, including how this interacts with
the KV-cache/offload mechanism described next.

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

This repo talks to Ollama via the **OpenAI-compatible endpoint** at `http://localhost:11434/v1` (overridable via `OLLAMA_BASE_URL`). Per-call overrides are set at the Ollama API level via a non-standard `options` request field:

```bash
curl http://localhost:11434/api/chat \
  -d '{"model": "llama2", "messages": [...], "options": {"num_ctx": 8192}}'
```

**As of issue #149, this repo *does* expose that override**, via the `ollama_num_ctx` config key
(env `TRADINGAGENTS_OLLAMA_NUM_CTX`, default unset). When set, `TradingAgentsGraph` forwards it as
`extra_body={"options": {"num_ctx": N}}` on every `ChatOpenAI` request for the `ollama` provider
(`tradingagents/llm_clients/openai_client.py`'s `OpenAIClient.get_llm`) — the same mechanism the
`curl` example above uses, just sent automatically on every call instead of typed by hand, and
identical across every call for the life of the run.

**As of issue #154, leaving `ollama_num_ctx` unset no longer means "no control at all."** Instead of a
single fixed value, `num_ctx` is derived **per request** from that request's own measured prompt size:

```
needed  = ceil(prompt_tokens_estimated * context_window_safety_margin) + ollama_num_ctx_response_headroom
num_ctx = min(needed, ollama_num_ctx_max)
```

This is computed fresh for every call (`OllamaChatOpenAI._get_request_payload` in
`tradingagents/llm_clients/openai_client.py`) and attached the same way (`extra_body.options.num_ctx`),
so a short analyst prompt gets a small `num_ctx` and a long Portfolio Manager prompt with a full debate
history gets a bigger one — every agent's prompt reaches Ollama untruncated instead of only the ones
that happen to fit under one static value picked for the whole run. `ollama_num_ctx_max` (default
`32768`, env `TRADINGAGENTS_OLLAMA_NUM_CTX_MAX`) is the ceiling on the derived value, and
`ollama_num_ctx_response_headroom` (default `2048`, env
`TRADINGAGENTS_OLLAMA_NUM_CTX_RESPONSE_HEADROOM`) is generation headroom reserved on top of the prompt
requirement (Ollama's `num_ctx` bounds prompt + completion together). When a request's `needed` value
exceeds `ollama_num_ctx_max`, the run aborts (see "Oversize-Prompt Enforcement" below) instead of
sending a prompt that would be silently truncated.

Setting `ollama_num_ctx` explicitly still bypasses all of this outright — no derivation runs, and the
one fixed value is used for every call, exactly as it was before #154. This remains useful when you want
a single predictable VRAM footprint (see "Practical Guidance" below) rather than one that varies call to
call with prompt size.

### Oversize-Prompt Enforcement (issues #149, #154)

Before dispatching any LLM call, `ContextWindowGuardHandler` (`tradingagents/llm_call_log.py`) compares
that call's #147 prompt-size estimate (`prompt_tokens_estimated`, tiktoken-based where available —
**not** the provider-reported `input_tokens`, which #148 showed becomes an unreliable constant on Ollama
past a size threshold) against the model's *known* context window. "Known" means one of two things:

- **A fixed window** (#149, unchanged): the model has an entry in the mapping `TradingAgentsGraph`
  builds from `ollama_num_ctx` (applied to both `quick_think_llm` and `deep_think_llm` when the
  configured provider is `ollama` **and `ollama_num_ctx` is explicitly set**) and/or
  `context_window_overrides` — a `{model_name: context_window_tokens}` dict config key, for any
  provider/model to opt in explicitly (explicit overrides win over the ollama-derived value for the
  same model name).
- **A derived ceiling** (#154, the default for `ollama` when `ollama_num_ctx` is left unset): rather
  than a fixed window, each request's own `needed` value (see the derivation formula above) is compared
  against `ollama_num_ctx_max` directly.

A model with **neither** is passed through **unchecked** — an unknown limit is never enforced, matching
this project's stance of not fabricating context-window numbers it doesn't actually know. In practice
this only applies to non-`ollama` providers without a `context_window_overrides` entry, since `ollama`
now always has *some* known limit (fixed or derived) by default.

When a prompt *is* checked and it doesn't fit, the run aborts immediately with a
`PromptContextOverflowError` naming the agent, model, measured/adjusted prompt size, and the limit that
was exceeded, and points back at this section and at `context_window_overrides`/`ollama_num_ctx`/
`ollama_num_ctx_max` for the remedy. The event is also written to that ticker's `llm_calls.jsonl` with
the same error-record shape a failed call gets (`input_tokens`/`output_tokens` null, `error` populated),
so a batch run (`run_trading_agents.py`) leaves diagnosable evidence behind even though it aborts that
ticker rather than continuing on a possibly-truncated prompt. Successful `ollama` calls also record the
`num_ctx` that was actually sent (`ollama_num_ctx` field, `None` when derivation doesn't apply to that
call), so truncation can be audited after the fact even without a failure.

**Escape hatch**: set `context_window_check_enabled=False` (env
`TRADINGAGENTS_CONTEXT_WINDOW_CHECK_ENABLED=false`) to disable the *abort* entirely. This only disables
the check — for `ollama` with `ollama_num_ctx` unset, `num_ctx` is still derived and sent per request as
described above (the check and the derivation are independent; disabling the former does not stop the
run from still trying to size `num_ctx` correctly, it just stops aborting when that isn't possible).

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

The TradingAgents pipeline writes contexts of varying sizes depending on the analysis stage (analyst reports, research synthesis, risk debate, etc.). To understand which agents consume the most tokens, use the per-call LLM log (issue #138, implemented in commit 774a6ba):

1. **Per-call LLM log**: `LLMCallLogHandler` (`tradingagents/llm_call_log.py`) is wired into every run's callbacks and appends one JSON object per line (JSONL) for every LLM call. It is controlled by the `llm_call_log_enabled` config key (env `TRADINGAGENTS_LLM_CALL_LOG_ENABLED`), **default `True`** — logging is on unless you disable it. Each record includes:
   - `ticker` and `date` the call belongs to
   - `run_id` and `agent` (the LangGraph node that made the call)
   - `model`, `message_count`, `prompt_chars`, `prompt_tokens_estimated` (chars/4 heuristic)
   - `input_tokens` / `output_tokens` (provider-reported, when available; e.g. Ollama's OpenAI-compatible endpoint supplies these)
   - `duration_seconds`
   - `error` — `null` for a successful call, otherwise a `"TypeName: message"` string for a failed call (failed calls are logged too, via `on_llm_error`, with null token counts)

   **Where the log lands** differs between the two entry points:
   - `run_trading_agents.py` (batch/multi-ticker): one JSONL file **per ticker** at `<report_dir>/<TICKER>_<DATE>_<TIMESTAMP>/llm_calls.jsonl` — the same per-ticker directory `save_report_to_disk` uses — so a multi-ticker `stocks.json` batch stays separable.
   - `cli/main.py` (interactive, single ticker+date per run): `<results_dir>/<ticker>/<date>/llm_calls.jsonl`.

2. **End-of-run summary**: alongside the JSONL log, both entry points write a per-agent aggregate (call count, failed-call count, total/max estimated prompt tokens, total output tokens) to `llm_calls_summary.json` in the same directory as that run's `llm_calls.jsonl`. `run_trading_agents.py` additionally writes a batch-wide roll-up across all tickers to `<report_dir>/llm_calls_summary.json`. Passing `--show-summary` to `run_trading_agents.py` prints these — per ticker as each run finishes, and once for the whole batch at the end (only when more than one ticker ran).

3. **Full prompt dumps** (issue #139, implemented in commit d850de5): an **opt-in** mode that writes the complete rendered prompt of every LLM call to disk, so you can read exactly what an expensive call sent. It is controlled by the `llm_call_log_prompts` config key (env `TRADINGAGENTS_LLM_CALL_LOG_PROMPTS`), **default `False`** — dumps are large and may contain fetched data (news text, fundamentals, web-search evidence) that users don't always want written to disk, so you opt in explicitly. It is additionally gated on `llm_call_log_enabled` being `True`: with the per-call log switched off, nothing is dumped either.

   When enabled, the handler writes one JSON file per LLM call into a `prompts/` subdirectory **next to that run's `llm_calls.jsonl`** (so `<...>/prompts/` in whichever of the two locations above applies), named `<run_id>.json`. Each file contains all messages of the call in order:

   ```json
   {
     "format": "chat_messages",
     "messages": [
       {"role": "SystemMessage", "content": "..."},
       {"role": "HumanMessage", "content": "..."}
     ]
   }
   ```

   `role` is the LangChain message class name (`SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`, ...). Non-chat calls, which carry plain string prompts rather than message objects, are written with `"format": "prompts"` and a `"prompt"` role instead. Failed calls are dumped too, so a call that blew up the context window leaves its prompt behind for inspection.

   Every `llm_calls.jsonl` record gains a `prompt_dump_path` field to tie the two together: the relative path `prompts/<run_id>.json` when dumping is on, or `null` when it is off. So the workflow is: sort the JSONL by `prompt_tokens_estimated`, take the worst offender's `prompt_dump_path`, and open that file.

   ```bash
   # Enable dumps for one batch run
   TRADINGAGENTS_LLM_CALL_LOG_PROMPTS=true \
     ./venv/bin/python run_trading_agents.py stocks.json --show-summary
   ```

### Find Your Bottleneck

Once you have the per-call log, sort its JSONL records by `prompt_tokens_estimated` (descending) to find which agent consumes the most context. The largest consumers are typically:

- **Researcher** — synthesis of analyst reports + evidence pack (can be 10–15K tokens)
- **Risk Debate** — repeated re-reading of the trader's plan across multiple speakers
- **Portfolio Manager** — aggregate of all prior stage outputs

Use this data to inform which optimizations matter most for your setup (see "Candidate optimizations" below).

## Candidate Optimizations (Follow-Up Issues)

The measurement run this section used to recommend has now happened:
**[`docs/analysis/prompt-size-findings.md`](analysis/prompt-size-findings.md)**
(issue #144) reports the actual corpus-wide numbers — which agent's prompts are largest and slowest, and
what a Portfolio Manager prompt is made of, segment by segment, with reproducible commands and cited
examples. Read that report before choosing an optimization; the headline findings are that the Portfolio
Manager is the critical prompt on both size and wall-time, that risk-debate history (~34% median) and
analyst reports (~32% median) dominate its prompt today, and that memory injection (~14% today, PM-only) is
the one segment with no size cap and is projected to overtake both of those as ticker history accumulates.

Ranking concrete optimizations against those numbers — with estimated savings, effort, and risk for each —
is now done: **[`docs/analysis/prompt-optimization-options.md`](analysis/prompt-optimization-options.md)**
(issue #145) ranks the available options into a shortening-first (Tier 1) vs. long-prompt-handling
fallback (Tier 2) list, gives an explicit cross-tier ranking with a "start here" recommendation, and ends
in a selection checklist. **Do not guess which optimization to try**: read the findings report, then the
optimization-options report, and pick from its checklist rather than this list.

## See Also

- **Issue #137** — the parent tracking issue for LLM context instrumentation and this guide
- **Issue #138** — per-call LLM call log implementation (`tradingagents/llm_call_log.py`, `llm_call_log_enabled` config key)
- **Issue #139** — opt-in full prompt dumps (`llm_call_log_prompts` config key, default off; dumps land in `prompts/<run_id>.json` next to `llm_calls.jsonl`)
- **CLAUDE.md** — project architecture and LLM provider configuration
- **Ollama Official Docs** — [FAQ](https://docs.ollama.com/faq) for complete context-length configuration and troubleshooting
