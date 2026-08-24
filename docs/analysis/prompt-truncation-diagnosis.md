# Prompt-truncation diagnosis: the constant 2051 and the 4096 ceiling

Status: diagnosis report (2026-08-24), issue #148, part of the #146 tracking issue. Built on #147's
tiktoken-based `prompt_tokens_estimated` (commit `de1ce762d161966f624c77b45ed5585a120cbcb0`) as the size
yardstick, and on a **live controlled reproduction against a real local Ollama server** (Ollama 0.32.3,
one NVIDIA RTX 3060 12 GB, models `ministral-3:3b` / `ministral-3:8b`) run for this issue. Every number
below is either read directly from the `reports/` corpus or produced by
`scripts/repro_ollama_token_anomaly.py`, which is checked into this repo and runnable against any local
Ollama instance (see "Reproduction recipe").

## TL;DR

- **Anomaly A (constant 2051) is reproduced live and has a well-evidenced, if not 100%-source-pinned, root
  cause**: once a prompt's true size (as tokenized by the model's own/native tokenizer, not this
  codebase's chars/4 or tiktoken estimate) crosses roughly 3.5–4.6K tokens, Ollama 0.32.3's
  OpenAI-compatible endpoint (`/v1/chat/completions`) starts reporting a **constant, wrong**
  `usage.prompt_tokens` of exactly **2051** — verified at real prompt sizes of ~6K, ~9K, and ~16K tokens,
  **and unaffected by an explicit `num_ctx=16384` override**, which rules out "the auto-fit context window
  is merely too small" as A's direct cause. This is a genuine provider-side reporting bug, not a real
  token count, and not primarily a context-size problem.
- **Anomaly B (the ~4096 ceiling) is explained by a genuinely non-deterministic, VRAM-dependent effective
  context window**: this Ollama version does **not** default to a fixed 4096-token context (contrary to
  what this repo's own `docs/local-models.md` previously stated) — with `OLLAMA_CONTEXT_LENGTH` unset, it
  auto-fits context length to free VRAM at model-load time ("4k/32k/256k based on VRAM" per `ollama serve
  --help`). This was directly observed live during this investigation: `ministral-3:8b` loaded at exactly
  `CONTEXT 4096` when free VRAM was low (855 MiB), and `ministral-3:3b` was pushed into partial CPU/GPU
  offload — the same failure mode `docs/local-models.md` already documents for a different reason. The
  historical corpus's 4041–4094 cluster is consistent with calls that landed on a 4096-tier auto-fit.
- **Both anomalies trace to the same root gap named in hypothesis 1**: nothing in this codebase sets
  `num_ctx` for the `ollama` provider (verified in `openai_client.py` and `trading_graph.py`), so the
  effective context window for every call is left entirely to Ollama's per-load VRAM-driven auto-fit —
  which is neither fixed nor visible to this codebase.
- **Yes, prompts in the existing corpus were very likely truncated** for a meaningful subset of calls,
  concentrated in Portfolio Manager and Researcher (the two largest-prompt agents per #144). See "The
  decisive question" below.
- **Recommendation for #149**: plumb an explicit `num_ctx` (via LangChain's `extra_body`/`model_kwargs` on
  `ChatOpenAI`, forwarded through `OpenAIClient`) so the context window is a known, controlled quantity
  instead of an invisible auto-fit value, and add a pre-flight oversize check using the already-computed
  `prompt_tokens_estimated` against that known `num_ctx`. Do **not** try to fix `input_tokens` itself as
  the truncation signal — Anomaly A shows it can't be trusted once truncation-adjacent behavior kicks in.

## Environment

- Ollama server version: `0.32.3` (`ollama --version`), reachable at `http://localhost:11434` — **Ollama
  is running in this environment** and was used for all reproduction below.
- GPU: one NVIDIA RTX 3060, 12288 MiB total VRAM (`nvidia-smi`).
- Models installed: `ministral-3:3b` (Q4_K_M, 3.8B params, GGUF `context_length` metadata 262144),
  `ministral-3:8b` (Q4_K_M, 8.9B params, GGUF `context_length` metadata 262144), `qwen3.5:9b`. The GGUF
  `context_length` field (`ollama show`) is the model's architectural maximum, **not** what Ollama actually
  serves at runtime — see below.
- `ollama show ministral-3:3b --modelfile`: only `PARAMETER temperature 0.15` is set. **No `PARAMETER
  num_ctx` line exists in either model's Modelfile** — confirms hypothesis 1's premise that nothing pins
  the context length at the model level.
- `OLLAMA_CONTEXT_LENGTH` is **not set** anywhere in this environment: not in the current shell, not in
  `systemctl show ollama --property=Environment`, not in the systemd unit (`/etc/systemd/system/
  ollama.service` and its `override.conf` only set `OLLAMA_HOST`). So the server falls back to its
  documented default: **`ollama serve --help`** for this version states plainly:
  `OLLAMA_CONTEXT_LENGTH   Context length to use unless otherwise specified (default: 4k/32k/256k based on
  VRAM)` — a **VRAM-tiered auto-fit**, not the flat 4096 this repo's own docs previously described.
- The live `llama-server` process Ollama spawns for `ministral-3:3b` was observed (via `ps aux`) with:
  `-c 11677 -np 1 --log-verbosity 4 --no-jinja --chat-template chatml --flash-attn auto -b 1024 -ub 1024
  --context-shift --keep 4`. **The `-c` (context) value changed across separate loads during this
  investigation — 11677, then 9596, then 9622, then 10887, then 4096 (for `ministral-3:8b` specifically,
  under memory pressure)** — confirming the context length is decided fresh at each model load based on
  free VRAM at that moment, not a fixed per-model or per-server constant. `--context-shift` confirms the
  server is configured to silently shift/discard context on overflow rather than error.
- This machine was **not idle** during the investigation: an unrelated process
  (`python -m rss_pipeline.extract_graph --model ollama:ministral-3:3b`) was concurrently hitting the same
  Ollama server, which at one point pushed `ministral-3:3b` into a `36%/64% CPU/GPU` split (via `ollama
  ps`) — directly reproducing the "Symptoms of Partial CPU Offload" scenario `docs/local-models.md`
  already documents, and independently confirming this deployment's context/serving behavior is sensitive
  to real-world concurrent load, not just this repo's own request pattern.

## Corpus re-verification

Re-ran the counts against the current `reports/` corpus (33 run directories with `llm_calls.jsonl`, dates
2026-08-22/23, 521 calls, 520 with non-null `input_tokens`) rather than trusting the issue body's numbers
as-is:

```bash
jq -r 'select(.input_tokens==2051)' reports/*/llm_calls.jsonl | grep -c input_tokens          # 202
jq -r 'select(.input_tokens==2051) | .agent' reports/*/llm_calls.jsonl | sort | uniq -c
#  66 Researcher, 32 Portfolio Manager, 31 Neutral Analyst, 31 Conservative Analyst,
#  20 Sentiment Analyst, 17 Trader, 5 Aggressive Analyst   (matches the issue body)
jq -r 'select(.input_tokens==2051) | .message_count' reports/*/llm_calls.jsonl | sort | uniq -c
# 165 have message_count==1, 37 have message_count==2
```

**Confirmed**: 202/521 = 39% of calls report exactly 2051, matching the issue. **One correction to the
issue body**: `message_count==1` holds for 165/202 (82%) of the 2051 instances, not "every observed
instance" — 37 (mostly Trader and Sentiment Analyst calls) have `message_count==2`. This matters for the
root-cause story below: message count is a *correlate*, not the trigger (see Anomaly A).

For the 4096 ceiling, I could **not** reproduce the issue's stated "85 calls" / "6 Portfolio Manager calls
on :8b" from the current corpus snapshot:

```bash
jq -r 'select(.input_tokens>=4041 and .input_tokens<=4094)' reports/*/llm_calls.jsonl | grep -c input_tokens   # 15
jq -r 'select(.input_tokens>=4041 and .input_tokens<=4094) | [.agent,.model] | @tsv' reports/*/llm_calls.jsonl | sort | uniq -c
#   8 Trader / ministral-3:3b
#   5 Aggressive Analyst / ministral-3:3b
#   1 Neutral Analyst / ministral-3:3b
#   1 Conservative Analyst / ministral-3:3b
# (zero Portfolio Manager, zero ministral-3:8b, even widening the band to 3900-4096 -> 40 calls, still zero PM/8b)
```

This is a real discrepancy from the issue body's figures that I cannot explain from the data alone (same
33-run corpus, same field) — possibly a different band/threshold or corpus snapshot was used originally.
It does not change the diagnosis (the *mechanism*, reproduced live below, is the same either way), but is
recorded here rather than silently reconciled, since I could not verify the original 85/6 figures.

## Reproduction recipe

`scripts/repro_ollama_token_anomaly.py` (checked into this repo) runs three experiments against a live
Ollama server over its OpenAI-compatible endpoint directly (bypassing this codebase's own client, so
results aren't an artifact of `tradingagents/`' own code):

```bash
./venv/bin/python scripts/repro_ollama_token_anomaly.py --model ministral-3:3b
# --fast for a smaller/quicker subset on a loaded/shared GPU
```

It (1) sends isolated single-message prompts of increasing known size and logs `usage.prompt_tokens`
against a real tiktoken count of what was sent; (2) repeats one size with and without an explicit
`num_ctx=16384` override (Ollama's OpenAI-compat `options` extension field); (3) embeds a unique random
marker at the very start and a different one at the very end of an oversize prompt and asks the model to
echo both back, to test whether front-of-prompt or back-of-prompt content survives.

**A note on reproducibility**: this machine's GPU is shared with other concurrent processes (see
"Environment" above), so absolute timings and even whether a given call completes before timeout will
vary run to run — this is itself part of the diagnosis, not just a caveat (see Anomaly B). The specific
constant value (2051) and the qualitative "reported becomes wrong and constant past a size threshold,
independent of num_ctx" finding reproduced consistently across every trial run for this report.

## Anomaly A: the constant 2051

### What was reproduced

Isolated, uniquely-worded prompts (no shared prefix, so prompt-cache reuse across calls is not a
confound), single `user` message, sent directly to `POST /v1/chat/completions`:

| requested filler tokens | real tiktoken count of full message | reported `usage.prompt_tokens` |
|---:|---:|---:|
| 200 | 212 | 784 |
| 1,000 | 1,013 | 1,649 |
| 3,000 | 3,013 | 3,809 |
| 6,000 | 6,013 | **2,051** |
| 9,000 | 9,013 | **2,051** |

Below the threshold, reported tokens scale with real size (with a ~1.3–1.6x multiplier over our tiktoken
estimate — see "Anomaly B" for why). Past a threshold somewhere between ~3,000 and ~3,500 real
(o200k_base) tokens, the reported count **stops scaling and locks to exactly 2051**, no matter how much
larger the real prompt gets. Re-running the same test via a shared-prefix pipeline (simulating several
agents reusing similar preamble content, one call per size, sizes increasing) showed the identical
pattern: 2,514 real tokens → reported 3,270 (correct-ish); 3,515 real tokens → reported 2,051; 4,515 real
tokens → reported 2,051. And the marker-survival experiment (see Anomaly B) reproduced the constant again
at real sizes of 4,055, 9,055, **and 16,055** tokens — always exactly 2051.

### Ruling out "context window too small"

The most obvious hypothesis is that 2051 is some function of the *effective* context window at that
moment (hypothesis 1 folded into hypothesis 2). This was tested directly: the same 6,013-token prompt was
sent once with Ollama's default auto-fit context (which was 9,596–11,677 tokens throughout this
investigation for `ministral-3:3b` — comfortably larger than the prompt) and once with an **explicit
`options: {"num_ctx": 16384}`** override in the request body (Ollama's OpenAI-compat endpoint honors this
non-standard field). **Both reported exactly 2051.** A prompt well within a 16,384-token window should
never be truncated or short-counted, yet the reported figure was identical to the untruncated-context
case. This rules out "the model's context window happened to be too small" as Anomaly A's *direct*
trigger — the auto-fit context length being large (9.5K–11.7K, confirmed via `ollama ps` throughout) did
not prevent the constant from appearing.

### What the evidence points to instead

- 2051 = 2 × 1,024 + 3. The live `llama-server` process for this model was launched with **`-b 1024 -ub
  1024`** (batch size / micro-batch size) — visible directly in `ps aux` output during this investigation.
  A prompt-token counter that only accumulates for a bounded number of prefill batches (e.g. an early-exit
  or truncated-accumulation bug in Ollama's OpenAI-compat usage-accounting translation layer, distinct
  from the actual `n_ctx` the runner is using) is at least as consistent with what was measured as any
  `num_ctx`-based theory, and is *more* consistent with the num_ctx-override result above than a
  context-shift/discard explanation (`n_ctx / 2` for an effective ~4096-token window would also land near
  2048, but that theory is contradicted by the constant persisting under an explicit 16,384-token window).
- `message_count==1` (present in 82%, not 100%, of the corpus's 2051 instances — see "Corpus
  re-verification") is a **correlate, not the cause**: this reproduction triggered the identical constant
  with single-message prompts (all experiments above). The correlation in the corpus most likely arises
  because the agents that build their entire prompt as one large `HumanMessage` string — Portfolio Manager
  (`tradingagents/agents/managers/portfolio_manager.py:126,150`: `messages = [HumanMessage(content=prompt)]`
  or `structured_llm.invoke(prompt)`), and similarly Researcher — are exactly the agents with the largest
  prompts (per #144), so they cross the ~3.5K-token trigger threshold far more often than smaller-prompt
  agents. Message count itself does not appear in any plausible mechanism this reproduction surfaced.
- **What would fully pin the exact source line**: `OLLAMA_DEBUG=1 ollama serve` log inspection during a
  triggering call, or reading Ollama's Go source for the OpenAI-compat `usage` translation in this
  version (0.32.3) — genuinely out of reach of black-box HTTP reproduction and not attempted here. This is
  recorded as the one open item on Anomaly A: **the qualitative behavior (constant, size-independent,
  num_ctx-independent, wrong) is fully confirmed by live reproduction; the exact internal code path that
  produces the specific value 2051 is not.**

## Anomaly B: the ~4096 ceiling

### Direct live evidence of a VRAM-driven auto-fit context

This is the most concrete finding of the investigation. Partway through reproduction, `ollama ps` showed:

```
NAME              ID              SIZE      PROCESSOR          CONTEXT    UNTIL
ministral-3:8b    1922accd5827    5.6 GB    100% GPU           4096       4 minutes from now
ministral-3:3b    f04aa1c738f6    4.3 GB    36%/64% CPU/GPU    10887      Stopping...
```

with `nvidia-smi` showing only 855 MiB free VRAM at that moment. **`ministral-3:8b` was auto-fit to
exactly a 4096-token context window** — the same figure the corpus's Anomaly B clusters just under — while
`ministral-3:3b` was simultaneously pushed into partial CPU/GPU offload. This is not a fixed default; it
is what Ollama's auto-fit algorithm computed *at that specific moment*, driven by how much VRAM a
concurrent process had claimed. Earlier in the same investigation, with more free VRAM, `ministral-3:3b`
auto-fit to 11,677, then 9,596, then 9,622, then 10,887 across separate loads (`OLLAMA_KEEP_ALIVE`
defaults to 5 minutes, so a model reloads — and re-runs auto-fit — repeatedly over the course of a longer
TradingAgents run).

**Conclusion**: the corpus's 4041–4094 cluster is fully consistent with calls that happened to run while
Ollama's auto-fit had landed on (or near) the 4k tier — which this investigation shows happens under
realistic, moderate VRAM pressure, not an edge case. `docs/local-models.md`'s previous claim that "by
default, Ollama uses a context window of 4,096 tokens for all models" is **outdated for this Ollama
version** (0.32.3) — the real default is a VRAM-tiered auto-fit ("4k/32k/256k" per `ollama serve --help`)
that can land anywhere in that range, including exactly 4096 under memory pressure. This document's
existence is cross-referenced from that section; see "Cross-reference" below for the correction made there.

### Why reported exceeds estimated (not a new bug)

The corpus shows "reported > estimated" for the ~4096-band calls (estimated ~3000–3400, reported
4041–4094). This reproduction's low-size (pre-2051-threshold) calibration data explains this directly —
it is a genuine tokenizer-family mismatch, not a new anomaly:

| real tiktoken (o200k_base) count | reported `usage.prompt_tokens` | ratio |
|---:|---:|---:|
| 212 | 784 | 3.70x (small-prompt, template-overhead-dominated) |
| 1,013 | 1,649 | 1.63x |
| 2,514 | 3,270 | 1.30x |
| 3,013 | 3,809 | 1.26x |

Ministral's own tokenizer (what Ollama's usage accounting actually counts against) produces **~1.26–1.63x
more tokens** than this codebase's chars/4 or tiktoken(o200k_base) estimate for the same text once
prompts are a few hundred tokens or larger (the overhead-dominated ratio at very small sizes is a
separate, template/special-token effect). A prompt this codebase estimates at ~3,200 tokens is very
plausibly ~4,000–4,200 tokens by Ministral's real tokenizer — landing right at a 4096 ceiling. This is the
same direction and magnitude of disagreement #147's own calibration write-up already flagged (tiktoken and
chars/4 "track each other closely" and both under-count relative to reported `input_tokens` on this
corpus) — this investigation confirms the mismatch is a genuine tokenizer-family difference, not
instrumentation error introduced by #147.

### Does content actually get dropped?

Yes, most likely, though this reproduction's evidence here is directionally strong but not airtight.
Marker-survival tests (unique 8-character codes placed at the very start and very end of an oversize
prompt, model asked to echo both back) at 4,055 / 9,055 / 16,055 real tokens (all well beyond the ~4096
ceiling, sent under Ollama's default auto-fit context):

| real tokens | start marker recalled? | end marker recalled? |
|---:|:---:|:---:|
| 4,055 | no | yes (exact) |
| 9,055 | no | no (garbled, closer to the end marker than the start) |
| 16,055 | no | yes (exact) |

**In every trial, the start-of-prompt marker was never correctly recalled, while the end-of-prompt marker
was recalled correctly or near-correctly in 2 of 3 trials.** This is consistent with front-of-prompt
content being silently dropped (the classic "keep the tail, discard the head" behavior of context-shift
overflow handling) rather than the model simply failing to attend to *any* far-away content symmetrically.
One caveat: a repeat of this test with an explicit `num_ctx=16384` override (comfortably larger than the
9,053-token prompt used) failed to recall **either** marker — a result inconsistent with clean
tail-preserving truncation, and plausibly attributable to this run coinciding with the GPU-contention
episode described above (partial CPU/GPU offload degrades a small model's generation quality, and 3B
instruction-following on a 9K-token needle-in-haystack task is not perfectly reliable even in-window).
**I cannot cleanly separate "hard context truncation" from "small-model long-context recall failure" from
black-box behavioral testing alone** — settling this fully would need either `OLLAMA_DEBUG=1` server logs
showing the actual token count fed to the model per request, or repeating the marker test many times on an
otherwise-idle GPU to get a clean success rate. Recorded as the one open item on Anomaly B.

## The decisive question: were prompts truncated in the existing corpus?

**Most likely yes, for a meaningful subset of calls — concentrated in the agents with the largest prompts.**

- The 15 calls in the corpus's 4041–4094 band (Trader, Aggressive Analyst, and a few Conservative/Neutral
  calls, all `ministral-3:3b`) very plausibly ran while the auto-fit context was down near the 4k tier
  (directly reproduced as a live, real occurrence above) — their real/native-tokenizer prompt size,
  applying the ~1.26–1.63x under-estimate factor found above to their ~2,934–3,453 estimated-token range,
  was likely already at or above whatever the actual ceiling was that call. **Likely truncated.**
- The 202 "2051" calls' true size cannot be read off the log at all (the reported figure is meaningless
  post-threshold), but their `prompt_tokens_estimated` already spans 2,989–10,917 — applying the same
  tokenizer factor implies true sizes plausibly in the ~3,800–17,700 range. Whether these were truncated
  depends on what the auto-fit context happened to be at that specific call (this investigation showed it
  ranges from 4,096 to 11,677+ depending on concurrent VRAM pressure at load time) — **plausible but not
  certain for every instance**; the largest of these (Researcher, Portfolio Manager) are the most likely
  candidates given #144's finding that Portfolio Manager prompts alone run up to 12,289 estimated tokens.
- **Which agents matter most**: Portfolio Manager (32 of the 2051 instances) is the single most
  consequential agent affected — it is already established (#144) as the largest and most critical prompt
  in the pipeline, and its output is the final BUY/SELL/HOLD decision. Researcher (66 instances) is the
  most frequently affected by raw call count.
- **From roughly what size onward**: real (native-tokenizer) prompt sizes above roughly 3.5–4.5K tokens
  are at risk of the Anomaly A reporting bug; content truncation risk depends on the auto-fit context at
  that moment, which this investigation showed ranges as low as 4,096 under realistic VRAM pressure.
- **Which segment is at risk**: per #144's Portfolio Manager segment breakdown
  (`docs/analysis/prompt-size-findings.md`), the prompt is assembled in this order: header + rating-scale
  instructions → past-context/memory injection (14.3% median) → analyst reports (32.0% median) →
  risk-debate history (34.4% median, the largest segment, written **last**) → closing note. The
  marker-survival evidence above (start-of-prompt content lost, end-of-prompt content more likely to
  survive) implies that **if** a Portfolio Manager call was truncated, the segments most at risk of
  silently vanishing are the header/instructions, the memory injection, and the analyst reports — while
  the risk-debate history, composed last and closest to the generation point, is the segment most likely
  to survive. This is the opposite of what "recency wins" intuition would suggest for *stage order*, and
  means the decision-critical analyst evidence — not the risk-team back-and-forth — is what is most
  exposed when a Portfolio Manager prompt overflows its context.
- **What it means for corpus validity**: for the calls affected (real fraction not fully quantifiable from
  the logs alone, per Anomaly A's reporting bug — but the 2051 cluster alone covers 39% of all calls in
  the corpus, and Portfolio Manager/Researcher are both represented), the final trading decision may have
  been made with the earliest-composed context silently missing. This does not necessarily invalidate
  every decision in `reports/` — many calls are well under any plausible context ceiling — but it means the
  corpus cannot currently be treated as "every call saw its full intended prompt" without per-call
  verification, which the current logging does not support (see recommendation below).

## Recommendation for #149

1. **Make the context window a known, controlled quantity instead of an invisible auto-fit value.** Add
   `num_ctx` (and optionally other Ollama-native `options`) as a pass-through kwarg on the `ollama`
   provider path: LangChain's `ChatOpenAI` forwards arbitrary extra body fields via `extra_body`/
   `model_kwargs`, and this reproduction confirmed Ollama's OpenAI-compat endpoint honors
   `{"options": {"num_ctx": N}}` in the request body. This requires a config key (e.g.
   `ollama_num_ctx` / `TRADINGAGENTS_OLLAMA_NUM_CTX`) threaded through `OpenAIClient.get_llm()` similarly
   to the existing `_PASSTHROUGH_KWARGS` mechanism in `tradingagents/llm_clients/openai_client.py`, since
   `num_ctx` is Ollama-specific and not a standard `ChatOpenAI` kwarg.
2. **Do not trust `input_tokens` as the oversize signal.** Anomaly A shows the provider-reported figure
   becomes a wrong constant exactly in the size range that matters most (right when a prompt might be
   approaching a context ceiling) — the one place a truncation check would most want to rely on it. Use
   the already-computed, already-tiktoken-backed `prompt_tokens_estimated` (from #147) compared against
   the *now-explicit* `num_ctx` from point 1 instead, since that comparison no longer depends on anything
   Ollama reports back.
3. **Fail loudly (or at minimum warn) before sending an oversize prompt**, per #146's tracking issue title
   — once `num_ctx` is known (point 1) and the estimate is trustworthy (point 2), a simple pre-flight
   check in the LLM call path (or in `LLMCallLogHandler`, which already sees every prompt) can raise/warn
   when `prompt_tokens_estimated` exceeds the configured `num_ctx` (with a safety margin, given the
   ~1.3–1.6x tokenizer-family under-count this investigation measured for Ministral specifically — a
   provider-family-aware margin, not a flat one, would be more accurate).
4. **This diagnosis's evidence does not support trying to make Ollama's `usage.prompt_tokens` accurate.**
   The constant-2051 behavior looks like an Ollama/llama.cpp-internal accounting bug unrelated to this
   codebase's request pattern; chasing it further would need Ollama-side (not TradingAgents-side)
   investigation and is not a good use of #149's scope.

## Cross-reference

`docs/local-models.md` §"Context-Length Knobs" now points here, and its previous "by default, Ollama uses
a context window of 4,096 tokens for all models" claim has been corrected to describe the VRAM-tiered
auto-fit this investigation directly observed.
