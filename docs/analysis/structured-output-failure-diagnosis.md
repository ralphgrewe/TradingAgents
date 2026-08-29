# Structured-output failure diagnosis: silent `None` from `function_calling` on Ollama

Status: diagnosis report (2026-08-29), issue #163, part of the #159 tracking issue. Follows the
precedent of `docs/analysis/prompt-truncation-diagnosis.md` (issue #148): a standing reference
explaining *why* the code looks the way it does, for the next person who hits a structured-output
failure on a local model. All findings below are measured (live probe against a real local Ollama
daemon, real corpus log evidence) — reproduce them if useful, but they are not re-derived here.

## TL;DR

- Two runs — `ministral-3:8b` via `config-01.json` and `qwen3.5:9b` via `config-01-qwen.json`, both
  against the 26-ticker `stocks.json` list, first ticker `SPCX` — aborted with `PortfolioDecisionError`
  on the very first ticker.
- **Root cause**: LangChain's `function_calling` structured-output method parses with
  `PydanticToolsParser(first_tool_only=True)`, which returns `None` (not an exception) when the model
  emits no tool call. Before issue #160 landed, `run_structured_with_tools` and
  `invoke_structured_or_freetext` only treated a raised exception as failure, so a silent `None` sailed
  through as if the call had succeeded.
- **Trigger, not cause**: prompt size. A live probe (below) shows both models return `None` under
  `function_calling` only once the prompt crosses roughly the real Portfolio Manager prompt's size
  (~29K chars); the same models, same schema, answer correctly at that size under `json_schema`. The
  models are capable — the wrong method was selected for them.
- **Why the wrong method was selected**: `capabilities.py`'s `get_capabilities` resolves by exact model
  ID, then regex, then `_DEFAULT`. No Ollama model ID matches anything but `_DEFAULT`, which sets
  `preferred_structured_method="function_calling"` and `supports_tool_choice=True` — but Ollama's
  OpenAI-compatibility layer does not honor `tool_choice`, so nothing actually forces a tool call.
  LangChain's own default is `json_schema`; this codebase's capability table overrides it.
- **Three fixes landed, in this order**: #160 (treat `None` as failure so the existing #153 retry can
  fire), #161 (make the method configurable and default Ollama to `json_schema` — the structural fix),
  #162 (deterministic text-extraction as a last-resort harness). See "Options considered" below.
- **One known-remaining gap, explicitly out of scope for #159**: `price_target: float | None` — small
  models sometimes emit a non-numeric string (`"N/A"`) for this field, which raises a `ValidationError`
  (not a silent `None`) and correctly triggers the #153 retry today.

## Symptom

Two local runs — `ministral-3:8b` (config `config-01.json`) and `qwen3.5:9b` (`config-01-qwen.json`),
both pointed at the same 26-ticker `stocks.json` (`SPCX` first) — aborted on the first ticker with
`PortfolioDecisionError`. Because `run_trading_agents.py:977` calls `sys.exit(1)` on any per-ticker
exception (see "Correction to CLAUDE.md" below), one Portfolio Manager failure on `SPCX` ended the
entire 26-ticker batch before any other ticker ran.

Earlier in the same runs, both logs also carry a second, more confusing line:

```
Researcher: structured-output invocation failed ('NoneType' object has no attribute 'lean')
```

This reads as if the LLM call itself raised — it did not. `ResearchBrief.lean` (`schemas.py:106`) is
the field `render_research_brief` reads first (`schemas.py:145`: `f"**Recommendation**:
{brief.lean.value}"`). Pre-#160, `invoke_structured_or_freetext` (`tradingagents/agents/utils/
structured.py`) called `return render(result)` unconditionally after `structured_llm.invoke(prompt)`,
with no check that `result` wasn't `None`. When the Researcher's `function_calling` call returned
`None` cleanly (the model wrote prose, no tool call), `render_research_brief(None, ...)` then raised
`AttributeError: 'NoneType' object has no attribute 'lean'` while trying to read `brief.lean.value` —
and that `AttributeError`, one level removed from the real event, is what the broad `except Exception:`
around the invocation logged. The invocation didn't fail; the render did, because nothing had checked
for the silent `None` first. This is the same underlying defect as the Portfolio Manager's
`PortfolioDecisionError`, just caught one call frame later and with a more misleading message. #160's
fix (see below) added the missing `if result is None` check to both `invoke_structured_or_freetext` and
`run_structured_with_tools`, which is why this exact message no longer appears post-fix.

## Root cause

LangChain's `function_calling` structured-output method binds the schema as a tool and parses the
response with `PydanticToolsParser(first_tool_only=True)`:

```python
json_results = super().parse_result(result, partial=partial)
if not json_results:
    return None if self.first_tool_only else []
```

When the model responds with plain prose instead of a tool call — which local models do more often
under long prompts — this parser returns `None` rather than raising. Before #160,
`run_structured_with_tools` (used by the Portfolio Manager and Swing Trader) and
`invoke_structured_or_freetext` (used by the Researcher, Trader, and Research Manager) both treated
*only* a raised exception as a structured-output failure. A silent `None` return therefore looked
identical to success at the point where failure detection happened — the #153 schema-repair retry,
built specifically for this class of weak-model failure, never fired, because nothing signaled that a
retry was needed.

## Log-forensic confirmation

`reports/SPCX_2026-08-29_20260829_165116/llm_calls.jsonl` (the `ministral-3:8b` run) shows the
Portfolio Manager made exactly **two** calls for `SPCX`:

| # | `message_count` | `output_tokens` |
|---|---:|---:|
| 1 | 1 | 1159 |
| 2 | 2 | 51 |

Call 1 is the tool-loop round (`run_structured_with_tools`'s bounded knowledge-base tool loop, which
for the Portfolio Manager runs even with zero tool calls executed — `message_count: 1` is the single
LLM turn before the loop exits with no tool calls requested). Call 2 is the structured-output attempt
itself, feeding the accumulated trace back in (`message_count: 2`) and producing only 51 output
tokens — consistent with a `PydanticToolsParser` call that emits little or no content because the model
wrote prose rather than invoking the schema-as-tool.

The reasoning technique worth reusing: **count the calls, don't just read the error.** Pre-#160, a
*raised* exception on the structured call would have produced a *third* call — the #153 schema-repair
retry, which appends a repair instruction and re-invokes. Two calls, not three, is the log-visible
signature of a silent `None` return slipping past the failure check entirely: the code never knew
anything had gone wrong, so it never retried. This is the kind of thing that's invisible from the error
message alone (there is no error message — that's the whole problem) but immediately visible from the
call count in `llm_calls.jsonl`.

(The `qwen3.5:9b` run's `reports/SPCX_2026-08-29_20260829_170502/llm_calls.jsonl` shows a different
shape for `SPCX` — three Portfolio Manager calls, because qwen3.5:9b's failure mode there is the
`ValidationError` on `price_target` described in "Known-remaining friction" below, which *does* raise
and *does* trigger the retry. Both models still ended up unable to produce a valid decision on that
ticker under `function_calling`, by two different mechanisms.)

## The measured comparison

Reproduced with a live probe against the Ollama daemon, using the real `PortfolioDecision` schema,
`num_ctx=16384`, varying only prompt size and structured-output method:

| Model | Prompt size | `function_calling` | `json_schema` |
|---|---|---|---|
| `ministral-3:8b` | ~200 chars | OK (`Hold`) | OK (`Hold`) |
| `ministral-3:8b` | 29,388 chars | **`None` returned silently** | **OK (`Overweight`)** |
| `qwen3.5:9b` | ~200 chars | `ValidationError` on `price_target` | — |
| `qwen3.5:9b` | 29,388 chars | **`None` returned silently** | **OK (`Buy`)** |

29,388 chars matches the real failing run's Portfolio Manager prompt (29,272 chars, ~7,324 estimated
tokens, per the `llm_calls.jsonl` prompt dump). The conclusion is plain: **prompt size is the trigger,
not the cause.** Both models are fully capable of emitting a valid `PortfolioDecision` at that exact
prompt size — they do it correctly under `json_schema` — so the fix was never "make the prompt smaller
until the model stops failing." The wrong structured-output method was selected for them. This matters
because the obvious-looking fix — shortening the Portfolio Manager prompt, which is already known to be
the largest prompt in the pipeline (`docs/analysis/prompt-size-findings.md`, #144) — would not have
worked, and per `LEARNINGS.md`'s May 1st 2026 entry, this repo has already been burned once by trimming
prompt content that turned out to be load-bearing.

## Why the wrong method was selected

`tradingagents/llm_clients/capabilities.py`'s `get_capabilities` resolves a model's structured-output
capabilities in three steps: exact ID match (`_BY_ID`), then regex match (`_BY_PATTERN`), then
`_DEFAULT`. Every entry in `_BY_ID`/`_BY_PATTERN` is DeepSeek- or MiniMax-specific; no Ollama model ID
(`ministral-3:8b`, `qwen3.5:9b`, or any other) matches either, so every Ollama model falls through to
`_DEFAULT`:

```python
_DEFAULT = ModelCapabilities(
    supports_tool_choice=True,
    supports_json_mode=True,
    supports_json_schema=True,
    preferred_structured_method="function_calling",
)
```

`_DEFAULT` claims `supports_tool_choice=True`, but Ollama's OpenAI-compatible endpoint does not
actually honor a `tool_choice` directive that would force the model to call the schema-as-tool — so
nothing prevents the model from answering in plain prose instead, which is exactly the condition that
makes `PydanticToolsParser(first_tool_only=True)` return `None`. Worth noting explicitly: LangChain's
own library default for `with_structured_output` is `json_schema`, not `function_calling` — this
codebase's capability table is the thing overriding that sensible default, for models (DeepSeek
thinking, MiniMax) where `function_calling` genuinely is the right choice given their own tool-call
quirks. Ollama was simply never given its own entry, so it inherited a default written with a different
provider family in mind.

## Options considered

Three fixes landed for #159, in this deliberate order — cheapest and narrowest first, structural fix
second, last-resort harness third:

1. **Quick fix (#160 — "Handle None returns from structured output as failures").** Treat a silent
   `None` return the same as a raised exception in both `run_structured_with_tools` and
   `invoke_structured_or_freetext`, so the existing #153 schema-repair retry actually fires for this
   failure mode. Cheap (one `if result is None` check per call site) and immediately makes recovery
   *possible* where before it was structurally impossible. Limitation: it does not make recovery
   *likely* — the retry re-sends the same `function_calling` binding that failed the first time, on a
   model that still can't reliably decide to call the schema-as-tool under a long prompt. This fix
   alone would not have saved the `SPCX` run if the retry also came back `None` (a real possibility
   the probe above doesn't rule out for a still-long repair-instruction-augmented prompt).

2. **Structural fix (#161 — "Make the structured-output method configurable and default Ollama to
   `json_schema`").** New config key `structured_output_method` (default `"auto"`), and an
   `OllamaChatOpenAI.with_structured_output` override that resolves `"auto"` to `json_schema` for the
   Ollama provider specifically, ahead of the capability table's `function_calling` default. This is
   the actual fix for the root cause identified above: `json_schema` is grammar-constrained at decode
   time on Ollama's side, so the model is not free to answer in unconstrained prose — the measured
   comparison table shows both models succeed under it at the exact failing prompt size. Ranked above
   #160 in terms of what actually solves the problem, but landed second because #160 is
   provider-agnostic safety net that also protects providers/models not covered by #161's Ollama-specific
   override.

3. **Harness (#162 — "Recover structured decision from free-text fallback via LLM-free text
   extraction").** A final, deterministic (no LLM call) rung on the recovery ladder: if a structured
   call still fails after #160's retry, attempt to parse a `response_model` instance directly out of
   the free-text fallback (bare JSON, fenced ```json``` block, bare fenced block, or the first balanced
   `{…}` object in prose). This is a harness, not a cure — it recovers the answer the model *already
   gave* in prose form, at zero extra cost, for whatever residual failures #161 doesn't already
   eliminate (other providers, or Ollama models running with `structured_output_method` forced away
   from `"auto"`).

## Agentic-pattern context

These three fixes map onto three general approaches to getting reliable structured output out of a
weak or uncooperative model, and the order they were applied in is not incidental:

- **Constrained/grammar-based decoding vs. tool-calling-as-convention.** `json_schema` on Ollama
  works by constraining the token-generation grammar itself — the model is *incapable* of emitting a
  response that doesn't validate against the schema, because non-conforming tokens are never sampled.
  `function_calling`, by contrast, relies on the model *choosing* to emit a tool-call token sequence
  and the server *choosing* to honor a `tool_choice` directive that would force that choice — a
  convention two parties have to cooperate on, not a decoding-time guarantee. Ollama's
  OpenAI-compatibility layer breaks exactly that convention (it accepts `tool_choice` in the request
  without enforcing it), which is why the same model that fails reliably under one method succeeds
  reliably under the other at the same prompt size.
- **Repair/reflection loops.** #153's schema-repair retry (and #160's fix making it actually reachable
  for the `None` case) is a single blind retry: re-send the same request with an explicit
  "your previous reply didn't parse, here's the schema again" instruction, and hope the model does
  better the second time. This is cheap and sometimes sufficient, but it is fundamentally a
  compensating control — it does nothing to prevent the first failure, it only gives the model a second
  try at avoiding it. A more thorough version (validate-and-critique, where the repair instruction
  names *which* field or constraint failed rather than restating the whole schema) was not needed here,
  since #161 addresses the underlying cause directly.
- **Deterministic extraction passes.** #162's text extraction is the cheapest possible recovery step
  precisely because it makes no further model call at all — it treats the free-text fallback as data to
  be mined, not a failure to be retried. It is the right *last* rung specifically because it can only
  recover an answer the model already produced; it cannot make a model produce a better answer, so it's
  no substitute for constraining generation in the first place.

The ordering principle the evidence here supports: **constrain generation first, repair second, extract
third.** Repair and extraction are compensating controls for a generation step that was never
constrained to begin with — they can recover value the first two layers below them missed, but they
can't manufacture correctness a properly-constrained decode would have guaranteed from the start. #161
(constrain) is doing the real work in this specific failure; #160 (repair) and #162 (extract) are
there for whatever #161 doesn't reach — other providers, or a user who explicitly forces
`structured_output_method` away from `"auto"`.

## Known-remaining friction

`PortfolioDecision.price_target: float | None` (`tradingagents/agents/schemas.py:456`) draws
non-numeric strings — `"N/A"`, `"$145.20"` — from small models often enough to be worth recording.
Pydantic raises a `ValidationError` when the model's tool-call arguments include a string where a
`float | None` is expected. Unlike the silent-`None` failure mode this document is centrally about,
this *does* raise, so the #153/#160 retry already fires correctly for it today — it's a working failure
path, not a silent one. It is recorded here as a known issue and explicitly **not** fixed as part of
#159: relaxing or coercing the `price_target` type is a schema change, and schema changes were ruled
out of scope for #159. A future fix (e.g. accepting a string and coercing/rejecting non-numeric values
with a Pydantic validator) is a candidate follow-up, not attempted here.

## Cross-references

- #159 — the tracking issue this diagnosis, and #160/#161/#162, are all part of.
- #160 — `Handle None returns from structured output as failures` (quick fix, Option 1 above).
- #161 — `Make the structured-output method configurable and default Ollama to json_schema` (structural
  fix, Option 2 above).
- #162 — `Recover structured decision from free-text fallback via LLM-free text extraction` (harness,
  Option 3 above).
- #156 — `Abort ticker on Portfolio Manager structured decision failure`, the hard-fail behavior
  (`PortfolioDecisionError`) that surfaced this failure mode as a batch-ending abort rather than a
  silently-wrong decision. See CLAUDE.md's corrected "Portfolio Manager structured decision requirement
  (issue #156)" section for the accurate description of what happens when it fires.
- `docs/analysis/prompt-truncation-diagnosis.md` (#148) — the sibling diagnosis this document's format
  and tone follow; a different local-model failure mode (silent prompt truncation) investigated with
  the same live-probe-plus-corpus-forensics method.
- `LEARNINGS.md` (May 1st, 2026 entry) — the prior finding that shortening prompts made this system
  *more* unstable, cited above as the reason "just shorten the Portfolio Manager prompt" was never a
  credible fix for this failure.
