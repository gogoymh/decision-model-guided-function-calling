# DM-Call specification for every language

The Python package (`src/dmcall`) is the reference implementation. Every other implementation
(`typescript/`, `go/`, `rust/`, `java/`) must behave the same way and pass the conformance files
in this folder. `spec/generate.py` regenerates them from the Python package (except
`conformance/paper_parity.json`, which was recorded once from the code behind the paper's tables).

## What every implementation provides

| Concept | Behavior (Python reference) |
|---|---|
| `DMCall` client | `src/dmcall/core.py`: options `threshold` (0.5, for the default Jev), `max_repairs` (3), `checks` (`both`, `function`, `arguments`), `history` (`cumulative`, `latest`), `feedback` (`hybrid`, `numeric`, `text`), `on_exhaustion` (`first`, `last`), `preserve_initial_no_call` (true), `execute_unverified` (false). Methods `call`, `run` (executes an accepted call with the tool's handler), `verify`. |
| `Result` | calls (`null` = unreadable, `[]` = no call), accepted (`null` for a preserved initial no call), exhausted, repairs, text, executed, outputs, latency, trace, final attempt, initial-no-call flag, plus attempts, probabilities and usage (`generator`, `decision_model`). |
| Decision models | `src/dmcall/decision.py`: `DecisionModel` (POST `{base_url}/v1/systemone`, bounded retries on 408, 429, 500, 502, 503, 504, 529, `judge(messages, tools, calls)` and an overridable `ask(state, questions)`), `Jev` (key from `TYPESAFE_API_KEY`, Bearer auth, model `jev-1.13.0`), `PPLXDecider` (local origin, no key, wire model `jev-latest`, response model must be `perplexity-ai/pplx-decider-v1.1-27b` or `autojev-qwen3.8-27b`, missing usage stays unknown, optional provenance check, HTTP only for localhost). |
| Generators | `src/dmcall/providers.py`: OpenAI (Chat Completions), OpenAI-compatible (any `base_url`, no key needed), xAI and OpenRouter (`max_tokens` field), Azure OpenAI (deployment URL, `api-key` header), Anthropic (Messages), Gemini (`generateContent`). A `"provider:model"` spec string builds one. Unreadable outputs (cut off, filtered, refused, malformed, stopped abnormally) become `calls = null` with a reason, never a no call. Chat Completions arguments may arrive as JSON text, as an object or as null, and the older `function_call` field is read when `tool_calls` is absent. |
| Schema check | `src/dmcall/schema.py`: `check_schema(tools, calls)`, a type-only JSON Schema check with Python jsonschema's error messages for the keywords listed in `conformance/schema_check.json`. |
| Feedback | `src/dmcall/feedback.py`: `label(p)` (five equal bands), `represent` for hybrid, numeric and text, `single_check` for the one-question variants. |
| Prompts | `prompts.json`: the generator system prompt, the history message header and footer, both questions and the repair instruction, character for character. |
| Tools | Tool definitions in OpenAI, Anthropic or Gemini format (lower-case Gemini types), each optionally paired with a handler. `normalize_calls` accepts calls in every SDK shape (`conformance/normalize_calls.json`). |

## Conformance files

| File | What a port checks |
|---|---|
| `prompts.json` | its constants equal these strings and values |
| `conformance/json_text.json` | the history message serializes JSON exactly like Python's `json.dumps(value, ensure_ascii=False)`: `", "` and `": "` separators, insertion key order (also for integer-like keys), floats in Python `repr` form (`1.0`, `1e+16`, `1e-05`), integers without a decimal point, non-ASCII kept, control characters escaped as `\u00XX` except `\t\n\r\b\f`. The parser must keep whether a number was written as an integer or a float. |
| `conformance/schema_check.json` | `check_schema` output for each case equals `expected`, including error order and wording; `configuration_errors` raise |
| `conformance/feedback.json` | labels, representations, single-question records and conflict errors |
| `conformance/normalize_calls.json` | call normalization |
| `conformance/tools.json` | a tool definition in each SDK's format becomes `{name, description, parameters}`; capital type names are lower-cased only where a schema is expected; `configuration_errors` are refused before any request |
| `conformance/providers.json` | request URL, auth headers and body for each generator, decoding of response bodies, and the `redirect` rule |
| `conformance/paper_parity.json` | 14 loop scenarios plus `verify`, recorded from the code behind the paper: replay `generator_responses` and `decision_responses` in order; the request bodies must equal `generator_requests` and `decision_requests`, and the history message text must match character for character |
| `conformance/loop_edges.json` | 7 more scenarios (Gemini, Azure, xAI, an OpenAI-compatible server, the PPLX Decider, text feedback, latest history, one-question checks, unreadable output, Unicode and float arguments); also compares URLs and the full result summary |

Request bodies are compared as JSON values; the history message inside them is a string and must
match exactly. Tests never use the network: inject an HTTP transport and serve recorded bodies.

## Rules

- Never put keys, request bodies or response bodies in errors, logs or string representations, including
  debug formats (`%#v`, `{:?}`, `toString`, `repr`).
- Never follow an HTTP redirect. One request goes to the configured host, and a 3xx answer is an error:
  following it would re-send the body and the key headers to another host.
- A readable response with no function call is a no call; it may contain text. A response that stopped
  abnormally or contains a malformed native call is unreadable (`calls = null`) and is repaired.
- A bad tool (invalid schema, a `$ref` that does not resolve inside its parameters) is refused when the
  client is built. `run` checks that every tool has a handler before it sends the first request.
  A referenced object must occupy a structural schema position, rather than a default value, other
  metadata or a map of named schemas. Boolean reference targets must be valid schemas for the draft.
- Compiled schemas are reused across requests and attempts while they remain in the bounded cache.
- Never send reference answers or labels to a generator or a decision model.
- Defaults equal the paper's settings above. Keep the public names DM-Call, DM-Call-Function
  (`checks = function`), DM-Call-Argument (`checks = arguments`) and DM-Call-Latest (`history = latest`).
