# DM-Call for Rust

DM-Call (Decision-Model-Guided Function Calling with Cumulative Verification Feedback) checks a
function call before it runs. An LLM proposes a call. A decision model answers two yes/no
questions about it as probabilities: is this the right function, and are these the right
arguments. A rejected call goes back to the same LLM together with every earlier rejected
candidate and its feedback, until a candidate is accepted or the repair limit is reached.

This crate is a port of the Python reference in `src/dmcall`. It sends the same requests,
byte for byte where it matters: the history message, the two questions and the feedback records
match the code behind the paper's tables, and the crate passes every case in `../spec/conformance`.

## Install

```toml
[dependencies]
dmcall = { git = "https://github.com/gogoymh/decision-model-guided-function-calling" }
serde_json = "1"
```

Cargo finds the crate by its name inside the repository. It needs Rust 1.85 or newer (the version ureq 3 requires).

| Feature | Default | What it does |
|---|---|---|
| `ureq` | on | A blocking HTTP client (ureq with rustls). Without it, pass your own `HttpTransport`. |
| `big-integers` | on | Turns on `serde_json/arbitrary_precision`, so an integer beyond 64 bits keeps every digit in the history text, as in Python. |

`arbitrary_precision` is a global `serde_json` feature: once one crate turns it on, every crate
in the build gets it. If it breaks another dependency (it is known to affect `#[serde(flatten)]`
and untagged enums that contain numbers), use
`dmcall = { git = "...", default-features = false, features = ["ureq"] }`. The only change is
that integers beyond 64 bits are then written as floats (`1e+20`) and `-0` as `-0.0`.

The crate always turns on `serde_json/float_roundtrip`, which makes float parsing exact, so
floats in model output keep the value Python would read.

## Quickstart

```rust,no_run
use dmcall::{DMCall, OpenAI, Tool};
use serde_json::json;

fn main() -> dmcall::Result<()> {
    let weather = Tool::function(
        "get_weather",
        "Current weather for a city.",
        json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
    )?
    .with_handler(|arguments| Ok(json!({"city": arguments["city"], "temp": 21})));

    // OPENAI_API_KEY for the generator; TYPESAFE_API_KEY for the default decision model, Jev.
    let agent = DMCall::builder(OpenAI::new("gpt-5-nano")?).tool(weather).build()?;

    let result = agent.run("Is it warm in Seoul right now?")?;
    println!("{result}"); // CallResult(call=get_weather(city="Seoul"), accepted=true, repairs=0, output=...)
    Ok(())
}
```

`call` proposes, verifies and repairs without running anything. `run` does the same and then
runs the final call with its handler, but only if the call was accepted.

## The API

### `DMCall`

```rust,ignore
let agent = DMCall::builder(generator)       // any Generator, such as OpenAI::new("gpt-5-nano")?
    .tools(vec![weather, clock])             // Tool definitions, each with an optional handler
    .handler("get_time", |args| Ok(json!("09:00")))   // set or replace a handler by name
    .decision_model(Jev::from_env()?)        // Jev from the environment when omitted
    .max_repairs(3)
    .checks(Checks::Both)
    .history(History::Cumulative)
    .feedback(Feedback::Hybrid)
    .on_exhaustion(OnExhaustion::First)
    .preserve_initial_no_call(true)
    .execute_unverified(false)
    .build()?;

let result = agent.call(&messages)?;         // propose, verify, repair; nothing runs
let result = agent.run(&messages)?;          // ...then run an accepted call
let verdict = agent.verify(&messages, &calls)?;   // check calls from any source
```

`messages` can be a `&str` (one user message), a `Vec<Message>` or JSON messages
(`serde_json::Value`, an array of `{"role", "content"}` objects). `call_with` and `run_with`
take `CallOptions` to override the tools, handlers, repair limit, exhaustion rule or
`execute_unverified` for one request. Replacement tools retain registered handlers of the same
name unless the request supplies another. `run` refuses a tool without a handler before sending
any request. Without a decision model, `.threshold(...)` and
`.jev_api_key(...)` configure the default Jev; with one, configure the model itself.

The builder methods take enums. Each also parses from the Python option text, for example
`"latest".parse::<History>()`.

### `CallResult`

| Field or method | Meaning |
|---|---|
| `calls: Option<Vec<Call>>` | The final candidate. `None` is an unreadable output (cut off, filtered, refused or malformed); `Some(vec![])` is a decision to make no call. |
| `accepted: Option<bool>` | Whether it passed every check. `None` for an initial no call returned without verification. |
| `exhausted`, `repairs`, `final_attempt` | No candidate passed within the limit; repairs made; the index of the final candidate in `trace`. |
| `text` | The final candidate's text, if any. |
| `executed`, `outputs`, `output()` | Whether `run` ran the call, and what the handlers returned. |
| `initial_no_call_preserved` | The initial no call was returned without asking the decision model. |
| `trace` | Every candidate, judgment, schema check and feedback record, oldest first. |
| `attempts()` | Each candidate with its verdict, probabilities and schema errors. |
| `probabilities()` | p_f and p_a of the final candidate (`None` when not asked). |
| `usage()` | Token counts summed over every generator and decision-model request. |
| `call()`, `no_call()` | The first call of the final candidate; whether it is a no call. |

### `verify`

`dmcall::verify(&messages, &calls, &tools, &decision_model)` checks calls you got elsewhere.
`messages` may include earlier tool calls and tool results (`tool_calls`, `tool_call_id`);
`calls` may be `Call`s or JSON in the OpenAI, Anthropic or Gemini shapes. Nothing is generated
or run. The `Verdict` has `accepted`, the normalized `calls`, both probabilities, the schema
errors and the feedback record.

### Other functions

- `check_schema(&tools, &calls)`: the local schema check S(c). It checks JSON types, required
  arguments and closed parameter lists, with Python `jsonschema`'s error messages and order.
  Value constraints such as `enum` and ranges are left to the decision model.
- `label(Some(p))`: the verbal band of a probability (`very likely incorrect` to `very likely correct`).
- `normalize_calls(&json)`: calls in any SDK shape as `Vec<Call>`.
- `feedback::represent`, `feedback::single_check`, `prompts::history_message`: the pieces the loop uses.
- `pyjson::dumps`: the Python-compatible JSON text used in the history message.
- Constants: `SYSTEM_PROMPT`, `FUNCTION_QUESTION`, `ARGUMENT_QUESTION`, `prompts::REPAIR_INSTRUCTION`.

### Tools

`Tool::new(json)` accepts an OpenAI (Chat Completions or Responses), Anthropic (`input_schema`)
or Gemini (`parameters` with upper-case types, or `parametersJsonSchema`) definition, and
`Tool::function(name, description, parameters)` builds one directly. The parameters must be a
valid JSON Schema object; references must resolve within the parameters, and invalid references
are refused when the tool is defined. `with_handler` attaches a closure that receives the
arguments object
and returns `Result<Value, BoxError>`.

## Generators

| Type | Key | Notes |
|---|---|---|
| `OpenAI` | `OPENAI_API_KEY` | Chat Completions; base URL from `OPENAI_BASE_URL` if set. |
| `OpenAICompatible` | none | Any Chat Completions server (vLLM, Ollama, LM Studio); needs `.base_url(...)`. |
| `XAI` | `XAI_API_KEY` | Sends `max_tokens`. |
| `OpenRouter` | `OPENROUTER_API_KEY` | Sends `max_tokens`. |
| `Meta` | `MODEL_API_KEY` | Meta Model API (Muse models such as `muse-spark-1.3`), Chat Completions at `https://api.meta.ai/v1`. Sends neither `reasoning_effort` nor `parallel_tool_calls` unless you set them, because Meta rejects `reasoning_effort: "none"` for Muse models. |
| `AzureOpenAI` | `AZURE_OPENAI_API_KEY` | The model name is the deployment; `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`. |
| `Anthropic` | `ANTHROPIC_API_KEY` | Messages API; base URL from `ANTHROPIC_BASE_URL` if set. |
| `Gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `generateContent`; `.thinking_budget(...)`. |

Each type has `new(model)` (everything from the environment) and `builder(model)` with
`api_key`, `base_url`, `max_output_tokens` (4096), `temperature`, `system_prompt`, `timeout`,
`max_retries`, `header`, `transport` and `sleep`; the Chat Completions types add
`reasoning_effort`, `parallel_tool_calls`, `max_tokens_field` and `path`.
`provider_from_spec("anthropic:claude-sonnet-4-5")` builds one from a `"provider:model"` string
(`openai`, `azure`, `anthropic`, `claude`, `gemini`, `google`, `xai`, `grok`, `openrouter`, `meta`),
for example `"meta:muse-spark-1.3"`.

An output that holds no complete proposal (the token limit, a content filter, a refusal or a
malformed call) becomes `calls: None` with the reason in `incomplete`. It is rejected and
repaired, never treated as a decision to make no call. To use another model, implement the
`Generator` trait.

## Decision models

| Type | What it is |
|---|---|
| `Jev` | TypeSafe's hosted decision model, `jev-1.13.0`, the one used in the paper. Key: `TYPESAFE_API_KEY`; `TYPESAFE_MODEL` and `TYPESAFE_BASE_URL` change the model and endpoint. |
| `PPLXDecider` | The open-weight Perplexity Decider v1.1 27B on your own server (`PPLXDecider::new("http://127.0.0.1:18081")`). Plain HTTP only for localhost, no key, proxies ignored, missing usage reported as unknown. `.check_provenance(true)` verifies the pinned checkpoint first. |
| `SystemOne` | Any other server that speaks `POST /v1/systemone`. |

All three take `threshold` (0.5 for both questions), `function_threshold`,
`arguments_threshold`, `timeout`, `max_retries` (3 for Jev, 0 for the PPLX Decider; retried on
408, 429, 500, 502, 503, 504 and 529), `transport` and `sleep`.

To plug in your own, implement `DecisionModel`: `ask` returns the probability of yes for each
question key, and `judge` (provided) runs the schema check, builds the questions and the
feedback record.

```rust,ignore
struct MyModel;

impl DecisionModel for MyModel {
    fn ask(&self, state: &Value, questions: &Map<String, Value>) -> Result<Answer, DecisionModelError> {
        let mut probabilities = HashMap::new();
        for key in questions.keys() {
            probabilities.insert(key.clone(), score(state, key));
        }
        Ok(Answer { probabilities, model: "my-model".into(), ..Answer::default() })
    }

    fn model(&self) -> &str { "my-model" }
    fn function_threshold(&self) -> f64 { 0.6 }
}
```

The decision model receives only the conversation, the function definitions and the candidate.
It never receives a reference answer.

## Options and the paper's variants

| Option | Default (the paper's setting) | Other values |
|---|---|---|
| `checks` | `Checks::Both`: DM-Call | `Checks::Function`: DM-Call-Function; `Checks::Arguments`: DM-Call-Argument |
| `history` | `History::Cumulative`: DM-Call | `History::Latest`: DM-Call-Latest (only the latest rejection, with its real round number) |
| `feedback` | `Feedback::Hybrid` (probability, threshold and label) | `Feedback::Numeric`, `Feedback::Text` |
| `max_repairs` | 3 | any `u32`; r repairs allow at most r + 1 proposals |
| threshold | 0.5 for both questions | set on the decision model |
| `on_exhaustion` | `OnExhaustion::First`: return the first candidate if none passes | `OnExhaustion::Last` |
| `preserve_initial_no_call` | `true`: a readable initial no call is returned without verification | `false` |
| `execute_unverified` | `false`: `run` runs only an accepted call | `true` |

Both questions are sent for every readable candidate with a call. `checks` selects which
judgments determine acceptance and appear in repair feedback; it does not reduce requests or
verification cost. A no call receives only the function question, and a preserved initial
no call receives neither.

The representation changes only what the generator reads. Acceptance always uses the numeric
probabilities, and `trace` keeps them.

## HTTP and testing

The API is blocking. Every request goes through an `HttpTransport`, so a client is easy to test
or to put behind your own HTTP stack. Any closure
`Fn(&HttpRequest) -> Result<HttpResponse, TransportError>` is a transport:

```rust,ignore
let transport = Arc::new(|request: &HttpRequest| {
    Ok(HttpResponse::json(200, &json!({"model": "jev-1.13.0", "answers": {...}, "usage": {...}})))
});
let jev = Jev::builder().api_key("test").transport(transport).build()?;
```

In async code, call the client from `tokio::task::spawn_blocking`. The clients are `Send` and
`Sync`, so one client can serve many threads.

The default HTTP transport never follows redirects.

Keys never appear in errors or `Debug` output. `ProviderError` and `DecisionModelError` keep
only a message, the status code, a sanitized request ID and the number of attempts.

## Differences from the Python package

- Tools are JSON Schema definitions with optional closures; the schema is not inferred from a
  function signature or docstring.
- The trace is typed. A preserved initial no call has `verification: None` instead of a record
  of kind `initial_no_call_preserved`.
- Calls are typed `Call { name, arguments }`. A call in `verify` input without a text name gets
  the name `""` (Python keeps `None`); both fail the schema check the same way.
- `temperature` is an `f64`, so `0` is sent as `0.0`. The two are the same JSON number.
- `serde_json` does not accept `NaN` or `Infinity` in model output, so arguments containing them
  are reported as not valid JSON instead of not finite. Both are rejected and repaired.
- `patternProperties` uses Rust's `regex` syntax, and `pattern` strings are not compiled by the
  schema check. Draft 3 schemas are handled as draft 4.
- The Python `repr` used in error messages treats a few rare unassigned Unicode code points as
  printable.
- There are no async methods; see above.

## Run the tests

```sh
cd rust
cargo test
cargo clippy --all-targets -- -D warnings
```

Provider tests use mock transports; the redirect regression uses only loopback sockets.
`tests/conformance.rs` replays every file in
`../spec/conformance` (`json_text`, `schema_check`, `feedback`, `normalize_calls`, `providers`,
`tools`, `paper_parity` and `loop_edges`) and checks `../spec/prompts.json`. The other test files port
the Python unit tests. `cargo test --no-default-features` checks the build without the HTTP
client and without `big-integers`.
