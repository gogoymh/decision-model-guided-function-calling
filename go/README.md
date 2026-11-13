# DM-Call for Go

This is the Go port of `dmcall`, the library behind the paper **[DM-Call: Decision-Model-Guided Function Calling with Cumulative Verification Feedback](https://arxiv.org/abs/2611.10977)**. The Python package in `src/dmcall` is the reference implementation, and this port sends the same requests: the conformance files in `../spec` are replayed in its tests, including the request bodies recorded from the code behind the paper's tables.

An LLM proposes a function call. A decision model answers two yes/no questions about it, as probabilities of yes: is this the right function, and are these the right arguments? A rejected call goes back to the same LLM together with every earlier rejected candidate and its feedback, until a candidate is accepted or the repair limit is reached. Nothing runs before a call is accepted.

The module uses only the Go standard library and needs Go 1.22 or newer.

## Install

```bash
go get github.com/gogoymh/decision-model-guided-function-calling/go
```

The package name is `dmcall`:

```go
import dmcall "github.com/gogoymh/decision-model-guided-function-calling/go"
```

Set the keys for the decision model and for your model provider:

```bash
export TYPESAFE_API_KEY=...   # Jev, the default decision model
export OPENAI_API_KEY=...     # or ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, MODEL_API_KEY (Meta), ...
```

## Quickstart

```go
package main

import (
	"context"
	"fmt"
	"log"

	dmcall "github.com/gogoymh/decision-model-guided-function-calling/go"
)

type weatherArgs struct {
	City string `json:"city"`
	Unit string `json:"unit"`
}

func main() {
	weather := dmcall.Tool{
		Name:        "get_weather",
		Description: "Current weather for a city.",
		Parameters: dmcall.MustParseJSON(`{"type": "object", "properties": {
			"city": {"type": "string", "description": "City name, e.g. \"Seoul\"."},
			"unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}},
			"required": ["city"]}`),
		Handler: dmcall.TypedHandler(func(ctx context.Context, in weatherArgs) (map[string]any, error) {
			return map[string]any{"city": in.City, "temp": 12}, nil
		}),
	}

	generator, err := dmcall.NewOpenAI("gpt-5-nano", dmcall.WithReasoningEffort("none"))
	if err != nil {
		log.Fatal(err)
	}
	client, err := dmcall.New(generator, dmcall.WithTools(weather))
	if err != nil {
		log.Fatal(err)
	}

	result, err := client.Run(context.Background(), dmcall.Prompt("Do I need a jacket in Seoul tonight?"))
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)           // Result(call=get_weather(city="Seoul"), accepted=True, repairs=0, output=...)
	fmt.Println(result.Call())    // the accepted call
	fmt.Println(result.Output())  // what the handler returned
	fmt.Println(result.Repairs)   // repairs it took (0 to 3)
}
```

`Run` executes the final call only if it was accepted. `Call` does everything except execute, so you decide:

```go
result, err := client.Call(ctx, dmcall.Prompt("Book a table for two at 7pm"))
switch {
case err != nil:
	// a transport or protocol failure, never a rejection
case result.Accepted != nil && *result.Accepted:
	// run result.Calls yourself
case result.NoCall():
	fmt.Println(result.Text) // the model answered without a tool
}
```

## How the loop works

1. **Propose.** The generator proposes a candidate: one call, or no call.
2. **Verify.** A local schema check tests that the candidate is readable, names an available function and supplies every required argument with its declared JSON type. The decision model then answers the function question and, when there is a call, the argument question.
3. **Accept.** A candidate is accepted when the schema check passes and every question asked scores at least the threshold (0.5). A readable candidate that fails the schema check is still judged, so its feedback carries both the schema errors and the scores.
4. **Repair.** A rejected candidate and its feedback join the history. The generator reads that history, oldest first and marked as not executed, in one message before the unchanged original request. The decision model sees only the request, the function schemas and the current candidate.
5. **Return.** The loop returns the first accepted candidate. If none is accepted within the repair limit, it returns the first candidate. A readable initial no call is returned without verification and marked as unverified (`Accepted` is nil). An unreadable output (cut off, filtered, refused or malformed) is never treated as a no call.

## API

### Client

| Function | What it does |
|---|---|
| `dmcall.New(generator, opts...)` | builds a client; without `WithDecisionModel` it uses Jev from `TYPESAFE_API_KEY` |
| `client.Call(ctx, messages, opts...)` | proposes, verifies and repairs; nothing is executed |
| `client.Run(ctx, messages, opts...)` | `Call`, then executes an accepted call with its tool's `Handler` |
| `client.Verify(ctx, messages, calls, opts...)` | checks calls from any source with the schema check and both questions |
| `dmcall.Verify(ctx, decisionModel, messages, calls, tools)` | the same without a client; a nil decision model means Jev from the environment |

`messages` is a `[]dmcall.Message` with `Role` and `Content`. `dmcall.Prompt(text)` makes a one-message conversation, and `dmcall.ParseMessages` reads a conversation in OpenAI's chat format, including earlier tool calls and tool results for `Verify`.

`Call`, `Run` and `Verify` stop when the context is done. If a tool's handler fails, `Run` returns the result together with the error.

### Result

| Field or method | Meaning |
|---|---|
| `Calls` | the final candidate: an empty slice for no call, nil if it was unreadable |
| `Call()` | the first call, or nil |
| `NoCall()` | the final candidate is a decision to make no call |
| `Accepted` | `*bool`: passed every check; nil for an initial no call returned without verification |
| `InitialNoCallPreserved` | the first candidate was a readable no call, returned without verification |
| `Exhausted` | the repair limit was reached without an accepted candidate |
| `Repairs` | repairs used |
| `FinalAttempt` | which entry of `Trace` is final (0 is the first candidate) |
| `Text` | the model's text, for example its answer when it made no call |
| `Executed`, `Outputs`, `Output()` | whether `Run` executed the call, and what the handlers returned |
| `Attempts()` | every candidate with its verdict, scores and schema errors |
| `Probabilities()` | the decision model's scores for the final candidate |
| `Usage()` | token counts of the generator and of the decision model |
| `Latency`, `Trace` | wall time, and the full record of candidates, judgments and feedback |

### Tools and calls

A `dmcall.Tool` has a `Name`, a `Description`, a JSON Schema in `Parameters` and an optional `Handler`. `Parameters` accepts an `*Object` (from `MustParseJSON`, which keeps the key order you wrote), a `map[string]any`, a `json.RawMessage` or any value `encoding/json` can marshal.

- `dmcall.ParseTool(def)` and `dmcall.ParseTools(data)` read definitions in OpenAI, Anthropic (`input_schema`) or Gemini format (`parameters` or `parametersJsonSchema`, upper-case types allowed).
- `dmcall.TypedHandler(fn)` decodes the call's arguments into a struct before calling `fn`.
- `dmcall.WithHandlers(map[string]Handler{...})` adds handlers to tools defined without one.
- `dmcall.NormalizeCalls(calls)` reads calls in every SDK shape: DM-Call's `{"name", "arguments"}`, OpenAI Chat Completions and Responses, Anthropic `tool_use` blocks, Gemini `functionCall` parts, `[]dmcall.ToolCall` or JSON text.
- `dmcall.CheckSchema(tools, calls)` runs the schema check alone, with the error messages of Python's `jsonschema`.

### Generators

| Provider | Constructor | Spec for `ProviderFromSpec` | Key |
|---|---|---|---|
| OpenAI | `NewOpenAI(model, ...)` | `openai:<model>` | `OPENAI_API_KEY` (base URL: `OPENAI_BASE_URL`) |
| Anthropic | `NewAnthropic(model, ...)` | `anthropic:<model>` or `claude:<model>` | `ANTHROPIC_API_KEY` (base URL: `ANTHROPIC_BASE_URL`) |
| Google Gemini | `NewGemini(model, ...)` | `gemini:<model>` or `google:<model>` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| xAI Grok | `NewXAI(model, ...)` | `xai:<model>` or `grok:<model>` | `XAI_API_KEY` |
| OpenRouter | `NewOpenRouter(model, ...)` | `openrouter:<model>` | `OPENROUTER_API_KEY` |
| Meta Model API (Muse) | `NewMeta(model, ...)` | `meta:<model>`, e.g. `meta:muse-spark-1.3` | `MODEL_API_KEY` |
| Azure OpenAI | `NewAzureOpenAI(deployment, ...)` | `azure:<deployment>` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` |
| Any OpenAI-compatible server | `NewOpenAICompatible(model, WithBaseURL(...))` | | optional |

Provider options: `WithAPIKey`, `WithBaseURL`, `WithHTTPClient`, `WithTimeout` (120 s per attempt), `WithMaxRetries` (3), `WithMaxOutputTokens` (4096; 0 sends no limit), `WithTemperature`, `WithSystemPrompt`, `WithHeaders`. OpenAI-compatible providers also take `WithReasoningEffort`, `WithParallelToolCalls` (false by default) or `WithoutParallelToolCalls`, `WithMaxTokensField` and `WithPath`. Azure takes `WithEndpoint` and `WithAPIVersion`, and Gemini takes `WithThinkingBudget`.

Muse models on the Meta Model API are reasoning models, and Meta rejects `reasoning_effort: "none"` for them. `NewMeta` therefore sends neither `reasoning_effort` nor `parallel_tool_calls` unless you set them with `WithReasoningEffort` or `WithParallelToolCalls`.

A response that is cut off, filtered or refused becomes an unreadable candidate: it is rejected and repaired, never read as a no call. HTTP failures return a `*dmcall.ProviderError`, which holds the status, a request ID and the provider's error code but never a key or a response body. Any type with `Generate(ctx, example, history)` can stand in for a provider; `dmcall.MessagesWithHistory` builds the conversation a generator should send, and `dmcall.SystemPrompt` is the generator prompt used in the paper.

### Decision models

```go
jev, _ := dmcall.NewJev(dmcall.WithThreshold(0.5))                 // TYPESAFE_API_KEY
pplx, _ := dmcall.NewPPLXDecider("http://localhost:8000")          // your own server
client, _ := dmcall.New(generator, dmcall.WithDecisionModel(pplx))
```

- **`NewJev`** is TypeSafe's hosted decision model. The paper uses `jev-1.13.0`, the default (`TYPESAFE_MODEL` or `WithModel` changes it; `TYPESAFE_BASE_URL` or `WithBaseURL` changes the endpoint). `WithThreshold` sets both questions, and `WithFunctionThreshold` or `WithArgumentsThreshold` overrides one.
- **`NewPPLXDecider`** is the open-weight Perplexity Decider v1.1 27B (`perplexity-ai/pplx-decider-v1.1-27b`, revision `14c9d90137962d300f3de95633f7b6535e80c5ce`), served through the checkpoint's own native inference server, which answers `POST /v1/systemone`. A generic chat server is not this model. Plain HTTP is accepted only for localhost. `WithCheckProvenance(true)` makes the client confirm the pinned checkpoint through `GET /provenance` first. The default timeout is 180 seconds with no retries.
- **`NewSystemOne(baseURL, model)`** reaches any other server that speaks the same protocol.
- **Your own decision model**: implement `dmcall.DecisionModel` (one method, `Judge`), or keep DM-Call's schema check, thresholds and feedback record and supply only the questions' answers with `dmcall.NewDecisionModel(name, source, asker)`, where `asker` is a `dmcall.Asker` or an `AskerFunc`. The questions are `dmcall.FunctionQuestion` and `dmcall.ArgumentQuestion`.

The decision models receive only the conversation, the function schemas and the candidate, never a reference answer. Failures return a `*dmcall.DecisionModelError` with a code such as `http_error`, `connection_failed`, `response_invalid` or `model_mismatch`; a failure is never a rejection.

### Feedback and JSON values

`dmcall.Label(p)` returns the verbal band of a probability. `dmcall.Represent(feedback, representation)` gives the generator's copy of a feedback record, and `dmcall.SingleCheck` removes the other question's verdict for variants that use one verdict.

The history message is JSON text inside a string, and it must equal what Python's `json.dumps(value, ensure_ascii=False)` writes. Go maps lose key order and `encoding/json` loses the difference between `4.0` and `4`, so DM-Call has its own JSON value: `dmcall.ParseJSON` returns `nil`, `bool`, `string`, `int64` (or `*big.Int`), `float64`, `[]any` or `*dmcall.Object`, an object that keeps its key order. `dmcall.JSONText` writes Python's text, and `dmcall.Equal` compares values the way Python compares parsed JSON. `*Object` implements `json.Marshaler` and `json.Unmarshaler`, and `obj.Decode(&v)` fills a struct.

## Options and the paper's variants

| Option | Default | Meaning |
|---|---|---|
| `WithDecisionModel(dm)` | Jev | the decision model |
| `WithThreshold(t)` | 0.5 | pass mark of both questions, for the default Jev |
| `WithJevAPIKey(key)` | `TYPESAFE_API_KEY` | key of the default Jev |
| `WithMaxRepairs(n)` | 3 | repairs after the first candidate |
| `WithChecks(c)` | `ChecksBoth` | `ChecksFunction` or `ChecksArguments` selects the verdict used for acceptance and repair feedback; both questions are sent for a readable call |
| `WithHistory(h)` | `HistoryCumulative` | `HistoryLatest` shows only the latest rejected candidate |
| `WithFeedback(f)` | `FeedbackHybrid` | `FeedbackNumeric` or `FeedbackText` |
| `WithOnExhaustion(r)` | `ExhaustionFirst` | `ExhaustionLast` returns the last repair when nothing is accepted |
| `WithPreserveInitialNoCall(b)` | true | return a readable first no call without verification |
| `WithExecuteUnverified(b)` | false | `Run` also executes the final candidate when none was accepted |
| `WithTools(tools...)`, `WithHandlers(m)` | none | the available tools and their handlers |

Both questions are sent for a readable call even when only one verdict decides acceptance and appears in repair feedback. A no-call receives only the function question when verification applies; a preserved initial no-call and an unreadable candidate receive neither.

The defaults are the paper's settings. `WithTools`, `WithHandlers`, `WithMaxRepairs`, `WithOnExhaustion` and `WithExecuteUnverified` can also be passed to a single `Call` or `Run`.

| In the paper | Options |
|---|---|
| DM-Call | defaults |
| DM-Call-Function: use the function verdict | `WithChecks(dmcall.ChecksFunction)` |
| DM-Call-Argument: use the argument verdict | `WithChecks(dmcall.ChecksArguments)` |
| DM-Call-Latest: only the latest rejection as feedback | `WithHistory(dmcall.HistoryLatest)` |
| Numeric feedback | `WithFeedback(dmcall.FeedbackNumeric)` |
| Text feedback | `WithFeedback(dmcall.FeedbackText)` |
| Hybrid feedback | `WithFeedback(dmcall.FeedbackHybrid)` |

## Tests

```bash
cd go
go test ./...
```

The tests never use the network. Every provider and decision model takes an `*http.Client` through `WithHTTPClient`, and the tests pass one whose `RoundTripper` serves recorded or scripted bodies. They replay every conformance file in `../spec`:

- `json_text.json`: the history message's JSON text
- `schema_check.json`: schema check results, with exact error messages and order, and the configuration errors
- `feedback.json`, `normalize_calls.json`: labels, representations, one-question records and call shapes
- `providers.json`: request URL, auth headers and body for each generator, and the decoding of responses
- `paper_parity.json`: 14 loop scenarios and `verify`, recorded from the code behind the paper; request bodies must equal the recorded ones, also as text
- `loop_edges.json`: 7 more scenarios, compared by URL, headers, body and the full result summary

The remaining tests port the Python unit tests in `tests/`.

## Differences from the Python package

- Tools are JSON Schema definitions. Go cannot read parameter names from a function, so the schema is not inferred from a signature as Python's `schema_of` does.
- `FunctionQuestion` and `ArgumentQuestion` are variables of type `Question`, because Go has no struct constants. `SystemPrompt` is a constant.
- `NormalizeCalls` returns `[]Call`. A call whose name is not a string gets an empty name, and arguments that are not a JSON object become nil. The schema check reports the same errors for these as Python does; only the echoed value differs.
- Message content is text. `Verify` leaves out the content of an assistant turn that only calls tools, as Python does for `content: null`.
- A token count a service did not report is absent from `Usage`; Python keeps it as `None`.
- JSON Schema patterns use Go's RE2 syntax, and drafts before 2020-12 are approximated (draft-03 is read as draft-04). References must point inside the schema (`#/...`).
- There are no `acall`, `arun` or `averify`; call `Call`, `Run` and `Verify` from a goroutine instead. There is no `close`: the HTTP clients need no closing.
- The per-attempt timeout covers the whole attempt, where httpx applies it to each connect, read and write.
