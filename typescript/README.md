# dmcall for TypeScript

DM-Call (Decision-Model-Guided Function Calling with Cumulative Verification Feedback) lets an LLM propose a function call and checks it before anything runs. A decision model answers two yes/no questions about the call as probabilities: is this the right function, and are the arguments right? A rejected call goes back to the same LLM together with every earlier rejected candidate and its feedback, until a candidate is accepted or the repair limit is reached.

This package is the TypeScript port of the Python reference in [`src/dmcall`](../src/dmcall). It sends the same requests byte for byte where it matters (prompts, questions, the feedback records and the history message) and passes every case in the shared conformance files in [`spec/`](../spec). It has no runtime dependencies and uses the global `fetch`.

## Install

The package is not on npm yet. Build it from a clone of the repository (Node 18 or newer):

```bash
git clone https://github.com/gogoymh/decision-model-guided-function-calling.git
cd decision-model-guided-function-calling/typescript
npm install
npm pack            # builds dist/ and writes dmcall-1.0.0.tgz
```

Then install the tarball in your project:

```bash
npm install /path/to/decision-model-guided-function-calling/typescript/dmcall-1.0.0.tgz
```

`npm install /path/to/.../typescript` also works after `npm run build`. The package is ESM only and ships its own type declarations.

Set the keys for the decision model and for your model provider:

```bash
export TYPESAFE_API_KEY=...   # Jev, the default decision model (https://console.typesafe.ai)
export OPENAI_API_KEY=...     # or ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, ...
```

## Quickstart

```ts
import { DMCall, tool } from "dmcall";

const getWeather = tool({
  name: "get_weather",
  description: "Current weather for a city.",
  parameters: {
    type: "object",
    properties: {
      city: { type: "string", description: 'City name, e.g. "Seoul".' },
      unit: { type: "string", enum: ["celsius", "fahrenheit"], default: "celsius" },
    },
    required: ["city"],
  },
  execute: async ({ city, unit = "celsius" }) => ({ city, temp: 12, unit }),
});

const agent = new DMCall("openai:gpt-5-nano", { tools: [getWeather], reasoningEffort: "none" });

const result = await agent.run("Do I need a jacket in Seoul tonight?");
result.call;      // { name: "get_weather", arguments: { city: "Seoul" } }
result.accepted;  // true: the schema check and both questions passed
result.output;    // { city: "Seoul", temp: 12, unit: "celsius" }
result.repairs;   // repairs it took (0 to 3)
```

`run()` executes the final call only if it was accepted. `call()` does everything except execute, so you decide:

```ts
const result = await agent.call("Book a table for two at 7pm");
if (result.accepted) {
  // run result.call yourself
} else if (result.noCall) {
  console.log(result.text); // the model answered without a tool
}
```

## Use any model

The generator is a `"provider:model"` string or a provider object. Options that DM-Call does not use itself (`apiKey`, `baseUrl`, `reasoningEffort`, ...) go to the provider.

```ts
import { DMCall, OpenAICompatible } from "dmcall";

new DMCall("openai:gpt-5-nano", { reasoningEffort: "none" });  // OPENAI_API_KEY
new DMCall("anthropic:claude-sonnet-5-5");                     // ANTHROPIC_API_KEY
new DMCall("gemini:gemini-2.5-flash");                         // GEMINI_API_KEY or GOOGLE_API_KEY
new DMCall("xai:grok-4");                                      // XAI_API_KEY
new DMCall("openrouter:meta-llama/llama-3.3-70b-instruct");    // OPENROUTER_API_KEY
new DMCall("meta:muse-spark-1.3");                             // MODEL_API_KEY (Meta Model API)
new DMCall("azure:my-deployment", { endpoint: "https://my-resource.openai.azure.com" }); // AZURE_OPENAI_API_KEY

// A local or self-hosted model behind an OpenAI-compatible server (vLLM, Ollama, LM Studio, ...)
new DMCall(new OpenAICompatible("gemma-4-31b-it", { baseUrl: "http://localhost:8000/v1" }));
```

| Provider | Spec | Key | Class |
|---|---|---|---|
| OpenAI | `openai:<model>` | `OPENAI_API_KEY` | `OpenAI` |
| Anthropic | `anthropic:<model>` | `ANTHROPIC_API_KEY` | `Anthropic` |
| Google Gemini | `gemini:<model>` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `Gemini` |
| xAI Grok | `xai:<model>` | `XAI_API_KEY` | `XAI` |
| Azure OpenAI | `azure:<deployment>` | `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT` | `AzureOpenAI` |
| OpenRouter | `openrouter:<model>` | `OPENROUTER_API_KEY` | `OpenRouter` |
| Meta Model API (Muse) | `meta:<model>` | `MODEL_API_KEY` | `Meta` |
| Any OpenAI-compatible server | | optional | `OpenAICompatible` |

Every provider takes `apiKey`, `baseUrl`, `maxOutputTokens` (4,096 by default), `temperature`, `systemPrompt`, `timeoutMs`, `maxRetries`, `headers`, `fetch` and `sleep`. OpenAI-style providers also take `reasoningEffort`, `parallelToolCalls` (false by default), `maxTokensField` and `path`. Azure takes `endpoint` (in place of `baseUrl`) and `apiVersion`, and Gemini takes `thinkingBudget`. `Meta` sends neither `reasoning_effort` nor `parallel_tool_calls` unless you set them, because the Meta Model API rejects `reasoning_effort: "none"` for Muse models. An option a provider does not know throws a `TypeError`, so a misspelled option fails early. A response that is cut off, filtered or refused is an unreadable candidate (`calls: null`): it is rejected and repaired, never read as "no call". `SYSTEM_PROMPT` is the generator prompt used in the paper.

Any object with `generate(example, history)` that returns `{ calls, content, parseErrors, incomplete }` can stand in for a provider.

## Decision models

The decision model answers the two questions. Pass a configured one as `decisionModel`.

```ts
import { DMCall, Jev, PPLXDecider } from "dmcall";

new DMCall("openai:gpt-5-nano"); // Jev with threshold 0.5 (TYPESAFE_API_KEY)

const jev = new Jev("...", { threshold: 0.5, model: "jev-1.13.0" });
new DMCall("openai:gpt-5-nano", { decisionModel: jev });

const pplx = new PPLXDecider("http://localhost:8000");
new DMCall("openai:gpt-5-nano", { decisionModel: pplx });
```

- **`Jev`** is TypeSafe's hosted decision model; the paper uses `jev-1.13.0`. The key comes from the first argument or `TYPESAFE_API_KEY`, the model from `model` or `TYPESAFE_MODEL`, and the endpoint from `baseUrl` or `TYPESAFE_BASE_URL`. `threshold` sets both questions, and `functionThreshold` or `argumentsThreshold` overrides one.
- **`PPLXDecider`** is the open-weight [Perplexity Decider v1.1 27B](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b), served on your own machine through the checkpoint's native inference server, which answers `POST /v1/systemone`. A generic causal chat server is not this model. Plain HTTP is accepted only for `localhost`; use HTTPS or an SSH tunnel otherwise. `checkProvenance: true` makes the client confirm the pinned checkpoint through `GET /provenance` first.
- **Your own decision model**: extend `DecisionModel` and override `ask(state, questions)` to return a probability of yes for each question. The questions are `FUNCTION_QUESTION` and `ARGUMENT_QUESTION`. Any object with `judge(messages, tools, calls)` also works.

Both decision models receive only the conversation, the function schemas and the candidate, never a reference answer. HTTP failures and malformed responses throw `DecisionModelError`; they are never treated as a rejection.

## Tools

A tool is a JSON Schema definition, optionally paired with the code that runs it:

```ts
import { tool } from "dmcall";

const search = tool({
  name: "search",
  description: "Search the web.",
  parameters: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
  execute: async ({ query }) => mySearch(query as string),
});
```

Plain definitions in OpenAI (Chat Completions or Responses), Anthropic (`input_schema`) or Gemini shape (upper-case types or `parametersJsonSchema`) work too. Give their handlers by name:

```ts
const schema = { type: "function", function: { name: "lookup", description: "Find a record.",
  parameters: { type: "object", properties: { id: { type: "string" } }, required: ["id"] } } };
const agent = new DMCall("openai:gpt-5-nano", { tools: [search, schema], functions: { lookup: myLookup } });
```

A handler receives the arguments as one object and may be async. `schemaOf(tool)` shows the definition DM-Call sends, and `checkSchema(tools, calls)` runs the schema check S on its own.

## Verify calls from any framework

Keep your agent and check its calls before you execute them. `verify` takes the conversation so far (earlier tool calls and tool results included) and the proposed calls, as plain objects or in the shape the OpenAI, Anthropic and Google SDKs return.

```ts
import OpenAI from "openai";
import { verify } from "dmcall";

const response = await new OpenAI().chat.completions.create({ model: "gpt-5-nano", messages, tools });
const calls = response.choices[0].message.tool_calls;
const verdict = await verify(messages, calls, tools);

if (verdict.accepted) {
  // passed the schema check and both questions: execute the calls
} else {
  console.log(verdict.functionProbability, verdict.argumentsProbability, verdict.schemaErrors);
  // verdict.feedback is the record DM-Call would send back to the model for a repair
}
```

## Options

```ts
new DMCall("openai:gpt-5-nano", {
  tools: [...],
  decisionModel: undefined,       // Jev by default
  threshold: 0.5,                 // pass mark of both questions (for the default Jev)
  maxRepairs: 3,                  // r: repairs after the first candidate
  checks: "both",                 // "function" or "arguments": ask one question only
  history: "cumulative",          // "latest": show only the latest rejected candidate
  feedback: "hybrid",             // "numeric" or "text"
  onExhaustion: "first",          // if nothing is accepted: return c_0 ("last": the last repair)
  preserveInitialNoCall: true,    // return a readable first no call without verification
  executeUnverified: false,       // true: run() also executes the final candidate when none was accepted
});
```

The defaults are the paper's settings. The paper's variants map to options as follows.

| In the paper | Options |
|---|---|
| DM-Call | defaults |
| DM-Call-Function: function question only | `checks: "function"` |
| DM-Call-Argument: argument question only | `checks: "arguments"` |
| DM-Call-Latest: only the latest rejection as feedback | `history: "latest"` |
| *Numeric* feedback, Φ(p) = p | `feedback: "numeric"` |
| *Text* feedback, Φ(p) = ℓ(p) | `feedback: "text"` |
| *Hybrid* feedback, Φ(p) = (p, ℓ(p)) | `feedback: "hybrid"` |

`call()` and `run()` also take `tools`, `functions`, `maxRepairs` and `onExhaustion` per request, and `run()` takes `executeUnverified`. A `fetch` option on `DMCall` is passed to the provider it builds from a spec string and to the default Jev.

### The result

| Field | Meaning |
|---|---|
| `calls` | final candidate: `[{ name, arguments }]`, `[]` for no call, `null` if unreadable; c_0 when nothing was accepted (`onExhaustion: "first"`) |
| `call` | the first call, or `null` |
| `accepted` | passed every check (`null` for a first no call returned without verification) |
| `initialNoCallPreserved` | the first candidate was a readable no call, returned without verification |
| `exhausted` | reached the repair limit without an accepted candidate |
| `repairs` | repairs used |
| `finalAttempt` | which candidate in `trace` is final (0: the first) |
| `text` | the model's text, e.g. its answer when it made no call |
| `executed`, `outputs`, `output` | whether `run()` executed the call, and what it returned |
| `attempts` | every candidate with its verdict, scores and schema errors |
| `probabilities` | the decision model's scores for the final candidate |
| `usage`, `latencyMs` | tokens of the generator and of the decision model, and wall time |
| `trace` | the full record: candidates, judgments and the feedback the generator was given |

## Python-compatible JSON

The history message is JSON text inside a string, and it must equal what Python's `json.dumps(value, ensure_ascii=False)` writes. `JSON.parse` loses whether a number was written as `4.0` or `4` and the order of integer-like keys such as `"10"` and `"2"`, so the library parses model output with its own parser and keeps both. The helpers are available as `dmcall/json` (`parse`, `stringify`, `fromPlain`, `toPlain`, `pythonRepr`). The public API takes and returns plain objects; where a value must keep a float such as `4.0` exactly, pass it as parsed by `dmcall/json`.

## Differences from the Python package

- Methods are async, and options and result fields use camelCase (`maxRepairs`, `finalAttempt`, `latencyMs`). The feedback records the generator reads keep their snake_case field names, because they are part of the protocol.
- Tools are JSON Schema definitions. There is no schema inference from function signatures; use `tool({ name, description, parameters, execute })`. Handlers receive one arguments object and may be async.
- Timeouts are in milliseconds (`timeoutMs`), and HTTP goes through an injectable `fetch` instead of an `httpx.Client`. There is nothing to close, so there is no `close()`.
- The schema check emulates Python jsonschema for the keywords DM-Call checks, including draft-04 to draft-07 `$schema` dialects. Patterns run as JavaScript regular expressions (Python's named groups and leading `(?i)`, `(?m)`, `(?s)` flags are translated), and `$ref` must point inside the same schema.
- A whole JavaScript number in a plain object is sent as an integer (`4`, not `4.0`); see the JSON section above.

## Tests

```bash
cd typescript
npm install
npm test
```

`npm test` compiles the sources and tests with `tsc` and runs them with `node --test`. The tests never use the network. They replay every case in `../spec/conformance/*.json` and check the constants against `../spec/prompts.json`, and they port the Python unit tests for the loop, the providers, the decision models, the schema check, the feedback representations and the tools.
