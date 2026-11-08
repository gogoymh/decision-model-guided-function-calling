# DM-Call for Java

DM-Call (Decision-Model-Guided Function Calling with Cumulative Verification Feedback) lets an LLM
propose a function call and has a decision model check it before anything runs. The decision model
answers two yes/no questions as probabilities: is this the right function, and are these the right
arguments. A rejected call goes back to the same LLM together with every earlier rejected candidate
and its feedback, until a candidate is accepted or the repair limit is reached.

This is the Java port of the Python package in `src/dmcall`. It passes every case in
`spec/conformance`: for each recorded scenario, the request bodies it sends are byte for byte the
bodies the Python reference sends, including the history message the generator reads.

## Build and install from the repository

The library is not on Maven Central yet. You need JDK 17 or newer and Maven 3.6.3 or newer (the
build was tested with JDK 21 and 27 and Maven 3.10).

```sh
cd java
mvn install
```

`mvn install` runs the tests and puts the jar in your local Maven repository. Then add the
dependency to your project:

```xml
<dependency>
  <groupId>io.github.gogoymh</groupId>
  <artifactId>dmcall</artifactId>
  <version>1.0.0</version>
</dependency>
```

For Gradle, add `mavenLocal()` to your repositories and
`implementation("io.github.gogoymh:dmcall:1.0.0")` to your dependencies.

The only runtime dependency is `jackson-databind`. HTTP goes through `java.net.http.HttpClient`.

## Quickstart

```java
import io.github.gogoymh.dmcall.*;

Tool weather = Tool.function("get_weather", "Current weather for a city.", Json.parse("""
        {"type": "object",
         "properties": {"city": {"type": "string", "description": "City name."}},
         "required": ["city"]}
        """), arguments -> "21 degrees in " + arguments.get("city").asText());

// OPENAI_API_KEY for the generator, TYPESAFE_API_KEY for Jev, the default decision model.
try (DMCall agent = DMCall.builder("openai:gpt-5-nano").tools(weather).build()) {
    Result result = agent.run("Is it warm in Seoul right now?");
    System.out.println(result.call());      // get_weather(city="Seoul")
    System.out.println(result.accepted());  // true
    System.out.println(result.output());    // 21 degrees in Seoul
}
```

`call(...)` proposes, verifies and repairs a call without running it. `run(...)` does the same and
then runs the accepted call with the tool's handler. A call that was never accepted does not run
unless you ask for it with `CallOptions.create().executeUnverified(true)`.

To reproduce the paper's generator settings, build the provider yourself:

```java
Generator nano = OpenAI.builder("gpt-5-nano").reasoningEffort("none").build();
DMCall agent = DMCall.builder(nano).tools(weather).build();
```

## The API

### `DMCall`

Build a client with `DMCall.builder(generator)` or `DMCall.builder("provider:model")`, add tools and
options, then call `build()`. A client built from a spec string owns its provider and the default
Jev, and `close()` releases them. A generator or decision model you pass in stays yours to close.

| Method | What it does |
|---|---|
| `call(messages)` | Proposes, verifies and repairs a call. Nothing runs. |
| `run(messages)` | `call`, then runs the accepted call with its handler. |
| `verify(messages, calls)` | Checks calls from any source with the schema check and both questions. |
| `call(messages, CallOptions)`, `run(messages, CallOptions)` | Per-request tools, handlers, repair limit, fallback rule or `executeUnverified`. |
| `callAsync`, `runAsync`, `verifyAsync` | The same on a thread pool, as `CompletableFuture`s. |

`messages` is a prompt string, a `Message` (`Message.user("...")`), or a list of messages given as
`Message`s, maps or Jackson `JsonNode`s with a `role` (system, developer, user or assistant) and text
`content`. `verify` also accepts earlier tool turns with `tool_calls` and `tool_call_id`.

### `Result`

| Accessor | Meaning |
|---|---|
| `calls()` | The final candidate as `List<Call>`. `null` means the output was unreadable (cut off, filtered, refused or malformed); an empty list means the model decided to make no call. |
| `call()` | The first call, or `null`. `Call` has `name()` and `arguments()` (a `JsonNode`). |
| `accepted()` | `Boolean`: whether the final candidate passed every check. `null` for an initial no call returned without verification. |
| `exhausted()`, `repairs()`, `finalAttempt()` | Whether no candidate passed, how many repairs were made, and the index of the final candidate in `trace()`. |
| `text()` | The final candidate's text, if any. |
| `executed()`, `output()`, `outputs()` | Whether `run` executed the call, and what the handler returned. |
| `initialNoCallPreserved()` | The first candidate was a readable no call, returned as it is. |
| `attempts()` | Every candidate with its verdict, probabilities and schema errors, oldest first. |
| `probabilities()` | The decision model's probabilities of yes for the final candidate. |
| `usage()` | Token counts summed over the generator and the decision model requests. |
| `trace()` | Every round: proposal, judgment, schema check and feedback record. |
| `toJson()` | A JSON summary with the same fields as the Python reference. |

### Tools

A tool is a JSON Schema function definition, optionally paired with a `ToolHandler`. Definitions
may be in OpenAI, Anthropic (`input_schema`) or Gemini format (`parameters` or
`parametersJsonSchema`, upper-case types allowed).

```java
Tool a = Tool.of(Json.parse(openAiStyleDefinition), arguments -> lookUp(arguments));
Tool b = Tool.function("get_time", "Current local time in a city.", parametersSchema);
DMCall agent = DMCall.builder(nano).tools(a, b).function("get_time", arguments -> now(arguments)).build();
```

A handler receives the arguments as an `ObjectNode`. A checked exception from a handler reaches the
caller as `ToolExecutionException`; a call without a handler raises `NoSuchElementException`.

### Helpers

| Name | What it does |
|---|---|
| `Verifier.verify(messages, calls, tools, decisionModel)` | Verifies calls without a `DMCall` client. Without a decision model it uses Jev from `TYPESAFE_API_KEY`. |
| `Calls.normalize(calls)` | Calls from OpenAI (Chat Completions or Responses), Anthropic or Gemini shapes as `List<Call>`. `null` means no call. |
| `SchemaCheck.check(tools, calls)` | The local schema check: the function exists and every required argument is present with its declared JSON type. Value constraints are left to the decision model. |
| `Feedback.label(p)`, `Feedback.represent(...)`, `Feedback.singleCheck(...)` | The verbal bands and the feedback representations. |
| `Prompts.SYSTEM_PROMPT`, `Prompts.FUNCTION_QUESTION`, `Prompts.ARGUMENT_QUESTION` | The exact text the paper's experiments sent. |
| `Json.dumps(node)` | Python's `json.dumps(value, ensure_ascii=False)`, used for the history message. |

## Generators

| Class | Endpoint | Key |
|---|---|---|
| `OpenAI` | Chat Completions, `https://api.openai.com/v1` (`OPENAI_BASE_URL`) | `OPENAI_API_KEY` |
| `OpenAICompatible` | any Chat Completions server, such as vLLM or Ollama; `baseUrl(...)` is required | none needed |
| `XAI` | `https://api.x.ai/v1` | `XAI_API_KEY` |
| `OpenRouter` | `https://openrouter.ai/api/v1` | `OPENROUTER_API_KEY` |
| `Meta` | Meta Model API (Muse models such as `muse-spark-1.3`), `https://api.meta.ai/v1` | `MODEL_API_KEY` |
| `AzureOpenAI` | a deployment at `endpoint(...)` or `AZURE_OPENAI_ENDPOINT` | `AZURE_OPENAI_API_KEY` |
| `Anthropic` | Messages API (`ANTHROPIC_BASE_URL`) | `ANTHROPIC_API_KEY` |
| `Gemini` | `generateContent` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |

Each class has a builder, for example `Gemini.builder("gemini-2.5-flash").temperature(0).build()`.
Common options are `apiKey`, `baseUrl`, `maxOutputTokens` (4096), `temperature`, `systemPrompt`,
`timeout` (120 seconds), `maxRetries` (3 retries on 408, 429 and 5xx), `header` and `transport`.
`Providers.fromSpec("openai:gpt-5-nano")` builds one from a spec string; the names are `openai`,
`azure`, `anthropic` (or `claude`), `gemini` (or `google`), `xai` (or `grok`), `openrouter` and
`meta` (for example `"meta:muse-spark-1.3"`).

`Meta` sends neither `reasoning_effort` nor `parallel_tool_calls` unless you set them, because Meta
rejects `reasoning_effort: "none"` for Muse models. The paper's `reasoningEffort("none")` setting for
gpt-5-nano therefore does not carry over to Muse models.
`Providers.builder(spec)` returns the builder so you can set options first.

Any other model can stand in: implement `Generator.generate(messages, tools, history)` and return a
`Proposal`. Keep an unreadable output (`Proposal.unreadable(...)`) separate from a decision to make no
call (`Proposal.of(List.of(), text)`).

## Decision models

| Class | What it is |
|---|---|
| `Jev` | TypeSafe's hosted decision model. Key from `apiKey(...)` or `TYPESAFE_API_KEY`; model `jev-1.13.0` (`TYPESAFE_MODEL`); endpoint `https://api.typesafe.ai` (`TYPESAFE_BASE_URL`). |
| `PPLXDecider` | The open-weight Perplexity Decider v1.1 27B on your own machine: `PPLXDecider.builder("http://127.0.0.1:18081").build()`. Plain HTTP only for localhost, no key, no HTTP retries, 180 second timeout. Missing token usage stays unknown. `checkProvenance(true)` checks the pinned checkpoint first. |
| `SystemOneDecisionModel` | The `POST /v1/systemone` protocol both of them speak, for any other server of that protocol. Subclass it and override `ask(state, questions)` to connect a decision model that speaks another protocol. |
| `DecisionModel` | The interface: `judge(messages, tools, calls)`. |

Thresholds belong to the decision model: `Jev.builder().threshold(0.5)`, or `functionThreshold` and
`argumentsThreshold` for one question each. When DM-Call creates the default Jev, set its threshold
and key on the `DMCall` builder with `threshold(...)` and `jevApiKey(...)`.

## Options and the paper's variants

The defaults are the paper's settings.

| Builder option | Default | Paper |
|---|---|---|
| `checks(Checks.BOTH)` | both questions decide | DM-Call |
| `checks(Checks.FUNCTION)` | | DM-Call-Function |
| `checks(Checks.ARGUMENTS)` | | DM-Call-Argument |
| `history(History.CUMULATIVE)` | every earlier rejection | DM-Call |
| `history(History.LATEST)` | | DM-Call-Latest |
| `feedback(FeedbackMode.HYBRID)` | probability, threshold and verbal label | Hybrid feedback |
| `feedback(FeedbackMode.NUMERIC)`, `feedback(FeedbackMode.TEXT)` | | Numeric and Text feedback |
| `maxRepairs(3)` | 3 | the repair limit r |
| `threshold(0.5)` | 0.5 | the pass mark of both questions |
| `onExhaustion(OnExhaustion.FIRST)` | the first candidate when none passes | the paper's rule; `LAST` returns the last |
| `preserveInitialNoCall(true)` | a readable initial no call is returned without verification | the paper's policy |
| `executeUnverified(false)` | `run` executes accepted calls only | |

The options also accept the spec's names as strings, for example `checks("function")`.

## Running the tests

```sh
cd java
mvn -q test
```

The tests read the conformance files from `../spec`, so run them from the repository checkout. They
serve recorded response bodies through a mock `HttpTransport` and never use the network.
`ConformanceTest` replays every case in `spec/conformance` (JSON text, schema check, feedback, call
normalization, provider requests and decoding, every paper-parity scenario plus `verify`, and every
loop edge scenario). The test cases come from the files themselves, so new cases run without code
changes. The other test classes port the unit tests in `tests/`.

## Differences from the Python package

- Python functions as tools (signature and docstring inference) have no Java counterpart. Tools are
  JSON Schema definitions with an optional `ToolHandler`.
- `OpenAI`, `XAI`, `OpenRouter`, `Meta` and `AzureOpenAI` share the abstract `ChatCompletionsProvider`
  with `OpenAICompatible` instead of extending it.
- Errors are `IllegalArgumentException` for invalid options and input, `ProviderException` and
  `DecisionModelException` for HTTP and protocol failures, `NoSuchElementException` for a missing
  handler and `ToolExecutionException` for a failing handler.
- The schema check follows JSON Schema Draft 2020-12, the default of the Python reference. A schema
  that names an older draft in `$schema` is still checked as Draft 2020-12. A `$ref` that cannot be
  resolved raises `IllegalArgumentException`.
- `Call.name()` is a `String`; `Calls.normalize` turns a name that is not a string into `null`.
