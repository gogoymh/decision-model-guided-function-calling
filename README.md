# JBFC: Jev-Backed Function Calling

**Your LLM's tool calls, checked before they run, and repaired when they are wrong.**

LLMs call the wrong function, invent argument values, and call tools when no tool applies. A JSON schema cannot catch any of that, because a wrong call can be perfectly well formed. JBFC puts a verifier between the model and your code. [Jev](https://typesafe.ai) answers two yes/no questions about every proposed call: *is this the right function?* and *are these the right arguments?* When a check fails, the same model gets the verdicts back and repairs the call. Only a call that passes runs.

```python
import jbfc

agent = jbfc.JBFC("openai:gpt-5-nano", tools=[get_weather, book_table])
result = agent.run("Is it jacket weather in Seoul tonight?")  # runs the call only if Jev accepts it
```

- **+9.7 points** on BFCL with `gpt-5-nano`: 63.8% → 73.5% call accuracy on 3,051 examples, and 77.0% with up to seven repairs.
- **Any model**: OpenAI, Anthropic, Gemini, xAI Grok, Azure OpenAI, OpenRouter, or any OpenAI-compatible server (vLLM, Ollama, LM Studio).
- **Plain Python tools**: pass functions; schemas come from type hints and docstrings.
- **Drop-in verification**: check calls produced by any SDK or agent framework with `jbfc.verify(...)`.
- **Small**: plain HTTP via `httpx`, no vendor SDKs. It is the same loop that produced the paper's numbers.

## How it works

![JBFC: the model proposes a call, Jev checks it, and rejected calls return to the model as a growing feedback history](docs/assets/jbfc-framework.png)

**(a)** The LLM proposes a call $c_t$, one function with its arguments, or no call. A local schema check and Jev judge it: Jev sees the request, the available functions and the call, and answers *right function?* and, when there is a call, *right arguments?* as probabilities of yes. A check passes at 0.8 or more. A rejected call and its verdicts $d_t$ are appended to the feedback history $H_t$, and the model tries again. **(b)** The history goes *before* the original request, oldest first, and is marked as not executed, so the model sees every earlier attempt and can change the function, the arguments, or decide not to call. (The scores in (b) are illustrative.)

The loop stops at the first proposal that passes every check, or after `max_repairs` repairs (2 by default). Nothing runs during repairs; `run()` executes only the final, accepted call.

## Results

All 3,051 examples from seven BFCL categories, one run, `gpt-5-nano` (reasoning effort `none`) as the generator, `jev-1.13.0` as the verifier, at most two repairs. From the paper in [`paper/`](paper/main.pdf):

| Category | Examples | Direct | **JBFC** |
|---|---:|---:|---:|
| Simple Python | 400 | 90.8 | **91.5** |
| Multiple | 200 | 91.0 | **91.5** |
| Irrelevance | 240 | 25.4 | **74.2** |
| Live simple | 258 | 73.6 | **75.2** |
| Live multiple | 1,053 | 67.7 | **71.2** |
| Live irrelevance | 884 | 48.2 | **63.2** |
| **Total** (excl. 16 live relevance) | 3,035 | 63.8 | **73.5** |

JBFC scores highest among the seven methods we compared, which share each example's first proposal:

| Method | Call acc. | Function acc. | Argument acc. | API sec. | USD / 1k requests |
|---|---:|---:|---:|---:|---:|
| Direct (no checks) | 63.8 | 74.8 | 81.2 | 1.30 | 0.056 |
| Schema repair | 65.7 | 76.4 | 81.8 | 1.36 | 0.059 |
| Self-refine (3 proposals, every request) | 69.8 | 80.0 | 82.3 | 4.01 | 0.190 |
| JBFC-F (function check only) | 72.6 | 83.5 | 81.8 | 2.61 | 0.190 |
| JBFC-A (argument check only) | 72.9 | 82.8 | **83.0** | 2.40 | 0.175 |
| JBFC-L (latest feedback only) | 72.8 | 83.1 | 82.6 | 2.92 | 0.217 |
| **JBFC** | **73.5** | **83.7** | 82.8 | 2.90 | 0.221 |

- Every repair method beats Direct; JBFC by 9.7 points (paired bootstrap 95% CI 8.5–11.0).
- Generic self-review spends three proposals on every request and still trails every Jev-based method, which average 1.5 to 1.8 proposals.
- Keeping the whole revision history adds 0.7 points over showing only the latest rejection (95% CI 0.1–1.4).
- Calls that pass all checks are 90.6% correct (68.4% of examples), so the verdict is also a useful confidence signal.

More repairs keep paying off, at a price. The same run, with JBFC's rejected trajectories continued up to seven repairs:

| `max_repairs` | 1 | 2 (default) | 3 | 5 | 7 |
|---|---:|---:|---:|---:|---:|
| Call accuracy | 72.6 | 73.5 | 75.3 | 76.2 | **77.0** |
| Mean repairs used | 0.42 | 0.75 | 1.07 | 1.69 | 2.29 |
| API seconds | 2.33 | 2.90 | 3.49 | 4.65 | 5.79 |
| USD / 1k requests | 0.172 | 0.221 | 0.273 | 0.378 | 0.483 |

Most of the extra gain is the model finally withholding an unnecessary call after repeated rejections.

What it costs: about four times Direct (still $0.22 per 1,000 requests with `gpt-5-nano`) and 1.6 extra API seconds on average. Most of the gain comes from requests where no tool applies, as the first trace below shows. In categories that need a call, Jev rejects most wrong calls, but a yes/no verdict says *that* a call is wrong, not *which* call is right, so fewer of them get fixed. See [Limits](#limits).

### Real traces from the run

```text
User:  How many kilograms are in a pound?
Tools: portfolio_returns(initial_investment, ending_value, period)

gpt-5-nano  portfolio_returns(ending_value=1, initial_investment=1)
Jev         right function? p=0.04                                  rejected
gpt-5-nano  (no call) "A pound is exactly 0.45359237 kilograms."
Jev         right to make no call? p=0.97                           accepted
```

```text
User:  Find the closest sushi restaurant with a patio in Boston.

gpt-5-nano  restaurant_search_find_closest(cuisine="Sushi", location="Boston",
                amenities=["Patio", "Wi-Fi", "Happy Hour", "Wheelchair Accessible"])
Jev         right function? p=0.95   right arguments? p=0.17       rejected
gpt-5-nano  restaurant_search_find_closest(location="Boston", cuisine="Sushi", amenities=["Patio"])
Jev         right function? p=0.97   right arguments? p=0.89       accepted
```

## Install

Python 3.11 or newer. From a clone of this repository:

```bash
pip install -e .        # or: uv sync
```

Set the keys for Jev and for your model provider:

```bash
export TYPESAFE_API_KEY=...   # Jev, from TypeSafe (https://typesafe.ai)
export OPENAI_API_KEY=...     # or ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, ...
```

## Quickstart

```python
import jbfc

def get_weather(city: str, unit: str = "celsius") -> dict:
    """Current weather for a city.

    Args:
        city: City name, e.g. "Seoul".
        unit: "celsius" or "fahrenheit".
    """
    return {"city": city, "temp": 12, "unit": unit}

agent = jbfc.JBFC("openai:gpt-5-nano", tools=[get_weather])

result = agent.run("Do I need a jacket in Seoul tonight?")
result.call       # {"name": "get_weather", "arguments": {"city": "Seoul"}}
result.accepted   # True: Jev and the schema check passed
result.output     # {"city": "Seoul", "temp": 12, "unit": "celsius"}
result.repairs    # repairs it took (0, 1 or 2)
```

`run()` executes the final call only if it passed. `call()` does everything except execute, so you decide:

```python
result = agent.call("Book a table for two at 7pm")
if result.accepted:
    ...                                  # run result.call yourself
elif result.no_call:
    print(result.text)                   # the model answered without a tool
```

## Use any model

The model is a `"provider:model"` string or a provider object. Options such as `api_key=` go to the provider; `jev_api_key=` goes to Jev.

```python
jbfc.JBFC("openai:gpt-5-nano")                       # OPENAI_API_KEY
jbfc.JBFC("anthropic:claude-sonnet-4-5")             # ANTHROPIC_API_KEY
jbfc.JBFC("gemini:gemini-2.5-flash")                 # GEMINI_API_KEY or GOOGLE_API_KEY
jbfc.JBFC("xai:grok-4")                              # XAI_API_KEY
jbfc.JBFC("openrouter:meta-llama/llama-3.3-70b-instruct")   # OPENROUTER_API_KEY
jbfc.JBFC("azure:my-deployment", endpoint="https://my-resource.openai.azure.com")  # AZURE_OPENAI_API_KEY

# Local or self-hosted, OpenAI-compatible (Ollama, vLLM, LM Studio, ...)
jbfc.JBFC(jbfc.OpenAICompatible("llama3.1", base_url="http://localhost:11434/v1"))

# Keys in code instead of the environment
jbfc.JBFC("anthropic:claude-sonnet-4-5", api_key="sk-ant-...", jev_api_key="...")
```

| Provider | Spec | Key | Class |
|---|---|---|---|
| OpenAI | `openai:<model>` | `OPENAI_API_KEY` | `jbfc.OpenAI` |
| Anthropic | `anthropic:<model>` | `ANTHROPIC_API_KEY` | `jbfc.Anthropic` |
| Google Gemini | `gemini:<model>` | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | `jbfc.Gemini` |
| xAI Grok | `xai:<model>` | `XAI_API_KEY` | `jbfc.XAI` |
| Azure OpenAI | `azure:<deployment>` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` | `jbfc.AzureOpenAI` |
| OpenRouter | `openrouter:<model>` | `OPENROUTER_API_KEY` | `jbfc.OpenRouter` |
| Anything OpenAI-compatible | | optional | `jbfc.OpenAICompatible` |
| Jev (verifier) | | `TYPESAFE_API_KEY` | `jbfc.Jev` |

Provider objects take `max_output_tokens` (4,096 by default), `temperature`, `system_prompt`, `timeout`, `max_retries` and `headers`; OpenAI-style providers also take `reasoning_effort`, and Gemini takes `thinking_budget`. A truncated, filtered or refused response is treated as a failed proposal and repaired, never as "no call".

## Tools

Plain functions work as they are: parameter types come from type hints (`str`, `int`, `float`, `bool`, `list[...]`, `dict[...]`, `Literal`, `Enum`, `Optional`, `TypedDict`, `Annotated[..., "description"]`), descriptions from Google, NumPy or Sphinx docstrings, and defaults from the signature. Use the decorator to override anything, or pass JSON Schema definitions in OpenAI, Anthropic or Gemini format:

```python
@jbfc.tool(name="weather", description="Current weather for a city.")
def get_weather(city: str, unit: Literal["celsius", "fahrenheit"] = "celsius") -> dict: ...

schema = {"type": "function", "function": {
    "name": "search", "description": "Search the web.",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}}
agent = jbfc.JBFC("openai:gpt-5-nano", tools=[get_weather, schema], functions={"search": my_search})
```

`jbfc.schema_of(fn)` shows the schema JBFC sends.

## Verify calls from any framework

Already have an agent? Keep it, and check its calls before you execute them. `verify` takes the conversation so far (earlier tool calls and tool results included) and the proposed calls, as plain dicts or as the message and tool-call objects of the OpenAI, Anthropic and Google SDKs.

```python
from openai import OpenAI

response = OpenAI().chat.completions.create(model="gpt-5-nano", messages=messages, tools=tools)
verdict = jbfc.verify(messages, response.choices[0].message.tool_calls, tools)

if verdict:                       # passed the schema check and both Jev checks
    execute(response.choices[0].message.tool_calls)
else:
    print(verdict.function_probability, verdict.arguments_probability, verdict.schema_errors)
    # verdict.feedback is the record JBFC would send back to the model for a repair
```

## The result

| Field | Meaning |
|---|---|
| `calls` | final proposal: `[{"name", "arguments"}]`, `[]` for no call, `None` if unreadable |
| `call` | the first call, or `None` |
| `accepted` | passed every check (`None` for methods without a verifier) |
| `exhausted` | hit the repair limit without passing |
| `repairs` | repairs used |
| `text` | the model's text, e.g. its answer when it made no call |
| `executed`, `output` | whether `run()` executed the call, and what it returned |
| `attempts` | every proposal with its Jev probabilities and schema errors |
| `usage`, `latency_s` | tokens for the model and for Jev, and wall time |
| `trace` | the full record: proposals, verdicts and feedback |

## Options and baselines

```python
jbfc.JBFC(
    "openai:gpt-5-nano",
    tools=[...],
    max_repairs=2,             # repairs after the first proposal
    threshold=0.8,             # Jev threshold for both checks
    method="jbfc",             # or a baseline: "direct", "schema", "self-refine",
                               # "jev-function", "jev-arguments", "jbfc-latest"
    execute_unverified=False,  # run() also executes a call still rejected at the limit
)
```

- `jbfc.Jev(api_key, threshold=0.8, function_threshold=None, arguments_threshold=None, model="jev-1.13.0")` configures the verifier; pass it as `jev=`.
- `acall`, `arun` and `averify` run the same steps in a worker thread for async code.
- Any object with `generate(example, history)` returning `{"calls": ..., "content": ...}` can stand in for a provider.
- `jbfc.SYSTEM_PROMPT` is the generator prompt used in the paper; replace it with `system_prompt=` on a provider.

## Limits

- **One decision per step.** JBFC decides a single call or no call, the BFCL setting it was evaluated on. Agents call it once per step, and parallel calls are out of scope. `call()` and `run()` take text messages, so give earlier tool results back to the model as text; `verify()` also accepts tool-call and tool-result messages.
- **A verifier is a model too.** Jev can accept a wrong call or reject a right one; accepted calls were 90.6% correct in the paper's run, not 100%.
- **Cost and latency.** Each proposal costs one generator call and one Jev call; with at most two repairs, that is up to three of each.
- **Evidence so far.** The numbers come from one model (`gpt-5-nano` on Azure OpenAI), mostly English requests, and one run. The other provider adapters follow each API's documented format and are covered by offline tests, not yet by benchmark runs.

## Reproduce the paper

The experiment code lives in [`src/jev_fc`](src/jev_fc) and the manuscript in [`paper/main.tex`](paper/main.tex). JBFC's loop in the library *is* `jev_fc.pipeline.run_example`, and offline tests check that the library sends the same prompts as the experiments.

```bash
make setup         # uv sync and a .env from .env.example
make prepare       # download and convert the pinned BFCL revision (3,051 examples)
make test          # offline tests and lint; no keys needed
make check-config  # show the configured routes, models and keys, without API calls
make table1        # Direct vs JBFC (6,102 records)
make table2        # all seven methods (21,357 records); also fills Table 1
make table3        # JBFC at repair limits 1, 2, 3, 5, 7, extending the Table 2 run
python scripts/analyze_results.py results/table2_gpt-5-nano_r2   # the numbers quoted in Results
```

The paper's runs used an Azure OpenAI deployment of `gpt-5-nano` and Jev behind an Azure API Management gateway; direct API keys work too (see [`docs/apim-gateway.md`](docs/apim-gateway.md)). With `WORKERS=8`, Table 2 took about two hours and $2.4 at list prices, and the Table 3 extension about 20 minutes and $0.8. Completed records are reused when a run resumes. Run notes and checks are in [`docs/validation.md`](docs/validation.md).

```text
src/jbfc/      the library: JBFC, providers, Jev, tools
src/jev_fc/    experiment code: repair loop, Jev client, BFCL data and scoring, CLI (jev-fc)
scripts/       table runs, run extension for Table 3, analysis, paper build and checks
paper/         manuscript and PDF
tests/         offline tests
docs/          benchmark, gateway and validation notes
```

## Citation

```bibtex
@misc{yu2026jbfc,
  title  = {Jev-Backed Function Calling: Iterative Repair with Cumulative Decision Feedback},
  author = {Yu, Minhyeong},
  year   = {2026},
  note   = {Enhans}
}
```

## Acknowledgements

This work was created as part of the No-More-Slops group activities at Pseudo Lab. Jev is TypeSafe's verifier model; the benchmark is the Berkeley Function Calling Leaderboard (BFCL).
