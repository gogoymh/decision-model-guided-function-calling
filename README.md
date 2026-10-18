<p align="center">
  <img src="docs/assets/jbfc-hero.png" alt="Jev-Backed Function Calling: Agentic In-Context Learning with Cumulative Decision Feedback (JBFC). The LLM proposes a call, Jev checks it, a rejected call goes back to the LLM with the verdict, and only a call that passes runs." width="100%">
</p>

<p align="center">
  <img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white">
  <img alt="Anthropic" src="https://img.shields.io/badge/Anthropic-D97757?style=flat-square&logo=anthropic&logoColor=white">
  <img alt="Google Gemini" src="https://img.shields.io/badge/Gemini-8E75B2?style=flat-square&logo=googlegemini&logoColor=white">
  <img alt="xAI Grok" src="https://img.shields.io/badge/xAI%20Grok-000000?style=flat-square&logo=x&logoColor=white">
  <img alt="Azure OpenAI" src="https://img.shields.io/badge/Azure%20OpenAI-0078D4?style=flat-square">
  <img alt="OpenRouter" src="https://img.shields.io/badge/OpenRouter-6566F1?style=flat-square&logo=openrouter&logoColor=white">
  <img alt="Ollama and vLLM" src="https://img.shields.io/badge/Ollama%20%C2%B7%20vLLM%20%C2%B7%20any%20OpenAI--compatible-30363d?style=flat-square&logo=ollama&logoColor=white">
</p>

<p align="center">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <a href="LICENSE"><img alt="MIT license" src="https://img.shields.io/badge/license-MIT-eab308?style=flat-square"></a>
  <img alt="599 offline tests" src="https://img.shields.io/badge/tests-599%20passing-16a34a?style=flat-square">
  <img alt="Dependencies: httpx and jsonschema" src="https://img.shields.io/badge/deps-httpx%20%2B%20jsonschema-6b7280?style=flat-square">
  <a href="paper/main.pdf"><img alt="Paper (PDF)" src="https://img.shields.io/badge/paper-PDF-b31b1b?style=flat-square&logo=latex&logoColor=white"></a>
</p>

<p align="center">
  <a href="#how-it-works"><b>How it works</b></a> ·
  <a href="#see-it"><b>See it</b></a> ·
  <a href="#install"><b>Install</b></a> ·
  <a href="#quickstart"><b>Quickstart</b></a> ·
  <a href="#use-any-model"><b>Any model</b></a> ·
  <a href="#verify-calls-from-any-framework"><b>Verify</b></a> ·
  <a href="#results"><b>Results</b></a> ·
  <a href="#faq"><b>FAQ</b></a> ·
  <a href="#reproduce-the-paper"><b>Reproduce</b></a>
</p>

LLMs call the wrong function, invent argument values, and call tools when no tool applies. A JSON schema cannot catch that, because a wrong call can be perfectly well formed. **JBFC** puts a verifier between the model and your code. [Jev](https://typesafe.ai) answers two yes/no questions about every proposed call: *is this the right function?* and *are these the right arguments?* When a check fails, the same model gets the verdicts back and repairs the call. Only a call that passes runs.

```python
import jbfc

agent = jbfc.JBFC("openai:gpt-5-nano", tools=[get_weather, book_table])
agent.run("Is it jacket weather in Seoul tonight?")  # runs the call only if Jev accepts it
```

> [!TIP]
> Already have an agent? Keep it. `jbfc.verify(messages, tool_calls, tools)` checks calls from the OpenAI, Anthropic or Google SDKs before you execute them. [Jump there](#verify-calls-from-any-framework).

## How it works

![JBFC: the model proposes a call, Jev checks it, and rejected calls return to the model as a growing feedback history](docs/assets/jbfc-framework.png)

**(a)** The LLM proposes a call $c_t$, one function with its arguments, or no call. A local schema check and Jev judge it: Jev sees the request, the available functions and the call, and answers *right function?* and, when there is a call, *right arguments?* as probabilities of yes. A check passes at 0.8 or more. A rejected call and its verdicts $d_t$ are appended to the feedback history $H_t$, and the model tries again. **(b)** The history goes *before* the original request, oldest first, and is marked as not executed, so the model sees every earlier attempt and can change the function, the arguments, or decide not to call. (The scores in (b) are illustrative.)

Each record is a labeled example: a call the model wrote, with Jev's verdict on it. Like the demonstrations in [many-shot in-context learning](https://arxiv.org/abs/2404.11018), more of them seemed to help in our run: the full history scored slightly above the latest record alone, and accuracy rose with the repair limit (which also adds attempts). Jev makes those labels in a fraction of a second, without reference answers.

The loop stops at the first proposal that passes every check, or after `max_repairs` repairs (3 by default). Nothing runs during repairs; `run()` executes only the final, accepted call.

## See it

Two real traces from the paper's run (`gpt-5-nano` on BFCL, `jev-1.13.0`). Red lines were rejected; green lines passed.

**An unnecessary call, withdrawn.** The only tool is unrelated, and the model calls it anyway:

```diff
  User:  How many kilograms are in a pound?
  Tools: portfolio_returns(initial_investment, ending_value, period)

- gpt-5-nano   portfolio_returns(ending_value=1, initial_investment=1)
- Jev          right function? p=0.04                              → rejected
+ gpt-5-nano   (no call) "A pound is exactly 0.45359237 kilograms."
+ Jev          right to make no call? p=0.97                       → accepted
```

**An invented argument, removed.** The user asked for a patio; the model added three amenities nobody asked for:

```diff
  User:  Find the closest sushi restaurant with a patio in Boston.

- gpt-5-nano   restaurant_search_find_closest(cuisine="Sushi", location="Boston",
-                  amenities=["Patio", "Wi-Fi", "Happy Hour", "Wheelchair Accessible"])
- Jev          right function? p=0.95   right arguments? p=0.17   → rejected
+ gpt-5-nano   restaurant_search_find_closest(cuisine="Sushi", location="Boston", amenities=["Patio"])
+ Jev          right function? p=0.97   right arguments? p=0.89   → accepted
```

## Why JBFC

<table>
  <tr>
    <td width="33%" valign="top"><b>🎯 Checks what schemas cannot</b><br>Wrong function, invented values, a call that should not happen at all. Jev judges meaning; a local schema check covers format.</td>
    <td width="33%" valign="top"><b>🔁 Repairs instead of failing</b><br>Rejected calls go back to the model with every earlier attempt and verdict. Nothing runs until a call passes.</td>
    <td width="33%" valign="top"><b>⚡ Light checks</b><br>A Jev verdict took about 0.3 s per call in our runs, less time than the model took to propose the call, and about $0.06 per 1,000 checks.</td>
  </tr>
  <tr>
    <td valign="top"><b>🔌 Any model</b><br>OpenAI, Anthropic, Gemini, xAI Grok, Azure OpenAI, OpenRouter, or any OpenAI-compatible server. Plain HTTP, no vendor SDKs.</td>
    <td valign="top"><b>🧰 Plain Python tools</b><br>Pass functions. Schemas come from type hints and docstrings; JSON Schema works too.</td>
    <td valign="top"><b>📄 Open study</b><br>The library runs the same loop as the paper. Code, the pinned BFCL revision and the checks behind every table are in this repo.</td>
  </tr>
</table>

## Install

Python 3.11 or newer. From a clone of this repository:

```bash
pip install -e .        # or: uv sync
```

Set the keys for Jev and for your model provider:

```bash
export TYPESAFE_API_KEY=...   # Jev, from TypeSafe (https://console.typesafe.ai)
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
result.repairs    # repairs it took (0 to 3)
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

<details>
<summary><b>Provider options</b></summary>

Provider objects take `max_output_tokens` (4,096 by default), `temperature`, `system_prompt`, `timeout`, `max_retries` and `headers`; OpenAI-style providers also take `reasoning_effort`, and Gemini takes `thinking_budget`. A truncated, filtered or refused response is treated as a failed proposal and repaired, never as "no call". Any object with `generate(example, history)` returning `{"calls": ..., "content": ...}` can stand in for a provider.

</details>

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

Keep your agent and check its calls before you execute them. `verify` takes the conversation so far (earlier tool calls and tool results included) and the proposed calls, as plain dicts or as the message and tool-call objects of the OpenAI, Anthropic and Google SDKs.

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

## Options and baselines

```python
jbfc.JBFC(
    "openai:gpt-5-nano",
    tools=[...],
    max_repairs=3,             # repairs after the first proposal
    threshold=0.8,             # Jev threshold for both checks
    method="jbfc",             # or a baseline: "direct", "schema", "self-refine",
                               # "jev-function", "jev-arguments", "jbfc-latest"
    execute_unverified=False,  # run() also executes a call still rejected at the limit
)
```

- `jbfc.Jev(api_key, threshold=0.8, function_threshold=None, arguments_threshold=None, model="jev-1.13.0")` configures the verifier; pass it as `jev=`.
- `acall`, `arun` and `averify` run the same steps in a worker thread for async code.
- `jbfc.SYSTEM_PROMPT` is the generator prompt used in the paper; replace it with `system_prompt=` on a provider.

<details>
<summary><b>The result object</b></summary>

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

</details>

## Results

> [!NOTE]
> A small study, not a benchmark claim: two generators (`gpt-5-nano` and the larger `gpt-5.4-mini`, reasoning effort `none`) and one verifier (`jev-1.13.0`) on 3,051 single-turn examples from seven BFCL categories, at most three repairs. Direct and JBFC ran three times per generator and the category numbers are their means; the method comparison and the repair-limit curve come from the first run. The repair limit was chosen after seeing that curve on the same examples; nothing else was tuned. Within a run, the methods of one generator share each example's first proposal. Intervals and caveats are in the [paper](paper/main.pdf).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/chart-categories-dark.png">
  <img alt="Accuracy by BFCL category, mean of three runs, Direct then JBFC, for gpt-5-nano and gpt-5.4-mini: Simple Python 90.8 to 91.0 and 90.5 to 91.2; Multiple 91.0 to 91.5 and 93.3 to 93.3; Irrelevance 27.2 to 81.7 and 87.1 to 81.1; Live simple 74.3 to 75.1 and 77.1 to 79.1; Live multiple 67.6 to 70.4 and 75.2 to 76.4; Live irrelevance 48.8 to 67.7 and 80.7 to 74.4; total 64.1 to 75.0 and 81.1 to 79.5." src="docs/assets/chart-categories-light.png" width="100%">
</picture>

- With `gpt-5-nano`, JBFC raised call accuracy from 64.1% to 75.0% on average (paired bootstrap 95% CI for the difference 9.8–12.1 points; +11.5, +10.6 and +10.6 in the three runs), mostly on requests where no tool applies (Irrelevance: 27.2% → 81.7%). Where a call is needed, the gains are small (Simple Python 90.8% → 91.0%, Multiple 91.0% → 91.5%): a yes/no verdict says *that* a call is wrong, not *which* call is right.
- With `gpt-5.4-mini`, JBFC lowered accuracy in every run, from 81.1% to 79.5% on average (95% CI for the difference −2.4 to −0.9). It still fixed more calls than it broke where a call was needed, but this model already skips most unrelated tools, Jev separated its remaining mistakes poorly, and Jev's rejections turned 102–110 correct no-calls into calls per run.
- A larger model alone beat the small model with JBFC: Direct with `gpt-5.4-mini` reached 81.1%, 6.1 points above JBFC with `gpt-5-nano` (95% CI 4.9–7.4), at 2.6 times the cost ($0.70 instead of $0.27 per 1,000 requests) and in about a third of the time.
- The three runs agree: each total moved by at most 0.6 points between runs (standard deviation 0.2–0.3), although about 30% of `gpt-5-nano`'s first calls (13% of `gpt-5.4-mini`'s) changed from run to run.
- In the first run with `gpt-5-nano`, every repair method scored above Direct. Generic self-review spends four proposals on every request and trailed every Jev-based method, which averaged 1.7 to 2.1.
- In the same run, keeping the whole history scored 1.6 points above showing only the latest rejection (95% CI 0.8–2.5).
- Accepted is not the same as correct: with `gpt-5-nano`, calls that passed every check (69% of examples in each run) were 91% correct.
- With `gpt-5-nano`, JBFC cost about five times Direct ($0.27 instead of $0.056 per 1,000 requests) and took 2.1 more API seconds.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/chart-methods-dark.png">
  <img alt="Call accuracy by method with gpt-5-nano, first run, at most three repairs: JBFC 75.3, JBFC-A 73.7, JBFC-L 73.6, JBFC-F 73.6, Self-refine 70.0, Schema repair 65.8, Direct 63.8, with API seconds and USD per 1,000 requests." src="docs/assets/chart-methods-light.png" width="100%">
</picture>

<a id="more-repairs"></a>
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/chart-limits-dark.png">
  <img alt="JBFC by repair limit: call accuracy 72.6% at r=1, 73.5% at r=2, 75.3% at r=3 (default), 76.2% at r=5, 77.0% at r=7 and 78.6% at r=10; cost per 1,000 requests $0.172, $0.221, $0.273, $0.378, $0.483 and $0.669." src="docs/assets/chart-limits-light.png" width="100%">
</picture>

Continuing the first `gpt-5-nano` run raised call accuracy from 75.3% at three repairs (the default) to 78.6% at ten, at 2.4 times the cost; most newly correct answers (128 of 148) were again cases where no tool applies. Even at ten repairs, at a cost similar to Direct with `gpt-5.4-mini`, JBFC stayed below it (78.6% against 81.1% on average). Three repairs is the default because each further point costs about four times as much; it was chosen after seeing this curve, and three repairs gave 74.9% in both later runs.

<details>
<summary><b>All numbers as tables</b></summary>

Mean of three runs per generator (the paper's Table 1):

| Category | Examples | `gpt-5-nano` Direct | `gpt-5-nano` JBFC | `gpt-5.4-mini` Direct | `gpt-5.4-mini` JBFC |
|---|---:|---:|---:|---:|---:|
| Simple Python | 400 | 90.8 | 91.0 | 90.5 | **91.2** |
| Multiple | 200 | 91.0 | 91.5 | **93.3** | **93.3** |
| Irrelevance | 240 | 27.2 | 81.7 | **87.1** | 81.1 |
| Live simple | 258 | 74.3 | 75.1 | 77.1 | **79.1** |
| Live multiple | 1,053 | 67.6 | 70.4 | 75.2 | **76.4** |
| Live irrelevance | 884 | 48.8 | 67.7 | **80.7** | 74.4 |
| **Total** (excl. 16 live relevance) | 3,035 | 64.1 | 75.0 | **81.1** | 79.5 |
| Standard deviation of the total across runs | | 0.3 | 0.2 | 0.2 | 0.2 |
| USD / 1k requests | | 0.056 | 0.272 | 0.703 | 1.700 |

First run (the paper's Table 2 and Figure 2):

| Method (`gpt-5-nano`) | Call acc. | Function acc. | Argument acc. | API sec. | USD / 1k requests |
|---|---:|---:|---:|---:|---:|
| Direct (no checks) | 63.8 | 74.8 | 81.2 | 1.30 | 0.056 |
| Schema repair | 65.8 | 76.4 | 81.8 | 1.36 | 0.059 |
| Self-refine (4 proposals, every request) | 70.0 | 80.3 | 82.2 | 5.47 | 0.265 |
| JBFC-F (function check only) | 73.6 | 84.6 | 81.6 | 2.99 | 0.223 |
| JBFC-A (argument check only) | 73.7 | 83.5 | 83.1 | 2.66 | 0.199 |
| JBFC-L (latest feedback only) | 73.6 | 84.0 | 82.7 | 3.47 | 0.262 |
| **JBFC** | **75.3** | **85.0** | **83.4** | 3.49 | 0.273 |

| `max_repairs` | 1 | 2 | 3 (default) | 5 | 7 | 10 |
|---|---:|---:|---:|---:|---:|---:|
| Call accuracy | 72.6 | 73.5 | 75.3 | 76.2 | 77.0 | 78.6 |
| Function accuracy | 83.0 | 83.7 | 85.0 | 86.0 | 86.5 | 87.7 |
| Mean repairs used | 0.42 | 0.75 | 1.07 | 1.69 | 2.29 | 3.17 |
| Reached the limit without passing | 33.4% | 31.8% | 30.9% | 30.2% | 29.7% | 29.2% |
| API seconds | 2.33 | 2.90 | 3.49 | 4.65 | 5.79 | 7.35 |
| USD / 1k requests | 0.172 | 0.221 | 0.273 | 0.378 | 0.483 | 0.669 |

</details>

## FAQ

**Is Jev just another LLM-as-a-judge?**
Jev is TypeSafe's decision model. It does not write explanations: it answers each yes/no question with a probability, in 0.30 s on average in the paper's run. JBFC asks two fixed questions (right function, right arguments) and also runs a local schema check.

**What does it cost?**
Each Jev check reads about 1,400 tokens, roughly $0.06 per 1,000 checks at $0.042 per million input tokens. With `gpt-5-nano`, JBFC cost $0.27 per 1,000 requests on average, against $0.056 for a bare call; a bare call to `gpt-5.4-mini` cost $0.70 and was more accurate.

**Does anything run while JBFC repairs a call?**
No. Proposals are marked as not executed, and `run()` executes only the final call after it passes (or, with `execute_unverified=True`, the last one).

**Can I use it with LangChain, the OpenAI Agents SDK, or my own loop?**
Yes. Call `jbfc.verify(...)` on the tool calls your framework produces, or let `JBFC.call()` make the decision for one step of your loop.

**Parallel tool calls?**
Not yet. JBFC decides one step at a time: a single call or no call, the setting it was evaluated on.

**Which models have been benchmarked?**
`gpt-5-nano` and `gpt-5.4-mini` (Azure OpenAI), each with and without Jev, three runs each. JBFC helped the first and lowered the second's accuracy in every run; see [Results](#results). The other adapters follow each provider's documented API and are covered by offline tests.

**When does JBFC help?**
When the model's mistakes are ones Jev can tell apart. In our runs that meant a small model calling tools that do not apply: Jev flagged those calls clearly and the model withdrew them. A stronger model made fewer such mistakes, Jev separated the rest poorly, and its rejections of correct no-calls cost more than its fixes. Measure on your own traffic before using it with a strong model; `verify()` returns verdicts without changing what your agent does.

## Limits

- **One decision per step.** JBFC decides a single call or no call. `call()` and `run()` take text messages, so give earlier tool results back to the model as text; `verify()` also accepts tool-call and tool-result messages.
- **A verifier is a model too.** Jev can accept a wrong call or reject a right one; accepted calls were about 90% correct in the paper's runs, not 100%, and with `gpt-5.4-mini` its rejections did more harm than good.
- **Cost and latency.** Each proposal costs one generator call and one Jev call; with at most three repairs, that is up to four of each.
- **Evidence so far.** Two models and mostly English requests; three runs of Direct and JBFC, one of the other methods. JBFC helped `gpt-5-nano` and hurt `gpt-5.4-mini` in every run; the repair limit was chosen on the first run.

## Reproduce the paper

The experiment code lives in [`src/jev_fc`](src/jev_fc) and the manuscript in [`paper/main.tex`](paper/main.tex). JBFC's loop in the library *is* `jev_fc.pipeline.run_example`, and offline tests check that the library sends the same prompts as the experiments.

```bash
make setup         # uv sync and a .env from .env.example
make prepare       # download and convert the pinned BFCL revision (3,051 examples)
make test          # offline tests and lint; no keys needed
make check-config  # show the configured routes, models and keys, without API calls
make table1        # optional: Direct vs JBFC at r=2 (6,102 records), not used by the tables
make table2        # all seven methods at r=2 (21,357 records), the source run of Tables 1-2
make table3        # Figure 2: JBFC at repair limits 1, 2, 3, 5, 7, 10, extending the Table 2 run
make compare       # Direct with gpt-5.4-mini on all examples (no Jev)
make r3            # Tables 1-2 at r=3: raise the Table 2 run, add JBFC to gpt-5.4-mini, fill both
make repeat        # Table 1 over three runs: runs 2-3 of Direct and JBFC for both models, then the mean
python scripts/analyze_results.py results/table2_gpt-5-nano_r3   # the first run's numbers quoted in Results
python scripts/table1_repeat.py   # Table 1's means, run-to-run spread and paired intervals
```

The paper's runs used an Azure OpenAI deployment of `gpt-5-nano` behind an Azure API Management gateway. Jev went through the same gateway until it stopped serving it; later requests (repairs 8 to 10 in Figure 2, the third repair of the other methods, JBFC with `gpt-5.4-mini`, and runs 2 and 3 of Table 1) call TypeSafe directly with `TYPESAFE_API_KEY` (same model; recorded in each run's manifest). See [`docs/apim-gateway.md`](docs/apim-gateway.md). With `WORKERS=8`, Table 2 took about two hours and $2.4 at list prices, extending it to ten repairs about 30 minutes and $1.4, the `gpt-5.4-mini` Direct run (`make compare`) about 8 minutes and $2.2, `make r3` about 26 minutes and $3.6, and `make repeat` about 47 minutes and $12. Completed records are reused when a run resumes. Run notes and checks are in [`docs/validation.md`](docs/validation.md).

<details>
<summary><b>Repository layout</b></summary>

```text
src/jbfc/      the library: JBFC, providers, Jev, tools
src/jev_fc/    experiment code: repair loop, Jev client, BFCL data and scoring, CLI (jev-fc)
scripts/       table runs, run extension for Figure 2, analysis, paper build and checks
paper/         manuscript and PDF
tests/         offline tests
docs/          benchmark, gateway, literature and validation notes; README images
```

</details>

## Citation

```bibtex
@misc{yu2026jbfc,
  title  = {Jev-Backed Function Calling: Agentic In-Context Learning with Cumulative Decision Feedback},
  author = {Yu, Minhyeong},
  year   = {2026},
  note   = {Enhans}
}
```

## License

[MIT](LICENSE). BFCL data are not part of this repository: `make prepare` downloads them from the pinned BFCL revision, under BFCL's own license.

## Acknowledgements

This work was created as part of the No-More-Slops group activities at Pseudo Lab. Jev is TypeSafe's verifier model; the benchmark is the Berkeley Function Calling Leaderboard (BFCL).
