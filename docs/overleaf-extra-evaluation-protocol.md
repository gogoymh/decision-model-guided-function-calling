# Overleaf follow-up evaluation protocol

Status: **pending model experiments**. Primary-source metadata, upstream data counts, schemas, and adaptation risks were inspected offline on 2026-10-20. No new generator/verifier request or benchmark tool execution was made for this protocol. Preparation of an input manifest does not constitute an experimental result.

## Recommended second benchmark

Use **API-Bank given-description first-call prediction**, clearly labeled as an adaptation. It is the closest of the inspected benchmarks to this paper's decision before executing a call. Keep BFCL as the main benchmark for no-call decisions; the proposed API-Bank adaptation has no validated no-call negatives.

| Candidate | Native evaluation | Fit to the current paper |
|---|---|---|
| API-Bank | Dialogue prefixes followed by API calls; official evaluator parses and executes the predicted call, then invokes a tool-specific correctness checker. | Extract the first API turn per dialogue. Its preceding context contains no API observation, and exactly one next call is predicted. Use static gold name/argument scoring, not the native execution-based score. |
| StableToolBench | Multi-step tool trajectories using cached or simulated API responses; Solvable Pass Rate/Win Rate assess completed answers with an LLM judge. | Useful later for end-to-end tool-use reliability. It does not directly supply the same single-call gold argument target or measure the same outcome. Do not replace its full evaluation with first-call exact match and retain the benchmark's name without qualification. |
| τ-bench | Simulated user conversations, domain policy adherence, database-state goals, repeated-run consistency. | Requires a larger extension to interactive execution and user simulation. The original official repository now warns that its tasks are outdated and points to the newer τ³/τ² repository. Defer this to a separate study. |

Sources: [API-Bank paper](https://aclanthology.org/2023.emnlp-main.187/), [API-Bank official README](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/README.md), [API-Bank evaluator](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/evaluator.py), [StableToolBench official README](https://github.com/THUNLP-MT/StableToolBench/blob/aa4ed9f4737ad98bd706663f01d63623c3427812/README.md), [τ-bench official README](https://github.com/sierra-research/tau-bench).

## Pinned API-Bank data and exact counts

Repository: `AlibabaResearch/DAMO-ConvAI`, revision **`faf754671f5b850e1b4d1ff870db37a6a2480df6`**.

The paper describes the complete benchmark as 314 dialogues, 753 calls, and 73 tools. Those aggregate numbers are **not the size of this adaptation**. The inspected path `api-bank/lv1-lv2-samples/level-1-given-desc/` contains:

- 214 JSONL dialogue files and 389 API turns.
- 212 dialogues with at least one API turn; use only their first API turn.
- 91 dialogues containing exactly one API turn. “First-call prediction” must not be described as all source dialogues being single-turn or single-call.
- Two dialogues without an API turn: `RegisterUser-ForgotPassword-DeleteAccount-GetUserToken-level-2-1.jsonl` and `SearchEngine-level-1-1.jsonl`. Exclude them; absence of an API annotation is not evidence of a gold no-call decision.
- Eight first-call gold/schema conflicts, listed below. Excluding them before inference yields **204 scored examples**.

| Excluded source files | Conflict |
|---|---|
| `CancelTimedSwitch-level-1-1.jsonl` through `CancelTimedSwitch-level-1-5.jsonl` | Gold uses `device_id`; the official class schema and `call()` require `name`. |
| `TimedSwitch-level-1-1.jsonl` through `TimedSwitch-level-1-3.jsonl` | Gold uses `device_id`; the official class schema and `call()` require `name`. |

Do not silently rename these labels. Save exclusions and their source hashes in the manifest. The six `GetToday` first-call records have `result.input: null` but a valid empty `param_dict`; obtain argument labels from `param_dict`, not `result.input`.

Data fields are `role`, `text` for `User`/`AI`; API records contain `role`, `api_name`, `param_dict`, and `result`. The `result` contains `input`, `output`, `exception`, and sometimes `api_name`. No `result` or target record may enter the generator/verifier prompt.

## Input preparation without future-label leakage

1. Enumerate source filenames in sorted order. Parse JSONL and locate the first `role == "API"` record. Keep only `history[:first_api_index]` as context. Preserve the dialogue's original message order; assert that it contains no API observation.
2. Build a **fixed, sorted catalog of 53 API subclasses** from `api-bank/apis/*.py`. Extract class names, `description`, `input_parameters`, and `output_parameters` through Python AST literals without importing modules, initializing databases, or calling tools. The catalog must be identical for all examples.
3. Do not use the official `Sample.apis` inventory unchanged: `Sample.from_chat_history` derives that inventory from **all API names appearing anywhere in the gold dialogue**, including future turns. Supplying that oracle inventory makes the task easier and exposes gold-derived information. The fixed full catalog avoids this issue, and the paper must state this adaptation.
4. Translate declared types into JSON Schema. Derive structural required parameters from nondefault named arguments of `call()` excluding `self`. Do not mark every listed parameter required: for example `ForgotPassword.call(status, **kwargs)` has only `status` structurally required, with conditional arguments described in the source. Record any separate local validation rule.
5. Normalize gold `param_dict` by declared type before scoring: preserve strings; parse integer strings as integers; map only exact `True`/`False` to booleans; safely parse `list`/`list(str)` string literals with `ast.literal_eval`, verify the expected list shape, and serialize as JSON. Do not execute model-generated text. Hash the normalization code and record its version.
6. Export model-visible input (`messages`, `tools`) separately from labels/metadata (`id`, source file, API index, split, ground truth). Assert that gold calls, result objects, exclusion reasons, IDs, and filenames are absent from prompts.

Schema/coercion sources: [ToolManager](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/tool_manager.py), [ForgotPassword](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/apis/forgot_password.py), [TimedSwitch](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/apis/timed_switch.py), [CancelTimedSwitch](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/apis/cancel_timed_switch.py).

## Immutable development/test manifest

Group near-related dialogues by `re.sub(r"-\d+$", "", Path(filename).stem)`, keeping variants such as `...-level-1-1` and `...-level-1-2` in one split. Assign a group to development when:

```python
int(hashlib.sha256(("apibank-firstcall-v1|" + group).encode()).hexdigest(), 16) % 5 == 0
```

All other groups are test. Applied to the **204 eligible** examples, this produces **48 development and 156 test** examples. These are a declared adaptation split, not official API-Bank split counts. Do not alter group rules after viewing outcomes.

Save a manifest with the upstream commit, source URLs, SHA-256 of every input/schema/license file, preparation-code digest, sorted eligible and excluded IDs, split rule/version, catalog hash, normalized JSONL hashes, count by split and API name, evaluation metric version, and UTC preparation time. Commit the manifest before model inference. Existing BFCL manifests and recorded trajectories remain immutable; new split-based experiments use new run directories.

The repository root has MIT text, but **`api-bank/LICENSE` is Apache-2.0**. The author's Hugging Face dataset card declares MIT. For the pinned GitHub subtree used here, preserve its Apache-2.0 license and attribution rather than making a blanket MIT statement. Record path-specific license provenance; do not add raw dataset files to a public release without their notices. Sources: [API-Bank license](https://github.com/AlibabaResearch/DAMO-ConvAI/blob/faf754671f5b850e1b4d1ff870db37a6a2480df6/api-bank/LICENSE), [author dataset card](https://huggingface.co/datasets/liminghao1630/API-Bank/blob/main/README.md).

## Predeclared experiment matrix

First evaluate Direct and full DMFC on BFCL and the adapted API-Bank test set, then the same F/A/L ablations where budget permits. Preserve one fixed generator/verifier configuration per comparison. For each model and repetition, generate one initial candidate per case and share it among methods. Different generator families must generate their own initial proposals.

- Repair budget `r = 3`: at most four proposals. Stop at the first passing candidate; use the same final fallback policy for applicable methods. No benchmark functions execute.
- For this transfer experiment, freeze the historical BFCL settings before any additional inference: r=3, both thresholds=0.5, keep_first. The 48 API-Bank development cases remain unused for outcome tuning; evaluate all 156 prespecified test cases without changing these settings. BFCL settings were selected after inspecting the original BFCL examples, so its historical and added-generator comparisons are exploratory, not held-out tuning.
- Run **three independent repetitions** with new initial proposals, immutable configs, and a separately indexed initial-proposal cache for each repetition. Never drop a repetition after seeing results. Report mean/sample standard deviation and paired case-level bootstrap intervals; use a fixed bootstrap seed and store it.
- API-Bank adaptation metrics: exact single-call accuracy (one predicted call with matching API name and normalized argument object), function accuracy, arguments conditional on correct function, empty/extra-call rate, schema-error rate, repair count, acceptance versus correctness, fixed/introduced errors, tokens, cost, and mean/p50/p95 time. Clearly label exact match as an adaptation: it can penalize semantically equivalent arguments and is not native API-Bank execution correctness.
- There are no gold no-call cases in this adaptation. Report its empty-call rate as an error rate, and evaluate unnecessary-call prevention on BFCL irrelevance cases separately.
- Measure API processing duration and full wall time separately. Report retry waits, tool execution exclusion, provider caching, concurrency, tokenizer/runtime/hardware, and serving route. For open-weight models, leave cost unavailable unless an explicit measured compute-cost basis exists.

## Additional generator families

Generator generalization is different from changing the decision/verifier model. Keep Jev fixed for the generator-family comparison; for verifier-family comparison, keep the generator and initial proposals fixed.

| Candidate | Verified official details | Follow-up interface |
|---|---|---|
| **Claude Sonnet 5.5**, `claude-sonnet-5-5` | Official overview: released September 28, 2026; active; 1M-token context; max output 128K; base input/output prices $2/$10 per million tokens. Cache read $0.10/Mtok; 5-minute/1-hour write $2.50/$4.00/Mtok. These are source prices, not measured study costs. | Native Messages API, user-defined client tools `{name, description, input_schema}`, `tool_choice={"type":"auto","disable_parallel_tool_use":true}`. Parse `tool_use` blocks and stop before execution. Forced `any`/specific-tool choice is unsupported on Sonnet 5.5. Do not register server tools or send `tool_result`. Verify actual account/route capabilities before running. |
| **Qwen3-4B-Instruct-2507**, `Qwen/Qwen3-4B-Instruct-2507` | Official model card: 4.0B parameters, only non-thinking mode, 262,144 native context, Apache-2.0 weights. Inspected Hub revision `cdbee75f17c01a7cc42f958dc650907174af0554`. | Use a separately pinned local runtime and a common structured-call prompt/parser. The official card describes OpenAI-compatible SGLang/vLLM serving. Avoid an agent runner that automatically executes tools. Record checkpoint, quantization, sampling, template, runtime, and hardware. |

Use a common 4,096-token proposal output cap for the main comparison if both interfaces support it, explicitly recording that this differs from Qwen's published recommended generation settings. Keep each provider's actual thinking/effort configuration in the manifest: `reasoning_effort="none"` is a GPT setting, not a universal provider setting. The Claude adapter sends no thinking/effort override; record this provider default and do not claim that it matches GPT reasoning_effort=none.

An APIM subscription key or an official model ID does not prove that a gateway deployment exists. Record the confirmed deployment/API route, API version, emitted model identity, and supported request shape before inference. Do not forward provider credentials through an unrelated gateway route.

Sources: [Claude Sonnet 5.5 official overview](https://platform.claude.com/docs/en/models/sonnet-5-5/overview), [Claude client tool protocol](https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview), [Claude Models API](https://platform.claude.com/docs/en/api/models/list), [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507/blob/cdbee75f17c01a7cc42f958dc650907174af0554/README.md).

## Connecting the three motivation claims to evidence

1. **Structure versus request alignment**: count schema-valid incorrect calls and unnecessary calls on BFCL. Compare Direct, schema-only, and full DMFC on matched cases. Cite prior tool-use work accurately; Toolformer learns API selection/timing/arguments and is not only a format validator.
2. **Function and argument errors are distinct**: use F/A ablations plus a paired error decomposition. Conditional argument scores have different denominators when function selection changes, so they cannot establish superiority in full-call correctness by themselves.
3. **Cumulative feedback versus latest only**: compare DMFC with DMFC-L under identical prompts apart from retained history, thresholds, initial candidates, and budget. Differences arise at the second repair; `r <= 1` cannot establish the value of cumulative history. Supplement accuracy with token/time overhead. Do not state that earlier work always discards history; this is a specific ablation question.

The added benchmark and generator arms stay **planned/pending** in author notes until complete validated runs exist. Only then update the abstract/Introduction's benchmark list, multi-model claims, and quantitative summary.
