# Jev source verification

Checked against first-party TypeSafe documentation on **2026-10-15 and 2026-10-16 (Asia/Seoul)**.
These are source notes, not experimental findings. No paid API request was made while researching.

## Identity and scope

Jev is TypeSafe AI's flagship **System One** model. It consumes text or structured state and typed questions and returns constrained decisions and probability values, rather than free-form explanations. The launch announcement attributes a new architecture, parallel sampler, and *Reinforcement Learning for Calibrated Decisions (RLCD)* to the vendor. Treat these as vendor descriptions, not independently reproduced research claims.

The official launch announcement is **Diogo Almeida (2026), “Introducing System One Models & Jev,” September 15**:
https://typesafe.ai/blog/introducing-system-one-models-and-jev

No first-party technical paper or peer-reviewed RLCD specification was identified in the official announcement or documentation consulted. Do not manufacture a Jev paper citation. Search results include third-party sites that resemble official Jev properties; the integration here uses only `typesafe.ai` and its subdomains.

## Verified HTTP contract

- Endpoint: `POST https://api.typesafe.ai/v1/systemone`
- Headers: `Authorization: Bearer <API_KEY>` and `Content-Type: application/json`
- Credential environment variable used by official SDKs: `TYPESAFE_API_KEY`
- Required request fields: `model`, `state`, and `questions`.
- `state` accepts a string, JSON object, or array.
- `questions` is a map of caller-chosen IDs to question objects. IDs are not used for inference, so each instruction must be self-contained.
- A Noul question is `{"type": "noul", "instructions": "a yes/no question", "criteria": {"true": "what yes means", "false": "what no means"}}`. `criteria` is optional; instructions and both criteria should have aligned polarity.
- Response shape: `{"model": "jev-1.13.0", "answers": {"question_id": {"type": "noul", "noul": 0.95}}, "usage": {"input_tokens": 296, "output_tokens": 20}}`.
- **Noul returns a yes probability, not a generated rationale and not a separate confidence field.** Near 0.5 indicates uncertainty. Choice and Score have additional confidence fields derived from their probability distributions. Do not confuse these interfaces.
- Multiple questions are evaluated in parallel and in isolation against the same state. This does not imply statistical independence of their errors, and multiplying their probabilities is not a justified joint-correctness estimate.

Primary references:
https://docs.typesafe.ai/api
https://docs.typesafe.ai/primitives/noul
https://docs.typesafe.ai/confidence
https://docs.typesafe.ai/introduction

Rechecked on 2026-10-17: the first-party Noul page defines its output as the probability that the answer is yes and explicitly states that it has no separate `confidence` field. The Confidence page gives an optional derived measure `abs(2*p - 1)` for Noul and the normalized maximum-option measure for Choice documented below. These API definitions should cite TypeSafe's documentation directly, rather than JEV-as-a-Judge. The initial experiment used threshold `0.8`; the current DMGFC manuscript and public library use `0.5`, selected after inspecting the BFCL results on 2026-10-18. Neither value is a fixed vendor recommendation. JEV-as-a-Judge's maximum-verdict probability is a separate study-specific measure.

## Model pin and cost

The model page on the access date lists:

| Field | Verified value |
| --- | --- |
| Stable pinned ID | `jev-1.13.0` |
| New preview ID | `jev-1.14.0` (released October 14, 2026) |
| `jev-latest` alias | `jev-1.13.0` |
| `jev-preview` alias | `jev-1.14.0` |
| Input price | USD 0.042 per million input tokens (USD 42 per billion) |
| Output price | Free |
| Default context | 64k per request, 32k state-plus-one-question |

Aliases move. Experiments must record the requested and returned model IDs. The accompanying harness pins 1.13.0 by default, makes the setting configurable, and records usage and wall-clock latency. Price is a dated configuration value, not a permanently guaranteed rate. The 1.14 announcement says `jev-latest` is expected to move after one week; use a version ID for reproducibility.

https://docs.typesafe.ai/models
https://typesafe.ai/blog/jev-1-14

## Errors, retries, and SDK

Official errors include 400 (invalid request/context), 401 (credentials), 413 (body exceeds 32 MiB), 415 (encoding), 422 (schema), 429 (rate limit), and 529 (overload). Official SDKs retry rate limits and overload using backoff and honor `Retry-After` when supplied. A verifier API failure must not be interpreted as a successful semantic check.

The official Python distribution is `typesafe-sdk`; import `TypeSafeClient` and `Noul` from `typesafe_sdk`. The first-party SDK accepts `client.system_one(state=..., questions=..., model=...)`; use `.answers[id].noul` and `.usage.input_tokens`. This repository uses the documented HTTP wire contract via `httpx` to keep the transport mockable without API keys.

https://docs.typesafe.ai/sdk/python
https://docs.typesafe.ai/sdk/python/api/types/responses
https://docs.typesafe.ai/sdk/python/api/retries

## Method design consequences for DMGFC

1. Jev receives the original conversation, available tool schemas, and current proposed calls. The generating LLM also receives actual prior feedback during repairs. Never send reference answers, evaluator labels, or benchmark ground truth to either model.
2. For a proposed call, ask separate function-selection and argument-correctness questions in one request. Accept only when both pass their respective probability thresholds **and** deterministic schema validation passes. For an empty call list, send only the function question and omit the argument threshold. Record the argument check as skipped, with no verdict or probability, rather than assigning a yes answer.
3. Function selection must check missing, unnecessary, and wrong functions, including whether an empty call set is appropriate. Argument verification is conditional on the selected functions and must check parameter roles and evidence, not merely JSON validity.
4. Return structured feedback containing failed dimension IDs, probabilities, thresholds, and deterministic schema errors. Jev does not produce a natural-language explanation; do not attribute a fabricated explanation to it.
5. Thresholds require validation. Vendor claims of calibration or type safety do not establish semantic correctness on function calling. Do not call this method zero-hallucination or formally safe.
6. Arithmetic, exact formats, type checks, and required-field checks should be deterministic where possible. Jev's published jaggedness includes numeric precision, temporal comparison, indirection, long distractor context, and adversarial content.
7. Keep the initial conversation unchanged. For each repair, serialize every prior proposal and its actual feedback in chronological order into one added user-context message before the original conversation. Preserve all original messages and mark prior proposals as NOT executed. The history informs the next proposal but never substitutes for checking that new proposal again.

https://docs.typesafe.ai/model-jaggedness/jev-1.13
https://docs.typesafe.ai/concepts/system-one

## Directly relevant existing TypeSafe recipes

- **Function calling**: converts natural-language trading requests to ordinary typed functions, using closed-set choices and Noul questions for presence/membership. It uses Jev as the router and argument selector directly, with minimum per-decision confidence; it is not the same-generator bounded repair method proposed here. https://docs.typesafe.ai/cookbooks/function_calling
- **SDE cascade**: a mini generative model extracts a record; Jev performs atomic per-field checks; an escalation gate sends flagged records to a stronger reasoning model. The page explicitly labels its performance figures as internal TypeSafe results. This is a close conceptual predecessor, so the paper must acknowledge it rather than claim novelty for verification cascades in general. https://docs.typesafe.ai/cookbooks/sde_cascade
- **Guardrails for LLMs**: batches multiple hazard checks and routes outputs by thresholds. https://docs.typesafe.ai/cookbooks/llm_guardrails

The defensible contribution is an auditable evaluation of **dual semantic checks, conjunctive acceptance, and bounded repair with cumulative feedback before the original request**, including generator-matched retry controls, component ablations, error transitions, overhead, and no ground-truth leakage. Completed measurements are reported in the [current manuscript](../paper/main.pdf) and [README results](../README.md#results): effects differ across generators and benchmarks, with one Jev verifier. These source notes do not independently validate those measurements.

## Additional documented Jev capabilities (outside this experiment)

The following capability notes were checked on 2026-10-16. They describe the official API and examples, not an implemented comparison arm in this project. **DMGFC** uses the LLM to propose functions and arguments and Jev to check them with two Noul questions when a call is present; no-call receives only the function question.

### Choice: options, probabilities, and confidence

The official `Choice` primitive selects one item from a finite set. The HTTP question type is `choice`, not `enum`. Python `Literal` or enum-style alternatives appear as keys in the question's `criteria` map. Each value may be a string, object, array, or `null`. Question IDs are not used for inference; option names and descriptions are shown to the model. The same documented `POST /v1/systemone` endpoint accepts these questions.

The response includes the chosen key, a probability for every option, and `confidence`. The selected key has a maximum probability; probabilities sum to one. These are output-format and probability properties, not evidence that the chosen answer is correct. The official Python SDK represents the question with `Choice(instructions=..., criteria=...)` and exposes the result as `response.answers[question_id].choice`.

Choice accepts at most **255 options**. The guide recommends an `other` or `none of the above` option when the listed choices do not cover every input. One Choice question selects one option; it does not by itself produce an unrestricted set of parallel function calls.

For `n > 1` options, the documentation defines:

```text
confidence = (max(probabilities) - 1/n) / (1 - 1/n)
```

This is different from both the winning option's probability and Noul's affirmative probability. For example, distributions `(0.6, 0.3, 0.1)` and `(0.6, 0.2, 0.2)` both yield confidence `0.4`. A Noul correctness threshold such as `0.8` cannot be interpreted as the same Choice-confidence threshold.

The vendor's Jev 1.13 jaggedness page says that option order can affect a Choice answer and that the model sometimes favors the first option. This is a documented limitation of that interface, not a measured result of the DMGFC experiment.

Sources:
https://docs.typesafe.ai/primitives/choice
https://docs.typesafe.ai/api
https://docs.typesafe.ai/sdk/python/api/types/questions
https://docs.typesafe.ai/confidence
https://docs.typesafe.ai/model-jaggedness/jev-1.13

### Function and argument selection in official examples

The official function-calling cookbook already uses Jev for routing and for selecting argument values from bounded alternatives. It uses one Choice question for a `Literal`, Noul membership questions for members of a `list[Literal[...]]`, and Noul questions for booleans and optional-field presence. In that example, unrestricted values such as the integer `limit` receive no question and retain their function defaults. Other recipes also construct values from bounded choices.

Do **not** claim that Jev cannot produce argument values in general. The Jev 1.13 jaggedness page says it is not trained to generate text and recommends getting candidate extractions from a generative model or regex before asking Jev to choose among them. This supports a narrower description of **DMGFC**: the LLM proposes function names and argument values from the request and supplied schemas, while Jev checks the current proposal.

The experiment does not use Choice for direct routing or fixed-option argument selection. Its repair context contains the accumulated prior calls and actual verifier feedback before the unchanged original request. Both function and argument choices can change in a new proposal. Each proposal receives all applicable correctness questions again; no-call skips the argument question, while a selected function with missing required arguments still receives it.

Sources:
https://docs.typesafe.ai/cookbooks/function_calling
https://docs.typesafe.ai/model-jaggedness/jev-1.13
https://docs.typesafe.ai/concepts/system-one
