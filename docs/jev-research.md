# Jev source verification

Checked against first-party TypeSafe documentation on **2026-10-15 (Asia/Seoul)**.
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

## Method design consequences

1. Send only original conversation, available tool schemas, and proposed calls. Never send reference answers, evaluator labels, or benchmark ground truth to Jev or the generating LLM.
2. Ask separate function-selection and argument-correctness questions in one request. Accept only when both pass their respective probability thresholds **and** deterministic schema validation passes.
3. Function selection must check missing, unnecessary, and wrong functions, including whether an empty call set is appropriate. Argument verification is conditional on the selected functions and must check parameter roles and evidence, not merely JSON validity.
4. Return structured feedback containing failed dimension IDs, probabilities, thresholds, and deterministic schema errors. Jev does not produce a natural-language explanation; do not attribute a fabricated explanation to it.
5. Thresholds require validation. Vendor claims of calibration or type safety do not establish semantic correctness on function calling. Do not call this method zero-hallucination or formally safe.
6. Arithmetic, exact formats, type checks, and required-field checks should be deterministic where possible. Jev's published jaggedness includes numeric precision, temporal comparison, indirection, long distractor context, and adversarial content.

https://docs.typesafe.ai/model-jaggedness/jev-1.13
https://docs.typesafe.ai/concepts/system-one

## Directly relevant existing TypeSafe recipes

- **Function calling**: converts natural-language trading requests to ordinary typed functions, using closed-set choices and Noul questions for presence/membership. It uses Jev as the router and argument selector directly, with minimum per-decision confidence; it is not the same-generator bounded repair method proposed here. https://docs.typesafe.ai/cookbooks/function_calling
- **SDE cascade**: a mini generative model extracts a record; Jev performs atomic per-field checks; an escalation gate sends flagged records to a stronger reasoning model. The page explicitly labels its performance figures as internal TypeSafe results. This is a close conceptual predecessor, so the paper must acknowledge it rather than claim novelty for verification cascades in general. https://docs.typesafe.ai/cookbooks/sde_cascade
- **Guardrails for LLMs**: batches multiple hazard checks and routes outputs by thresholds. https://docs.typesafe.ai/cookbooks/llm_guardrails

The defensible contribution is an auditable evaluation of **dual semantic checks, conjunctive acceptance, and bounded repair with the same function-calling generator**, including generator-matched retry controls, component ablations, error transitions, overhead, and no ground-truth leakage. Empirical benefits remain hypotheses until experiments run.
