# Native Perplexity Decider experiments on DGX Spark

The added adapter is an explicit alternative verifier for new experiments. The
default `dmgfc.Jev` and the frozen `src/jev_fc` implementation remain unchanged.
Existing result IDs, dataset files, traces and initial-proposal caches are not
renamed or rewritten. No additional benchmark performance is claimed here.

## Pinned model and access

Use the table's **Perplexity Decider v1.1 27B**, not an unmodified causal Qwen model.

| Item | Value |
|---|---|
| Official artifact | [perplexity-ai/pplx-decider-v1.1-27b](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b) |
| Immutable revision | `14c9d90137962d300f3de95633f7b6535e80c5ce` |
| Base artifact | `Qwen/Qwen3.8-27B`, revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| License | Apache 2.0; retain the artifact's LICENSE and NOTICE |
| Weight files | 12 safetensors files, including the separate readout |
| Weight bytes | 52,173,413,928 bytes, approximately 48.59 GiB |
| Attention | Saved `noncausal_full_attention`; native linear-attention behavior retained |
| Readout | 255-option BF16 decision head, not a full-vocabulary generation head |
| Calibration temperature | `1.0087417621345625`, applied once by native inference |
| Native input limit | 8,192 tokens per prepared question branch; overflow is rejected |

Hugging Face metadata reported `private=false` and `gated=false`, and an
unauthenticated HEAD request to a pinned weight file returned HTTP 200 on
October 24, 2026. The README still contains private/authentication wording; the
actual public endpoint was checked separately. No Hugging Face token is needed
for this verified revision. If access changes, report that condition rather than
substituting another model.

The photographed table's latency is measured on a dedicated H100 for local
models and a hosted round trip for Jev. It is not a DGX Spark measurement.
Measure time on the available GB10 device before comparing deployments.

## Native runtime

The official source is bundled in `source/src/autojev/` at the same model
revision. Use it unchanged for the model computation. Generic causal vLLM chat
serving would change this checkpoint's attention and readout behavior.

The official [requirements](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/requirements.txt)
are pinned as follows:

```text
Python >=3.12
torch==2.14.0
torchvision==0.29.0
transformers==5.17.0
accelerate==1.15.0
pillow==12.3.0
fastapi==0.141.1
uvicorn==0.52.4
httpx==0.28.1
python-dotenv==1.2.3
datasets==5.0.1
matplotlib==3.11.2
kernels==0.16.1
flash-linear-attention==0.5.2
safetensors==0.8.0
numpy==2.5.3
huggingface-hub==1.31.0
imageio-ffmpeg>=0.6
```

The training/data/plotting dependencies are not needed by the client adapter.
The native loader needs Torch, Transformers, processor dependencies and the
checkpoint's model code. The server additionally needs FastAPI and Uvicorn.
Do not install these into unrelated running environments.

Public distribution metadata confirms Python 3.12 aarch64 wheels for Torch
2.14.0 with CUDA 13.0, Torchvision 0.29.0 and Triton 3.8.0. FLA 0.5.2 depends on
`fla-core==0.5.2`; both have universal Python distributions. Availability of
packages does not by itself prove GPU inference or fast kernels work: record the
installed package versions, CUDA availability, GPU architecture, kernel warnings
and actual smoke result on the target device.

Download the checkpoint with `snapshot_download` using the full immutable
revision, preserve LICENSE/NOTICE and verify all expected file sizes/checksums.
The available DGX's existing shared cache and services must remain intact.

The prepared DGX has an isolated Python 3.12 runtime at
`/home/enhans/.local/share/decision-model-guided-function-calling`.
Its launcher verifies the checkpoint identity and uses the native server wrapper
with the required `/provenance` route:

```bash
/home/enhans/.local/share/decision-model-guided-function-calling/start-native.sh
```

Use `stop-native.sh` at the same location to stop this service. These scripts
preserve other listeners and containers. The launch survives SSH disconnection;
automatic startup after a reboot is not configured. The runtime's `README.md`,
`serve_native.py`, `runtime-freeze.txt`, `model-download-receipt.json`, and
`runtime-verification.json` record the actual installation and source hashes.
Initial weight transfer takes several minutes on this device; wait for `/health`
before sending model requests.

The launcher also points Triton to Python 3.12 development headers extracted from
the official Ubuntu ARM64 package into this runtime. No host packages were
changed. NVIDIA's ARM64 cuSPARSELt 0.8.1 wheel has an upstream `sbsa` metadata tag
that produces a `pip check` warning; its actual library is AArch64 and the GPU
BF16 check passed. This exact exception is recorded in the runtime verification;
the upstream package and native inference code are preserved.

For the strict adapter, the deployment wrapper adds a read-only `/provenance`
route to the same loaded service. This route must describe the actual checkpoint:

```json
{
  "model_id": "perplexity-ai/pplx-decider-v1.1-27b",
  "revision": "14c9d90137962d300f3de95633f7b6535e80c5ce",
  "native_attention": "noncausal_full_attention",
  "temperature": 1.0087417621345625,
  "max_input_tokens": 8192,
  "inference_batch_size": 8,
  "one_branch_per_question": true
}
```

This metadata must come from the deployment that verified and loaded the pinned
checkpoint. It is not a client override to relabel an arbitrary service. A
different model, revision, attention mode or temperature stops verification.
If batch metadata is unavailable, forward-call accounting stays `null`.

The native server uses an inherited `autojev-qwen3.8-27b` name and accepts
`jev-latest` in its request format. The adapter verifies provenance and records
the actual PPLX artifact separately from those wire identifiers. These runs must
not be labelled as hosted Jev or AutoJev results. The model snapshot omits the
playground HTML referenced by `/`; use `/health` and `/v1/systemone` for checks.

Bind the service to localhost. From the Mac, use an SSH tunnel rather than
opening a public unencrypted endpoint:

```bash
ssh -N -L 18081:127.0.0.1:18081 dgx_spark
```

## Explicit adapter usage and bounded smoke

```python
import dmgfc

with dmgfc.PerplexityDecider(base_url="http://127.0.0.1:18081") as decider:
    agent = dmgfc.DMGFC(generator, tools=tools, jev=decider)
    result = agent.call("The user's request")  # no application tool execution
```

The adapter reuses the exact frozen function-selection and argument-correctness
questions. A no-call candidate receives only the function-selection question.
A selected function with missing arguments still receives both questions and
must also pass deterministic schema checks. Inputs contain only observable
messages, tool definitions and proposed calls. Dataset IDs, categories, ground
truth and offline scores are not added to model state.

Run the fictional protocol smoke from the repository on the DGX using its
prepared Python environment. For a Mac client, use a local Python environment
with this repository's dependencies and the SSH tunnel described above.

```bash
/home/enhans/.local/share/decision-model-guided-function-calling/venv/bin/python scripts/dgx_decider_smoke.py \
  --base-url http://127.0.0.1:18081 \
  --output results/dgx-decider-smoke/receipt-001.json
```

The smoke sends eight POSTs with zero HTTP retries and no paid generator calls
or tool execution. Five toy checks are followed by a scripted missing-argument
candidate, a repair with the requested city and a no-call trajectory. The three
scripted proposals are labelled separately from API generations. Expected toy
labels stay outside the requests. The receipt includes
probabilities, actual model provenance, source/configuration fingerprints and
measured counters. It explicitly says `synthetic=true`; it is not an accuracy
benchmark or a paper result. The runner never replaces an existing receipt.
Optional service authentication uses the explicitly named
`DGX_DECIDER_API_KEY` environment variable; keys are excluded from fingerprints,
receipts and error text. No `.env` file is loaded.

Context overflow, an unavailable service, invalid probability, mismatched model
or malformed response raises a sanitized `DeciderError`. Such failures are not
converted to probability zero or a semantic rejection. Preserve failed attempts
and decide their benchmark treatment before a full run.

## Accounting

| Receipt field | Meaning |
|---|---|
| `api_calls` | Actual `/v1/systemone` POST attempts, including retries |
| `control_api_calls` | Separate provenance GETs; these are not model decisions |
| `decision_http_time_s` | Measured sum of decision HTTP attempt durations |
| `control_http_time_s` | Measured provenance HTTP duration |
| `latency_s` | Frozen loop-compatible verification duration, excluding retry sleep; includes schema/provenance work |
| `wall_time_s` | End-to-end adapter verification wall time including retry waits |
| `usage.input_tokens` | Native prepared branch tokens reported by the server |
| `usage.output_tokens` | Reported generated tokens; native decision service reports zero |
| `question_evaluations` | Number of question branches in the successful request |
| `successful_forward_batches` | Successful batched model invocations if the native batch protocol is verified |
| `forward_calls` | Total forward invocations only when known; retries make it unknown |
| `successful_response_usage` | Successful response counters, separate from unknown failed-attempt usage |
| `estimated_usd` | `null`; hardware/electricity cost has not been measured |

The official server creates one row per question and invokes the model on batches
of at most eight rows. Two correctness questions therefore mean two question
branches but one batched model invocation. A no-call question also uses one
invocation. Token counts include the prompt repeated in separate branches.
Missing counters stay `null`. With a retry, total usage and total forward calls
stay `null` because a failed HTTP attempt may already have performed computation.
The successful-response usage remains available separately. Do not equate API
request count, number of questions, output tokens and model forward calls.

## Prepare paired benchmark inputs without new API calls

Datasets and historical caches are not present in a fresh Git clone when they
were ignored in the original checkout. Bring over only the selected prepared
dataset, its existing initial cache, the matching generator manifest and the
pinned evaluator. Keep their bytes unchanged and avoid copying credentials.

```bash
/home/enhans/.local/share/decision-model-guided-function-calling/venv/bin/python scripts/prepare_dgx_decider_experiment.py \
  --dataset <prepared-dataset.jsonl> \
  --initials <matching-initial.jsonl> \
  --generator-manifest <source-manifest.json> \
  --source-records <source-records.jsonl> \
  --base-url http://127.0.0.1:18081 \
  --output results/dgx-decider-preparation/plan-001.json
```

The preparation script sends no network requests. It hashes the real files,
canonical `src/dmgfc` implementation, compatibility wrappers, frozen experiment
sources, model identity and each cached initial proposal. It
rejects duplicate/foreign IDs, partial replicates, mismatched cache config IDs
and a dataset that differs from the source manifest's declared digest. Legacy
caches without per-row config IDs require the matching source records: every
cached proposal must equal the corresponding `trace[0].proposal`, and every
available source record config ID must match its manifest. No config fields are
invented or inserted into the old cache. Source
data and caches are never rewritten. The output is a **plan**, marked
`ready_for_benchmark=false`; this script fingerprints inputs and creates a
comparison plan. The runnable BFCL preflight below additionally checks the
audited dataset manifest, official evaluator and generator interface. A local
path is recorded as provenance, not treated as a model input.

## Run the separate BFCL experiment

`scripts/run_dgx_decider_experiment.py` implements a serial comparison of Direct
and the full function/argument gate using native Perplexity Decider. It reuses
the original first proposal for each selected example. The original cache and
records remain read-only; the new run has its own model identity, source hashes,
configuration fingerprint and output directory. Random-selection and multi-agent
arms are not included in this runner.

For the transferred DGX inputs, an offline preflight command is:

```bash
inputs="$HOME/.local/share/decision-model-guided-function-calling/inputs-20261024"
/home/enhans/.local/share/decision-model-guided-function-calling/venv/bin/python scripts/run_dgx_decider_experiment.py \
  --dataset "$inputs/data/bfcl/examples.jsonl" \
  --dataset-manifest "$inputs/data/bfcl/manifest.json" \
  --initials "$inputs/results/table1_gpt-5-nano_r2/initial.jsonl" \
  --generator-manifest "$inputs/results/table1_gpt-5-nano_r2/manifest.json" \
  --source-records "$inputs/results/table1_gpt-5-nano_r2/records.jsonl" \
  --checker "$inputs/data/bfcl/upstream/ast_checker.py" \
  --limit 5 --max-repairs 3 --max-new-generator-calls 15 \
  --output results/dgx-decider-bfcl/preflight-001
```

Preflight makes no model requests and does not need a credential. It uses the
source manifest's generator model, API interface, endpoint, output cap and
sampling settings. It records the new frozen source hash and the historical
initial-run hash separately; if they differ, the manifest states that fact.
The audited BFCL revision is
`6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`; the copied full dataset's SHA256 is
`4cad37e0a68506c8ce6369a454796a3695327eb860e9645430b120e5f3ef28e3`.

To actually run, supply the required generator credential in the process
environment, choose a **new** output directory and add `--execute` to the same
command. The supplied historical generator uses Azure API Management, requiring
`APIM_SUBSCRIPTION_KEY`. No key or `.env` was copied to the DGX by these scripts.
The default five-case run permits at most 15 new generator requests, from three
repairs per case; it does not generate new initial proposals. Both HTTP clients
use zero retries. Checkpoint provenance must pass before any paid repair request.
Changing the limit, repair count, generator or inputs changes the run identity.

The runner applies threshold 0.5 and `keep_first`: the first passing candidate is
final, or the original candidate is returned when the repair budget is exhausted.
Every attempted candidate stays in the trace. Both arms finish their model
operations for a case before the official BFCL evaluator runs, so gold-derived
scores cannot enter subsequent repair prompts. Application functions never run.

New files include `manifest.json`, `native-model-provenance.json`,
`repair-generations.jsonl`, `native-verifications.jsonl`, `records.jsonl` and
`summary.json`. Successful model outputs are written immediately, preserving
already-generated proposals if a later native request fails. Failures stop the
run and append sanitized metadata to `errors.jsonl`; response bodies and
credentials are omitted, and unknown failed-attempt token usage remains `null`.
There is no implicit resume or overwrite. Preserve an incomplete run and use a
separately reviewed continuation protocol before continuing it.

Standalone generator usage/time/API counts include the cached first proposal
once per arm. New generator counts cover repair requests only. Both canonical
`input_tokens/output_tokens` and legacy `prompt_tokens/completion_tokens` are
recognized, including reported cached-input tokens. Native usage/POST counts,
control GETs, question branches and forward batches stay separate. Execution
wall time excludes historical initial generation; its recorded latency is
reported separately. Local hardware and total dollar cost remain `null`.

## Verified preparation on October 24, 2026

The GB10 service passed actual native inference and remains available at
`127.0.0.1:18081`. The first two-question Noul request took 19.7168 seconds,
including initial kernel compilation. The subsequent fictional function,
argument, repair and no-call smoke completed eight POSTs, 13 question branches
and eight forward batches in 4.9864 seconds total. Scripted repair and no-call
paths were accepted, with no paid generator requests or tool execution.
These are protocol smoke measurements, not benchmark accuracy or representative
latency measurements.

All 48 checkpoint files were verified. The prepared 3,051-example BFCL inputs
and both original generator caches passed offline preflight with zero model API
calls. Existing DGX containers were preserved. The explicit live experiment
still needs `APIM_SUBSCRIPTION_KEY`; no credentials were copied to the device.

Before a full experiment, fix the generator models/settings, evaluator revision,
comparison arms, repair limits, repeated runs, concurrency and budget. Use the
same cached first proposal for every arm and charge its generation cost once to
each arm's standalone cost. Prespecify time/token/request comparisons as well as
iteration comparisons. The historical choice of `r=3`, threshold 0.5 and
`keep_first` followed inspection of those examples; do not describe those data
as an untouched confirmation set. Random-selection and multi-agent arms still
need separate implementation and protocol review. The bounded toy smoke and
offline preflight do not launch paid generators. The real runner requires
`--execute`; no paid experiment matrix was launched while implementing it.
