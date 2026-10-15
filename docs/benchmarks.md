# Benchmark protocol and provenance

The experiment uses a pinned subset of the Berkeley Function Calling Leaderboard
(BFCL v4), a widely used public function-calling benchmark. The target is a
single-turn proposal before execution, so the default suite includes Python
single-call tasks with one or several candidate tools and call/no-call decisions.
It excludes Java/JavaScript, parallel calls, multi-turn environments, web search,
and memory tasks. Report it as the **BFCL v4 single-turn Python subset**, not as
the full BFCL leaderboard aggregate.

## Version and preparation

Official source: <https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard>

Pinned revision: `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
Data source directory: `berkeley-function-call-leaderboard/bfcl_eval/data/`.
The `.json` source files contain one JSON object per line. The original prompts
and answer files, exact checker source, and upstream Apache-2.0 license are
downloaded together. Every downloaded file is checked against a SHA-256 digest
embedded in the preparation code. Cached files are rechecked, so subsequent runs
can be completely offline. Unknown revisions require an explicit code/pin review.
Normalized data are also rehashed on loading when their adjacent preparation
manifest is present, preventing silent use of accidentally edited labels.

```bash
python -m jev_fc.benchmarks --output-dir data/bfcl --seed 1729
```

This produces `examples.jsonl`, `dev.jsonl`, `test.jsonl`, `manifest.json`, the
original files under `raw/`, and the checker/license under `upstream/`. The
manifest records counts, split procedure, source URLs, revision, and hashes.
The counts below were measured from the pinned source, rather than copied from
upstream prose that may describe an older version.

| Category | Complete | Development | Test | Outcome checked |
|---|---:|---:|---:|---|
| `simple_python` | 400 | 40 | 360 | One function and its arguments |
| `multiple` | 200 | 20 | 180 | Select one function among candidates; arguments |
| `irrelevance` | 240 | 24 | 216 | Produce no call |
| `live_simple` | 258 | 25 | 233 | User-contributed single-function tasks |
| `live_multiple` | 1,053 | 105 | 948 | User-contributed candidate selection and arguments |
| `live_irrelevance` | 884 | 88 | 796 | User-contributed no-call tasks |
| `live_relevance` | 16 | 1 | 15 | Produce a call; argument correctness is **unlabelled** |
| **Total** | **3,051** | **303** | **2,748** | |

Within each category, examples are ranked by
`sha256(f"{seed}:{id}")`; 10% (rounded down, with a minimum of one) enter development and
the rest enter test. Changing input order does not change membership. All three
files retain the original IDs. The development split is the only split for
choosing verifier thresholds, prompts, or models; freeze these choices before
the test run. This is our experimental split of public benchmark data, not a
claim that BFCL provides a hidden held-out test set. Existing training or
pretraining contamination cannot be ruled out.

Four source annotation inconsistencies are recorded in the manifest and retained
unchanged in the primary denominator: `live_simple_106-63-0` and
`live_simple_112-68-0` give empty alternative sets for required array arguments;
`live_multiple_862-181-3` includes a gold `journey_start_time` parameter absent
from its schema; `live_multiple_964-207-0` similarly includes an absent `question`
parameter. The pinned checker cannot accept a call satisfying both those schemas
and gold alternatives. A separately labelled sensitivity analysis may exclude
these four preidentified IDs; do not repair labels or select exclusions after
observing model performance. A deterministic gold-candidate compatibility check
passes the other 1,907 AST examples; that is software validation, not a model score.

`multiple` means one required call selected from several provided functions in
this data/checker; it does not mean a multi-step execution trace. Both simple
and multiple use the original possible-answer lists. No answer file exists for
the relevance/irrelevance categories: the label there is call presence/absence.

## Normalization and label isolation

Each example contains `id`, `category`, `messages`, `tools`, `ground_truth`,
`bfcl_functions`, `name_map`, and `dataset_revision`. Functions exposed to models
use the standard OpenAI `{"type":"function","function":...}` schema. Python
`dict`, `float`, and `tuple` become JSON Schema `object`, `number`, and `array`.
`any` becomes `string`, matching the official BFCL OpenAPI mapping. Function
names are made API-compatible; an explicit alias map restores the original
names before scoring. Prose, enums, requiredness, and defaults are retained.
The original function schemas and answers are kept unchanged for evaluation.
The BFCL-only `optional` annotation is removed from exposed schemas; the
containing `required` list already controls requiredness.

**Serialize only `messages` and `tools` in model and verifier prompts.**
`public_example` strips answers, raw functions, and alias metadata, but retains
ID/category for routing and logging. Category names themselves reveal labels
(e.g. `irrelevance`) and therefore must not be part of the prompt. Offline score
errors may contain acceptable gold values and must never become repair feedback.
Repair feedback must come only from schema validation and Jev's judgement of
the user request, available schemas, and current proposal.

## Official AST checker, without model execution

`jev_fc.scoring.score_example` calls the pinned official
`eval_checker/ast_eval/ast_checker.py` for the four labelled AST categories. Its
SHA-256 digest is
`2aae7a68461a8f76c0be3894c8901b66b56967a1989d3ab066051e3fb97f1538`.

The loader removes only the checker's import statements for BFCL modules. All
function bodies and constants are unchanged. It supplies an equivalent language
enum and a minimal model setting (`underscore_to_dot=False`) after aliases have
been restored. Java/JavaScript converter dependencies are unreachable because
only Python is supported. This avoids installing the upstream package's many
unrelated model/GPU dependencies. A different or edited checker hash is rejected.
The code is Apache-2.0, and its original file and license are retained on disk.

The adapter consumes parsed JSON calls directly; it does not execute generated
Python or benchmark functions. Thus these results measure BFCL AST correctness,
not real API execution success or side-effect safety. BFCL accepts certain
equivalent string forms and numeric representations; exact byte equality would
be a different metric and is deliberately not substituted here.

Reported fields are:

- `correct`: official AST validity on labelled tasks, or the appropriate
  call-presence decision on presence-only tasks.
- `function_correct`: exact multiset of selected function identities on labelled
  tasks; no-call correctness on irrelevance. `null` on `live_relevance`, whose
  particular function identity is not labelled.
- `arguments_correct`: official AST validity **conditional on correct function
  identities**. It is `null` for wrong functions or presence-only tasks. A
  conditional argument metric must use only non-null rows as its denominator.
- `presence_correct`: available only on presence tasks. A presence pass must
  not be described as correct parameter selection.

Keep AST accuracy and relevance/irrelevance accuracy separate. A macro-average
of these categories, if presented, is a custom aggregate with stated weights.
It is not the weighted score shown on the public BFCL leaderboard. In particular,
the 15 test examples in `live_relevance` yield an imprecise estimate.

Malformed structured output is scored as incorrect. It must not be converted to
an empty call list, which would incorrectly reward failures on no-call tasks.
Network failures, exhausted retries, and interrupted runs must be explicitly
reported rather than silently removed from the denominator.

## Independent upstream cross-check

The full official package is optional. To reproduce the upstream environment at
the same revision, install it into a separate environment (its dependencies are
substantially heavier than this project's):

```bash
git clone https://github.com/ShishirPatil/gorilla.git work/gorilla
git -C work/gorilla checkout 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
python -m venv work/bfcl-venv
work/bfcl-venv/bin/python -m pip install -e work/gorilla/berkeley-function-call-leaderboard
```

The exact programmatic scoring hook is:

```python
from bfcl_eval.constants.enums import Language
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

# original_functions/ground_truth are the untouched BFCL fields;
# decoded_calls is [{"original.function.name": {"argument": value}}, ...].
verdict = ast_checker(
    original_functions, decoded_calls, ground_truth,
    Language.PYTHON, category, "gorilla-openfunctions-v2",
)
```

Here `gorilla-openfunctions-v2` selects the official configuration with name
rewriting disabled; it does **not** mean that this model generated the calls.
After restoring aliases, that setting matches our adapter. The standard
`bfcl evaluate --model MODEL_NAME --test-category ... --partial-eval` command is
also supported upstream, but requires its registered model handler and result
serialization format. This project intentionally uses the direct checker hook
so that a custom generator/verifier does not impersonate another model handler.

## Experimental reporting

Run the same examples with the same initial proposal under direct generation,
schema-only feedback, self-feedback, Jev verification without repair, and Jev
verification with one through three repair rounds. A repair budget of three
means at most **four proposals**: the initial one and three revised proposals.
Report the precise cap behavior; forced final dispatch and abstention are
different policies. BFCL evaluation here simulates that final dispatch and does
not perform external calls.

Include paired changes in final AST accuracy, exact function selection,
conditional argument accuracy, repair/regression counts, call/no-call accuracy,
verifier acceptance coverage, generator/verifier requests, tokens if returned,
and end-to-end latency. Compute paired confidence intervals or a paired test on
fixed examples. Distinguish deterministic schema validity from semantic
verification and report an ablation for each check. No experimental result can
be inferred from the number of prepared examples or from passing local tests.
