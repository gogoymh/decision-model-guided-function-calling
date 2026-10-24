# DMGFC public API migration

The current distribution is `dmgfc`, matching **Decision-Model-Guided Function
Calling with Cumulative Verification Feedback (DMGFC)**. New applications should
import `dmgfc` and construct `dmgfc.DMGFC`. This rename keeps the repair protocol,
provider settings, default Jev verifier and recorded experiments unchanged.

## Install from the repository

```bash
git clone git@github.com-personal:gogoymh/decision-model-guided-function-calling.git
cd decision-model-guided-function-calling
pip install -e .                 # or: uv sync
```

The package metadata now names the distribution `dmgfc`. Update application
dependency declarations that named `dmfc` to use the renamed repository or
distribution. These instructions install the checkout; they do not imply that a
new package has been published to PyPI.

If this environment already has the earlier `dmfc` distribution installed,
remove it before reinstalling the checkout so that two distributions do not own
the same compatibility modules and console commands:

```bash
python -m pip uninstall dmfc
python -m pip install -e .
```

The renamed distribution includes the old import paths; uninstalling the earlier
distribution does not require rewriting existing applications.

## Public names

| Interface | Current name | Compatible earlier name |
|---|---|---|
| Python import | `import dmgfc` | `import dmfc`, `import jbfc` |
| Agent class | `dmgfc.DMGFC` | `dmfc.DMFC`, `jbfc.JBFC` |
| Submodules | `dmgfc.core`, `.providers`, `.jev`, `.tools`, `.perplexity` | corresponding `dmfc` submodules |
| Evaluation command | `dmgfc-eval` | `dmfc-eval`, `jev-fc` |
| Cumulative function and argument judgments | `dmgfc` | `dmfc`, `jbfc`, `jev_joint` |
| Function judgments only | `dmgfc-f` | `dmfc-f`, `jbfc-f`, `jev_function` |
| Argument judgments only | `dmgfc-a` | `dmfc-a`, `jbfc-a`, `jev_arguments` |
| Latest rejected candidate only | `dmgfc-latest` | `dmfc-latest`, `jbfc-latest`, `jev_joint_latest` |

Baseline options `direct`, `schema` and `self-refine` retain their names. Provider
classes, verification helpers, result fields and the `jev=` verifier option are
available from the canonical `dmgfc` package.

`Result.method` uses the canonical public names even when the caller supplies a
legacy alias. Update application comparisons against values such as `"dmfc"` to
use `"dmgfc"`. Saved experiment records still use their original `jev_*` IDs.

```python
import dmgfc

agent = dmgfc.DMGFC(
    "openai:gpt-5-nano",
    tools=[get_weather],
    method="dmgfc",
    max_repairs=3,
    threshold=0.5,
    on_exhaustion="first",
)
result = agent.call("What is the weather in Seoul?")
```

To verify another framework's proposed calls, use
`dmgfc.verify(messages, tool_calls, tools)`. To use the prepared native decision
model, pass `dmgfc.PerplexityDecider(...)` as `jev=`; its model identifier and wire
protocol remain the pinned values described in the
[DGX guide](dgx-decider-experiments.md).

## Evaluation CLI and reproducibility

```bash
dmgfc-eval --help
dmgfc-eval run --help
```

For a separately authorized evaluation, the `run` command accepts public method
names such as `--methods direct,dmgfc,dmgfc-f,dmgfc-a,dmgfc-latest`. It translates
them to the frozen experiment engine's `direct` and `jev_*` IDs. Help commands
make no model requests; `run` can make paid model requests.

The historical `src/jev_fc` implementation, raw method IDs, result directory
names, cache contents, dataset manifests and recorded source/configuration hashes
keep their original values. Existing reproduction scripts continue to use those
values and compatible command aliases. Dated migration/review notes and archived
manuscripts describe the names in use when they were written.

New DGX preparation plans fingerprint the canonical `src/dmgfc` implementation
alongside compatibility wrappers and the frozen experiment engine. Old receipts
and manifests remain evidence of the earlier source state and are not rewritten.
