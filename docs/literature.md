# Literature and experiment protocol

Research date: October 15, 2026. The bibliography is `paper/references.bib`.
These notes separate verified descriptions of prior work from proposed experiments.
No paid model evaluation has been run, and there are no empirical findings for the proposed method yet.

## Jev identity and primary documentation

Jev is TypeSafe AI's flagship **System One** model. It accepts a state and typed questions and returns structured answers, including probabilities and confidence fields. It is not an ordinary text-generating LLM. The relevant endpoint is `POST https://api.typesafe.ai/v1/systemone`.

- [Official model reference](https://docs.typesafe.ai/models) (`typesafe2026models`). As retrieved on October 15, 2026, `jev-1.13.0` is the stable version behind `jev-latest`, and the October 14 release `jev-1.14.0` is behind `jev-preview`. Pin a version rather than an alias in experiments, and log the version returned by the API.
- [Official HTTP API](https://docs.typesafe.ai/api).
- [Noul primitive](https://docs.typesafe.ai/primitives/noul): a yes/no question returns a probability of the answer being yes. The correctness-oriented questions should make the positive label explicit. Do not reverse the meaning of a risk-oriented probability in code.
- [Confidence](https://docs.typesafe.ai/confidence): probability and confidence are distinct quantities. A reported high probability is not itself a guarantee of correctness, and the proposed benchmark must measure calibration and false acceptance rather than assume the provider's general claims transfer to this task.
- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe announcement by Diogo Almeida, September 15, 2026. The provider describes its training as RLCD; this announcement is not a peer-reviewed Jev paper. Do not invent one or present provider performance claims as findings of this project.

The official `llms.txt` index also contains two especially close examples:

1. [Function calling cookbook](https://docs.typesafe.ai/cookbooks/function_calling) (`typesafe2026functioncalling`). Jev directly selects a function and closed-set arguments through Choice/Noul questions. Function names and argument values have enumerated alternatives; unconstrained text, numbers, and dates are left at defaults in that example. The proposed project instead verifies the complete call proposed by a generative LLM, including arguments that are not closed-set values.
2. [SDE cascade cookbook](https://docs.typesafe.ai/cookbooks/sde_cascade) (`typesafe2026sdecascade`). A small generative model extracts structured data, Jev evaluates per-field errors, and failures escalate to a larger reasoning model. The proposed project uses function-call selection/argument checks and bounded repair by the same configured generator. Therefore the broad idea of an LLM proposal followed by a Jev verifier is established in the provider's own examples.

## Closest formal prior work

| Citation | Verified source and scope | Relationship to this project |
|---|---|---|
| Mekala et al., 2024, **ToolVerifier: Generalization to New Tools via Self-Verification** (`mekala2024toolverifier`) | [arXiv](https://arxiv.org/abs/2402.14158), [full text](https://arxiv.org/html/2402.14158v2). Seven authors: Dheeraj Mekala, Jason Weston, Jack Lanchantin, Roberta Raileanu, Maria Lomeli, Jingbo Shang, Jane Dwivedi-Yu. | Explicitly decomposes tool selection and parameter generation and verifies both. A tool-selection model is fine-tuned on synthetic data; contrastive questions resolve top-two candidates; parameter verification compares alternatives. This precludes claiming that the two-part decomposition or verification itself is new. |
| Hamad et al., 2025, **ToolCritic: Detecting and Correcting Tool-Use Errors in Dialogue Systems** (`hamad2025toolcritic`) | [arXiv](https://arxiv.org/abs/2510.17052). Hassan Hamad, Yingru Xu, Liang Zhao, Wenbo Yan, Narendra Gyanchandani. The verified reference is the preprint; no venue is asserted. | A diagnostic critic identifies eight tool-use error types and provides targeted feedback for the main LLM to revise its response. This is direct prior work for critic-driven tool-call correction. |
| Gou et al., 2024, **CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing** (`gou2024critic`) | [arXiv](https://arxiv.org/abs/2305.11738); the primary page identifies ICLR 2024. | External tool feedback supports progressive output revision in QA, program synthesis, and toxicity reduction. Do not claim the use of external feedback is novel. |
| Madaan et al., 2023, **Self-Refine: Iterative Refinement with Self-Feedback** (`madaan2023selfrefine`) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91edff07232fb1b55a505a9e9f6c0ff3-Abstract-Conference.html), vol. 36, pp. 46534–46594. | Same-model feedback and revision supply a necessary control for improvement due simply to extra inference. |
| Shinn et al., 2023, **Reflexion: Language Agents with Verbal Reinforcement Learning** (`shinn2023reflexion`) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html), vol. 36, pp. 8634–8652. | Linguistic feedback and episodic reflection improve later attempts without updating weights. The project shares feedback-conditioned inference but does not implement episodic memory or environmental trial-and-error. |
| Huang et al., 2024, **Large Language Models Cannot Self-Correct Reasoning Yet** (`huang2024selfcorrect`) | [arXiv](https://arxiv.org/abs/2310.01798); the primary page identifies ICLR 2024. | Motivates measuring rather than assuming repair benefits. Its negative results concern intrinsic reasoning self-correction, so do not generalize them to all modern tool-calling models or claim they prove Jev will help. |

The defensible contribution is an **implemented and reproducible evaluation of a particular Jev-backed pre-execution protocol**, including separate semantic judgments, their conjunction, bounded repair, and measured quality/cost/error tradeoffs. Without experiments it is a method-and-protocol draft, not a demonstrated accuracy improvement. Avoid “first,” “guarantees,” “eliminates hallucination,” and “state of the art.” If the paper needs a stronger novelty claim, establish it through experiments and further literature review rather than wording alone.

## Benchmarks and tool-learning context

| Citation | Verified publication | Use |
|---|---|---|
| Patil et al., 2025, **The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to Agentic Evaluation of Large Language Models** (`patil2025bfcl`) | [PMLR](https://proceedings.mlr.press/v267/patil25a.html), ICML 2025, vol. 267, pp. 48371–48392. Correct author order from PMLR: Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, Joseph E. Gonzalez. | Primary evaluation source. Pin the dataset/evaluator revision and name the exact included categories. A single-turn subset is not the full BFCL V4 agentic leaderboard. Use the official category evaluator wherever possible. |
| Li et al., 2023, **API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs** (`li2023apibank`) | [ACL Anthology](https://aclanthology.org/2023.emnlp-main.187/), EMNLP 2023, pp. 3102–3116. | Representative complementary runnable API/dialogue benchmark; useful for later external validation. Cite as context unless an adapter and its tests are actually implemented. |
| Guo et al., 2024, **StableToolBench: Towards Stable Large-Scale Benchmarking on Tool Learning of Large Language Models** (`guo2024stabletoolbench`) | [ACL Anthology](https://aclanthology.org/2024.findings-acl.664/), Findings ACL 2024, pp. 11143–11156. | Explains reproducibility issues caused by live API drift and motivates frozen data/evaluators. Its cached/simulated server plus evaluator is not equivalent to a local exact-match check. |
| Qin et al., 2024, **ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs** (`qin2024toolllm`) | [arXiv metadata](https://arxiv.org/abs/2307.16789), [official project repository](https://github.com/OpenBMB/ToolBench) confirms ICLR 2024. | ToolBench dataset and tool-use learning background. Do not confuse it with the different ToolBench dataset used by ToolVerifier without examining their sources. |
| Patil et al., 2024, **Gorilla: Large Language Model Connected with Massive APIs** (`patil2024gorilla`) | [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/e4c61f578ff07830f5c37378dd3ecb0d-Abstract-Conference.html), vol. 37, pp. 126544–126565. | Function-call training and API hallucination background. The arXiv preprint is dated 2023, but the cited proceedings version is 2024. |

The bibliography deliberately prioritizes close verification work over adding every foundational tool-use reference. Its 14 entries can be typeset in one or two reference pages. URLs and source metadata should be preserved in the `.bib` file even if a compact manuscript bibliography uses shortened venue names.

## Prespecified experimental protocol

### Definitions that must match the implementation

- A **proposal** is the complete predicted call set for one user request, including zero calls when the request should not invoke a tool. Define whether sequential dependencies are supported; do not silently treat them as independent parallel calls.
- The **selection** judgment asks whether the proposed function names, multiplicities, and abstention satisfy the request given the entire available catalog. It must catch missing and extra functions, not merely whether each proposed name exists.
- The **arguments** judgment asks whether the values are supported by the request, assigned to the correct parameters, complete where required, and consistent with the selected signatures. It must cover semantic mistakes after ordinary schema checks.
- Acceptance at round `t` is `schema_valid(t) AND selection_pass(t) AND arguments_pass(t)` on the **same candidate**. Jev evaluates all well-formed, parseable candidates, including those that fail schema validation, but the deterministic schema floor overrides acceptance. If any required check fails, use feedback to revise. Re-evaluate both semantic checks after every revision; a prior pass must not remain latched. Repair retains all previous proposal/feedback pairs in the conversation history.
- Treat malformed output and schema validation separately. A schema-only repair control reveals whether gains come only from fixing types/required fields rather than semantic verification.
- Define the retry cap explicitly. Recommended notation: `r in {0,1,2,3}` **repairs after the initial proposal**, hence at most `r+1` generator calls and verifier evaluations. If the chosen implementation caps total proposals at three, write that instead. Do not use “three iterations” ambiguously.
- Define behavior after budget exhaustion. **Always-last** passes the final proposal to benchmark evaluation even if Jev rejects it or schema validation fails; malformed proposals score as incorrect. This matches forced completion after bounded repair. **Abstain** withholds rejected proposals. Report them separately and never treat withheld errors as corrected predictions.
- The experiment should evaluate or execute calls only in the benchmark's controlled evaluator. Local parsing/equivalence is a proposal correctness test, not live execution or demonstrated downstream task success.

### Data splits and leakage control

1. Freeze a dataset revision, evaluator revision, ordered example IDs, category selection, preprocessing, model versions, prompts, thresholds, sampling settings, and repeat seeds before the test run.
2. Split development/test by example IDs, preferably also protecting shared function templates when tuning prompts. A deterministic hash split is acceptable if fixed in a manifest. Report the resulting counts by category.
3. Use development examples to choose thresholds and prompt wording. Test labels never enter the generator or verifier payload. Gold labels are read only by offline evaluation code after predictions are produced.
4. The verifier may see the user request, available function schemas/descriptions, selected calls, and explicit execution constraints. It must not see reference calls, reference argument alternatives, official correctness labels, or evaluator feedback.
5. Reject missing-label or unsupported-category records during preparation with explicit counts and reasons. Never silently discard hard failures at scoring time. Keep parse errors, API errors, rate limits, and exhausted loops in denominators or report their treatment explicitly.
6. Keep provider errors distinct from method failure, use bounded retry/backoff for transient HTTP failures, and save resumable records. Log model version returned, usage, elapsed time, and a digest of request configuration without recording API keys.
7. Freeze LLM capability mode (native tool calling versus text-JSON emulation). Different output modes are separate experimental conditions.

### Required comparison rows

| Method | Purpose |
|---|---|
| Direct generator | One proposal, no semantic verifier or repair. |
| Schema-only repair | Repairs deterministic structural/type violations; separates semantics from format improvements. |
| Self-feedback repair | Same generator critiques and repairs its candidate with the same proposal cap; measures gains from extra generation and feedback without Jev. Clearly document the actual critic prompt and whether critique consumes an additional call. |
| Jev selection only | Uses only selection feedback for the gate and repair. |
| Jev arguments only | Uses only argument feedback for the gate and repair. |
| Jev joint | Requires both criteria and feeds back both judgments. |
| Jev joint at caps 0, 1, 2, 3 | Identifies marginal repair benefit, regression, overhead, and early stopping. A zero-repair verifier arm is useful for studying acceptance alone. |

The same proposal cap is a **cap-matched** comparison, not necessarily a token-, time-, or dollar-matched one. Measure observed generator/verifier calls, tokens, cost, and latency. For a stronger budget control, compare quality at matched observed spend or latency, or report a quality-cost curve. A fixed number of repeated samples followed by oracle-best selection is not a fair deployable baseline. Use identical initial proposals across methods when feasible, paired by example and generator seed, and make the reuse/cache policy explicit.

The implemented budget ablation uses **trajectory-prefix replay** rather than new model runs. Execute the joint-check method with a maximum of three repairs, then derive caps 0, 1, and 2 by stopping the same trace at the corresponding proposal (or its earlier first acceptance). This is valid here because the cap is not disclosed to either model, no tool executes during the repair trajectory, and all earlier feedback is unchanged by the later stopping decision. Reuse the same trace for the optional abstention dispatch policy. Compute candidate scores and logical usage/latency from the selected prefix. Raw usage and observed wall time are saved; logical estimated cost is not an actual provider bill and cannot include unreported usage from failed requests. Do not extrapolate missing proposals after a provider error, and do not describe these policy replays as independent replicate runs.

### Metrics and denominators

Let `N` be all test requests, `C_i` the official correctness of the final candidate, and `A_i` the joint gate decision. Report every metric with the relevant sample count.

- **Final candidate correctness:** `sum(C_i)/N`, scoring the final prediction regardless of gate. This is the primary correction metric for always-last mode. Use official AST/equivalence or executable evaluation; name a custom local proxy as a proxy.
- **Selection correctness:** compare function names and call multiplicities under the benchmark's task semantics; include correct empty call sets. State how unordered parallel calls are matched.
- **Argument correctness conditional on correct selection:** correct argument assignments divided by cases with correct function selection, with denominator shown. Joint correctness must also be reported so this conditional metric cannot hide selection errors.
- **Acceptance coverage:** `sum(A_i)/N`. For always-last mode distinguish this from execution coverage, which may remain 100% for valid proposals.
- **Accepted precision:** `sum(A_i*C_i)/sum(A_i)`. Undefined when no examples are accepted; display `NA`, not zero.
- **False acceptance among accepted:** `sum(A_i*(1-C_i))/sum(A_i)`, equal to one minus accepted precision. Explicitly distinguish from **false-positive rate among incorrect candidates**, `sum(A_i*(1-C_i))/sum(1-C_i)`.
- **False rejection among correct candidates:** `sum((1-A_i)*C_i)/sum(C_i)`.
- **Correction and regression:** among initially incorrect candidates, the fraction corrected by repair; among initially correct candidates, the fraction made incorrect. Also report unchanged and unresolved cases.
- **Loop diagnostics:** mean/median and distribution of proposal counts, acceptance round, cap exhaustion, generator parse/schema failures, verifier errors, and no-call accuracy on irrelevance cases.
- **Efficiency:** end-to-end request latency (median and p95), separately timed generator/verifier network calls, actual input/output tokens, and estimated cost with dated prices. Include retries and normal network time; state concurrency. Separate cached replay from live timing.
- **Calibration (optional diagnostic):** Brier score or reliability bins for each check on labeled held-out candidates. Do not call the two judgments statistically independent merely because they are separate questions, or multiply probabilities without testing that assumption.

Use paired differences on the same test examples with a 95% paired bootstrap confidence interval (fixed bootstrap seed and iteration count) for the principal direct-versus-joint comparison. Bootstrap examples rather than individual tool calls when a request has multiple calls. If several generator seeds are used, show across-run dispersion and consider a hierarchical bootstrap or report each run separately. Avoid claiming significance from overlapping/non-overlapping unpaired intervals alone.

### Results tables to reserve in the four-page draft

1. **Main comparison:** method, final correctness, selection correctness, conditional argument correctness, acceptance/coverage, false acceptance, mean proposals, latency, estimated cost. Split into two compact tables if necessary.
2. **Ablation:** check configuration and cap, joint correctness, corrected initial errors, regressions, cap-exhaustion rate.
3. **Category breakdown:** exact included BFCL category names, `N`, direct correctness, joint correctness, paired difference (CI).

Every unmeasured result cell should be `--`/`pending`; the caption must say that no benchmark runs have yet been performed. Counts that are actually computed during dataset preparation may be filled, clearly distinguished from model results. The abstract and conclusion should state the proposed protocol and evaluation questions, without anticipating improvements. After execution, revise the abstract, experimental discussion, limitations, and conclusion to reflect the measured result even if the intervention fails to improve accuracy.

## Metadata cautions

- BFCL's website citation has an awkward author ordering/format; this bibliography follows the archival PMLR record.
- Reflexion's current arXiv record lists an additional author relative to the NeurIPS proceedings. This bibliography cites the archival NeurIPS paper and its five listed authors consistently.
- ToolVerifier and ToolCritic entries are explicitly preprints because no publication venue was established from the primary records used here.
- ToolVerifier refers to a benchmark also called ToolBench. Its four evaluated API domains are weather, booking, home, and cat; do not conflate these with OpenBMB's 16,000-API ToolBench merely because the names match.
- The Jev model alias and provider cookbooks can change. The experiment must save immutable configuration and retrieval dates and should pin the stable model ID.
