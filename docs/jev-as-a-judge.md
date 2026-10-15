# Reading notes: JEV-as-a-Judge

Reviewed on October 15, 2026. The review covered the **complete version-5 main text, limitations, and Appendices A–Q**, not just the title or abstract. The full HTML was obtained from arXiv; reading the appendices was divided between two research agents and integrated with the main-text review.

## Exact reference

**Yubo Li, Yidi Miao, Ramayya Krishnan, and Rema Padman. 2026. _JEV-as-a-Judge: Accept When Confident, Escalate When Unsure._ arXiv:2609.26550, version 5.** All four authors are affiliated with Carnegie Mellon University in the paper.

- [Version-5 abstract and metadata](https://arxiv.org/abs/2609.26550v5)
- [Complete version-5 HTML](https://arxiv.org/html/2609.26550v5)
- [Version-5 PDF](https://arxiv.org/pdf/2609.26550v5)
- DOI: [10.48550/arXiv.2609.26550](https://doi.org/10.48550/arXiv.2609.26550)
- First arXiv submission: September 22, 2026, 15:05:56 UTC.
- Version 5: October 9, 2026, 13:41:59 UTC. The document's cover date is September 29, but the versioned arXiv record identifies the October 9 revision used here.
- Bibliography key: `li2026jevjudge` in `paper/references.bib`.
- The verified publication type is an **arXiv preprint**. No peer-reviewed venue is asserted.

The GitHub repository `danielgshea/jev-as-a-judge`, whose README is titled “Jev as a judge for agent evals,” is a different project. It is not the requested paper and is not used as its bibliographic identity. Version 1 of the arXiv paper also has different sample sizes and cascade findings; the current notes and citation use **version 5 throughout**.

## Research question and method

The paper studies whether Jev can serve as an inexpensive first judge of **finished outputs**, reserving a stronger reasoning judge for cases where Jev is uncertain. This is serial escalation between judges, not iterative rewriting of a function call.

The main configuration is TypeSafe **Jev 1.13.0**. Except for an interface audit, the paper sends **one Choice question per request**. Every generative comparator receives the same task instructions, structured state, allowed labels, and decision-plus-probability contract, without a requested rationale. Its probability estimates are verbalized, whereas Jev returns them through its typed interface. The comparison therefore concerns complete deployed configurations; the paper does not isolate an architectural cause.

For label probabilities `p_k`, the reported confidence is **`q = max_k p_k`**, the probability of the selected verdict. This is explicitly **not Jev's native `confidence` field**, whose formula is not public. Correlations between native confidence and `q` are high but do not make the quantities interchangeable.

For pairwise judgments, the frozen policy judges both candidate orders and aligns probabilities before averaging:

`p_bar(A) = [p_first(A,B) + 1 - p_first(B,A)] / 2`.

The chosen label is the more probable semantic response. If its order-averaged confidence reaches threshold `tau`, Jev's verdict is final. Otherwise, an **independent stronger judge**, which does not see Jev's verdict, supplies the final decision. Invalid first-stage outputs escalate; exact probability ties receive half credit. Both Jev orders are included in fee accounting. Repeated judgments are not rewritten until they become better answers.

## Experimental scope

The main comparison contains **17 configurations**: Jev plus 14 generative judges and two reward models, PairRM and Skywork-Reward-V2. The base workloads total **5,172 judgments**:

| Workload | Number | What is judged |
|---|---:|---|
| RewardBench | 1,500 pairs | Preference across chat, difficult chat, safety, and reasoning |
| JudgeBench, GPT-4o split | 350 pairs | Objective correctness, including knowledge, reasoning, math, and code |
| HaluEval QA | 3,000 answers from 1,500 questions | Evidence-grounded factuality, one correct and one hallucinated answer per question |
| Earlier final-answer data | 150 replies | Agreement with existing correct/incorrect/no-answer labels |
| GSM8K conversation control | 108 replies | Final answers from repeated challenges; all references correct |
| Synthetic evidence control | 64 judgments | Support, contradiction, missing information, and distractors |

The 642-item pilot and 4,530-item held-out set share no source question. Public-task pilot examples are divided 40/60 into selection and pilot-test data. The **96 selection preference pairs** (64 RewardBench and 32 JudgeBench) determine the original frozen routing thresholds. The held-out preference evaluation has **1,610 pairs**, comprising 1,340 RewardBench and 270 JudgeBench pairs.

Follow-ups include 100 four-way RewardBench 2 prompts, 500 RM-Bench prompts with style combinations and reversed orders, matched multiple-choice/free-response conditions, 400 grounded summaries, 200 reference-free responses, open-weight decision models, a live replication of 510 already studied pairs, and a prospective live experiment with 570 held-out pairs from two new workloads.

The main multi-family comparison and many slice analyses are exploratory. The frozen routing rule, primary human-adjudication outcome, and prospective live-test protocol were specified before their relevant evaluation outcomes. A held-out sample is not the same as a fully preregistered model-selection process: the paper makes this distinction explicitly.

## Verified version-5 findings

These are **results reported by Li et al.**, not results produced by our function-calling project.

### Stand-alone judging: main Table 1 and Sections 5–6

- Jev reaches **92.5% on RewardBench**, compared with GPT-6 Astra's 92.5%; **87.3% on HaluEval**, compared with 88.4%; and **78.6% on JudgeBench**, compared with 93.1%.
- The JudgeBench gap is **−14.6 percentage points**, with a 95% source-cluster interval of **[−18.9, −10.3]**. It is particularly large for deriving or tracing a result in reasoning, code, and mathematics.
- On the separate matched timing panel, Jev's median outcome latency is approximately **0.15 seconds**, its p95 is **0.24 seconds**, and its estimated fee is **$0.044 per 1,000 judgments**. GPT-6 Astra costs $12.182 per 1,000 on that panel, giving the reported **0.36% fee ratio**. These are measured configuration/workload comparisons, not universal prices per judgment or intrinsic model-inference times.
- Jev ranks eighth of the 15 LLM-style judges on the pooled 4,850 public judgments. GPT-6 is a reference comparator, not uniformly the most accurate judge; GPT-5.6 Sol has higher pooled accuracy and roughly half its fee.
- The authors interpret the pattern as **reading a verdict from supplied evidence versus deriving a result**. It is an observed operating pattern, not proof that the proprietary model is incapable of reasoning.

### Original frozen cascade: main Tables 3–4; Appendix L/Table 30

The original threshold uses the **point-estimate rule**, not the later lower-confidence-bound rule: choose the highest-coverage threshold whose selection-set cascade accuracy falls no more than two points below its fallback. The grid and rule were fixed in advance; the threshold for Jev → GPT-6 Astra is **0.9**.

| Held-out population | Jev alone | Cascade | GPT-6 alone | Cascade minus GPT-6 | Escalated | Reported fee ratio |
|---|---:|---:|---:|---:|---:|---:|
| RewardBench, 1,340 | 92.8% | 93.7% | 92.4% | +1.27 pp, CI [0.45, 2.03] | 24.8% | 0.267 |
| JudgeBench, 270 | 81.3% | 92.2% | 93.0% | −0.74 pp, CI [−2.22, 0.74] | 64.8% | 0.668 |
| Pooled, 1,610 | 90.9% | 93.4% | 92.5% | **+0.93 pp, CI [0.24, 1.66]** | 31.5% | **0.414** |

Jev-alone values here use aligned, order-averaged probabilities; they are not the single-order numbers in Table 1. The conservative upper pooled fee ratio is **0.440**, compared with the reported-usage ratio 0.414. The pooled sample is **83% RewardBench**, so its gain should not be presented as uniform improvement on difficult correctness. These table values are offline policy simulations from retained outputs.

### Live replication: Section 8; Appendix O/Table 37

The frozen 0.9 policy is rerun live on **510 previously analyzed pairs**: all 270 held-out JudgeBench pairs and 240 RewardBench pairs. Both Jev orders are submitted concurrently and the fallback runs only on escalations.

- Cascade accuracy is **92.5%**, compared with **93.1%** for GPT-6; paired difference **−0.59 pp**, CI **[−1.95, 0.59]**.
- The fee ratio is **0.572**, and the escalation rate is 46.5%.
- Median pair latency is **0.27 seconds**, compared with GPT-6's **2.10 seconds**; p95 is 4.26 versus 4.49 seconds. Escalated items wait for both stages, so the typical speedup shrinks on workloads that often escalate.
- Live and offline accept/escalate choices agree on 98.6% of pairs.

This confirms an offline simulation on previously studied items; it is **not** the new-workload prospective test and does not have the pooled 1,610-pair accuracy/fee result.

### Prospective test: Section 9, main Table 5; Appendix N

Two new correctness workloads are collected under a protocol frozen before model calls: **500 PPE correctness pairs** and all **270 JudgeBench Claude-split pairs**. Each uses 100 local selection pairs; the remaining **400 PPE + 170 JudgeBench = 570 pairs** are held out.

The selection rule now requires the one-sided 95% lower confidence bound on the mean paired cascade-minus-fallback difference to exceed −0.02:

`mean(d) - 1.645 * sd(d) / sqrt(n) >= -0.02`, where `d` is −1, 0, or 1.

The rule selects **0.95 for PPE** and **0.99 for JudgeBench**, with every item escalated if no threshold qualifies. On the held-out data, the cascade matches GPT-6's **observed aggregate accuracy** on each workload: **88.2% on PPE** and **94.7% on JudgeBench**, pooled **90.2%**. This is equality of empirical accuracy, not identical decisions, an equivalence theorem, or a guarantee of parity on future requests.

The pooled cascade escalates **74.2%**, costs **75.5% of GPT-6's fee**, and has median latency **1.87 versus 1.91 seconds**. Thus a weaker first-stage workload produces much smaller savings than RewardBench. The cascade's pooled paired interval is **[−0.70, 0.70] pp**. Of the held-out Claude-split pairs, 57 questions were seen in the earlier GPT-4o split and 113 were not; parity is reported in both groups.

## Confidence, calibration, and failure modes

### A probability interface is not a reliability guarantee

Section 7 separates accuracy, calibration, and error ranking. On base judgments, Jev's Brier scores are **0.113 / 0.297 / 0.196** for RewardBench / JudgeBench / HaluEval. GPT-6 is substantially better on JudgeBench. Apparent relative calibration can reverse when questionable HaluEval labels are corrected (Appendix D/Table 11). A probabilistic API therefore needs task-specific empirical validation.

### Threshold transfer and label budget

Temperature scaling fit on pilot data improves held-out HaluEval NLL but worsens RewardBench and JudgeBench NLL; fitted temperatures also point in different directions. A single universal temperature is not supported.

The original 96-pair threshold sample has only 32 JudgeBench examples. For a GPT-5.6 fallback it selects 0.7, which loses **4.81 pp on held-out JudgeBench**, despite acceptable pooled/selection behavior. The paper's repeated label-budget study (Appendix L/Table 32) finds the point-estimate rule exceeds a two-point loss in about **40–50%** of draws. The lower-bound rule reduces this to about 5% at 96 or 192 labels, and per-workload thresholds reduce it to **0.6% at 96** and **0.1% at 192**. These are simulations on the same existing pool and do not establish protection under distribution shift.

### Fixed output type and presentation matter

In the 48-example initial interface audit, equivalent Choice and Noul probabilities differ by **0.055 on average**; Choice and Score differ by 0.015; complementary Noul questions miss summing to one by **0.045 on average**; and one close hard decision changes between Choice and Noul. All were asked within a shared request. The authors identify co-question context, stochastic variation, and rounding as possible contributors and recommend tuning for a fixed output type (Section 7.3; Appendix J).

Jev changes no verdict across 96 repeated requests but changes four of 48 paraphrased rubrics. Candidate reversal flips 3.7% of RewardBench and 11.1% of JudgeBench decisions. Most flips are low-confidence, so the gate absorbs much of the order sensitivity. Averaging both orders raises pairwise accuracy slightly but does not uniformly improve calibration or error ranking; four-way rotation averaging reduces accuracy from 72% per rotation to 70% averaged while improving AUROC (Section 7.3; Appendix K). “More votes” is not automatically a better-calibrated decision.

### Confidently wrong cases remain

- On RM-Bench, Jev's accuracy falls from **85.4% with matched styles to 76.6% when the wrong answer is more elaborate**. GPT-6 falls from 91.8% to 90.1%. Two-order error-ranking AUROC weakens from **0.839 to 0.764** (Section 7.4; Appendix F).
- On **200 reference-free prose judgments**, Jev is near chance at **53.5%**, despite mean maximum probability **0.91**. Error-detection AUROC is **0.498**, so confidence thresholding does not help. Grounded summaries differ in both content and label provenance, preventing a clean causal “remove the evidence” interpretation (Appendix H).
- Serial escalation cannot correct errors shared by both judges. At `q=1`, Jev and GPT-6 agree in correctness on all but two of 2,332 public items; very high confidence does not establish truth (Sections 7–8).

## Study limitations and appendix qualifications

- **Finished outputs, not agent control.** The limitations section explicitly says that tool choice, next-action decisions inside a long trajectory, and error propagation into later steps are not tested. This is the clearest distinction from the present function-calling experiment.
- **One proprietary Jev version.** Weights, size, training data, and training recipe are not inspectable. Benchmark contamination cannot be excluded. Architecture cannot be isolated from settings, serving, model size, or reasoning effort.
- **Human adjudication is internal.** Appendix D audits 183 selected items, not a random benchmark sample. One research-team human annotator is blind to labels and judge outputs; an LLM second pass flags disagreements, and an author resolves them. This is a deviation from the planned two-human procedure and is not independent external human validation. The labels of the separate 150-item final-answer set have undocumented annotators.
- **Exploratory comparisons.** Multi-family comparisons, threshold sweeps, and slice analyses are retrospective. Only the specified frozen policies, primary adjudication outcome, and prospective test have the stronger advance-specification status.
- **Operational measurements.** Timing comes from a separate matched 120-judgment panel with a persistent client and fixed pacing, not the contended bulk-evaluation loop. It includes network, provider, and retry time. Costs are usage-based estimates with conservative reservations for missing usage, not invoices or matched compute.
- **Open-model appendix added in v5.** Appendix A compares 21 open-weight typed decision models. Six are within sampling error of Jev's **88.3% pooled accuracy**; the highest observed score is **89.0%**, with no model significantly exceeding Jev in the reported paired comparison. Several larger open models have better error-ranking AUROC. This narrows any claim that useful typed judging is unique to proprietary Jev. These models are tested as stand-alone judges; their escalation policies, deployment fees, and transfer to follow-up/prospective workloads are not established by this appendix.
- **Prospective scope remains finite.** The genuinely new-workload experiment covers two correctness workloads and 570 held-out pairs; equality of aggregate accuracy should not be generalized to arbitrary tasks.
- **Reproduction has an explicit boundary.** Appendix Q supplies public sampled inputs and rubrics, retained judgments, accounting/provenance, offline analyses, and prospective/live test records. It excludes private conversation data, raw API response logs, account metadata, and credentials. Some historical calibration, repeated-request, interface, and rubric-ablation findings are retained summary tables rather than completely regenerated pipelines. Offline analysis does not require paid API calls, and the study trains no new model.

## Relationship to Jev-Backed Function Calling (**JBFC**)

| Dimension | Li et al.'s JEV-as-a-Judge | **JBFC** |
|---|---|---|
| Object evaluated | Finished candidate outputs, preference pairs, factuality, final answers | An unexecuted LLM-proposed function name and arguments |
| Jev interface | Primarily one Choice per request | Two Noul questions in one shared request |
| Gating quantity | `q = max label probability`: confidence in **either** verdict | `p(function correct)` and `p(arguments correct)`: probabilities of **yes** |
| Gate outcome | Keep a confident Jev verdict, including a negative label | Pass only if the schema floor and both correctness checks pass |
| Failure/uncertainty response | Independent stronger judge replaces the verdict | Failed checks condition a new proposal from the same generator, at most three repairs |
| Pair-order treatment | Aligned two-order probability average for frozen pairwise policy | No pairwise A/B comparison; fixed normalized tool catalog and two-check schema |
| Primary evaluation | Judgment accuracy, error ranking, calibration, fee, latency | BFCL candidate correctness, selection/argument diagnostics, gate errors, repair/regression, fee, latency |
| Evidence not supplied by the prior work | Function-call repair effectiveness or safe execution | Must be measured in our experiments; cannot be inferred from prior judge benchmarks |

The paper directly motivates **measuring whether a cheap probabilistic checker adds value, retaining an independent correctness evaluator, and calibrating thresholds locally**. It also strengthens the need to report false acceptance, false rejection, correction/regression, and correlated errors rather than acceptance alone.

Its 0.9 confidence threshold does **not** validate our untuned 0.8 correctness thresholds. `max(p_yes, 1-p_yes)` and `p_yes` answer different questions; a confidently negative validity judgment would have high confidence in their sense but must fail our execution gate. Its Choice findings also do not automatically transfer to two batched Nouls. Our calibration script is an independently specified development-data analysis, not a reproduction of their paired-cascade lower-bound selection rule.

## Suggested concise manuscript treatment

**Related work:** “Li et al. study JEV-as-a-Judge, retaining confident Jev verdicts and escalating uncertain cases to an independent reasoning judge. Their version-5 evaluation finds workload-dependent accuracy–cost benefits, while confidence routing weakens under misleading answer styles and fails on reference-free prose. They evaluate finished outputs, leaving tool selection and error propagation inside agents untested. **JBFC** instead evaluates two batched Noul checks on an unexecuted function call and uses failed judgments to repair the proposal with the same generator.”

**Method or limitations:** “Because Choice and Noul probabilities can differ even for aligned questions, we treat thresholds as interface- and workload-specific rather than transferring the confidence thresholds of JEV-as-a-Judge.”

Any numerical claim should be explicitly attributed to Li et al., tied to version 5 and its population, and kept separate from our still-pending BFCL results. The new reference belongs in both `paper/references.bib` and the standalone manuscript's inline bibliography.
