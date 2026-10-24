# Research repository instructions

## Canonical DMGFC Python library (user instruction, 2026-10-24)

The latest user request makes `dmgfc` the distribution and public import, `DMGFC`
the class, and `dmgfc-eval` the current evaluation command. New examples and
method options use `dmgfc`, `dmgfc-f`, `dmgfc-a`, and `dmgfc-latest`. Keep
`dmfc.DMFC`, `jbfc.JBFC`, their submodule imports and old commands/options as
compatibility aliases. This supersedes the older canonical public API names below.
Move the current library implementation into `src/dmgfc` without changing the
frozen `src/jev_fc` implementation, recorded method IDs, source hashes or results.
The default verifier, prompts, thresholds, repair limits and dispatch behavior stay
the same. New experiment manifests must hash the actual canonical source files.

## DMGFC naming and DGX Spark preparation (user instruction, 2026-10-24)

The repository is now `gogoymh/decision-model-guided-function-calling`. The current
paper title is **Decision-Model-Guided Function Calling with Cumulative Verification
Feedback** (subtitle retained following the user's PR comment) and its displayed method
names are **DMGFC**, **DMGFC-F**, **DMGFC-A**, and **DMGFC-L**. This explicitly permits
title/acronym substitutions in otherwise frozen current manuscript passages,
figures, generated display labels, and current documentation. Preserve all other
wording, numerical results, cited titles, historical archives, raw method IDs,
recorded source hashes, and the existing compatible `dmfc.DMFC` Python API.

The user authorized cloning this renamed repository onto `ssh dgx_spark` and
preparing one open decision model from the supplied comparison table. Use the
`github.com-personal` SSH alias. Keep this experiment opt-in and separate from the
historical Jev runs. Record the actual model ID, immutable revision, license,
native inference settings, input/source hashes, and measured device accounting.
Do not send gold answers or IDs to models, alter historical benchmark defaults,
claim full experiments from smoke tests, or run paid generator experiments as part
of preparation. Preserve existing DGX workloads and credentials; bind the new
native decision-model service to loopback only.

## Restore supplied polished text (user instruction, 2026-10-21)

The user's latest request supersedes all earlier permission to rewrite the text
already supplied in the original WWW ZIP. Preserve the exact original abstract
block in main.tex, the full sec/introduction.tex including red text and xxx, and
the original Task Definition heading in sec/methodology.tex. Keep additions in
originally empty sections and measured results. Do not rewrite these frozen
passages to satisfy feedback, page limits, or factual cleanup without an explicit
new instruction. This restoration is a draft with four total pages, a placeholder
and an unrun multi-verifier claim in the supplied Introduction; do not call it a
submission-ready manuscript. Keep current anonymous ACM formatting and archive
the previous revision at paper/versions/2026-10-21-before-polished-text-restore.
The restored \input{sec/abstract} requires that empty source in the upload ZIP.

## Anonymous submission and public DMFC naming (user instruction, 2026-10-20)

The user requested fixes to the Overleaf upload files for conference submission and
the repository's code names/content. This supersedes the earlier author-facing
nonacm format and public JBFC naming rules. The active paper uses
sigconf,anonymous,review, omits author identities and acknowledgements, and keeps
the original ACM class/style, standard body typography, and four TOTAL pages.
Explain the scientific Web relevance and the appropriate track on page one.
The author-facing source/PDF is preserved in
paper/versions/2026-10-20-author-review (source commit 8385b38).

Public usage is import dmfc and dmfc.DMFC. Keep jbfc/JBFC and old method strings
as compatibility aliases. The default verifier remains Jev; do not claim multiple
verifier families have been evaluated. Keep src/jev_fc byte-identical and preserve
recorded jev_* method IDs, experiment fingerprints, cached inputs, and raw data.
Displayed method names, current documentation, package metadata and visual assets
use DMFC, DMFC-F, DMFC-A, DMFC-L. Historical archives and provenance keep their
original names. Regenerate the upload ZIP and verify it builds from its own files.

## Overleaf feedback revision (user instruction, 2026-10-20)

The user supplied additional Overleaf screenshot comments and asked to resolve them.
This explicitly permits targeted Abstract/Introduction edits, superseding the earlier
freeze for this revision. Explain the three study questions with citations and current
evidence, align the summary with no-call handling and cumulative repair, remove xxx and
unrun multi-verifier claims, and distinguish completed BFCL results from pending
cross-benchmark/cross-provider experiments. Preserve all measured results and core run
hashes. The pre-Overleaf source/PDF is archived at
paper/versions/2026-10-20-before-overleaf (commit 927e895).
Screenshots 2 and 3 truncate comments after show more; do not invent their hidden text.

The additional matrix has three generators from two provider families and one verifier. New runs are separate from historical BFCL traces. API-Bank is an exploratory first-call exact-match adaptation with a fixed 53-tool catalog, family-grouped 48 development/156 test cases, and no no-call negatives. Its common native-candidate instruction was refined after format-pilot outcomes; preserve those pilots and disclose that history. Four development cases checked interface output. Never describe the added test set as untouched confirmation. Claude BFCL keeps every canonical case, counts known native schema rejection as unreadable/incorrect, and preserves the 8-worker prefix before a 32-worker continuation. Report concurrency and failed-unsaved-attempt limitations; stored standalone cost is not an invoice total.

The combined main table uses scripts/additional_paper_tables.py and the marker COMBINED MAIN TABLE: V1. It requires all historical and new completed grids, hashes, cached initials, source traces, gates, offline scores, and accounting. make final preserves this six-row table and continues to regenerate the existing single-run ablation and latency figure. Do not overwrite the combined benchmark results with the legacy four-row table.

## Reviewer revision (user instruction, 2026-10-20)

The user asked to resolve PDF feedback outside the Abstract and Introduction.
Those two source passages remain byte-identical to the imported draft; their red
draft text, xxx, and unrun multi-verifier claim require a later author revision.
Other manuscript wording and layout may be polished within the actual study scope.
The active manuscript integrates related work into Methodology and uses two tables
plus the first-page mechanism comparison and measured accuracy/API-time figure.
The WWW 2027 short-paper budget is four TOTAL pages including references and appendix,
superseding the old four-body-page rule below. make verify uses acm-short;
original and acm-draft remain available for historical versions.
The author-facing review PDF uses sigconf,nonacm to avoid sample conference metadata.
Before submission, prepare a separate anonymous/review version and check track relevance.
The imported pre-revision source/PDF is preserved at paper/versions/2026-10-20-acm-import/.
make final preserves the compact table captions/layout and updates the measured plot
including p95; it still derives runs from recorded data only. On this checkout, the
historical full derivation encounters absolute cache paths from the original host.
Do not rewrite raw manifests or run new API experiments just to regenerate the paper;
verified existing final summaries can feed final_tables.update_paper directly.

## ACM manuscript migration (user instruction, 2026-10-20)

The user supplied the WWW 2027 ACM draft and asked to preserve its existing content,
fill only missing material from the previous manuscript, and save the old version.
This supersedes the wording freeze and manuscript naming rules below for this migration.
The new abstract, Introduction, author list, red draft passages and template content stay as supplied.
`paper/main.tex` now uses the supplied `acmart.cls` and `sec/` inputs. Copied manuscript sections
use **DMFC**, **DMFC-F**, **DMFC-A**, and **DMFC-L**; verifier identifiers, library names,
experiment method IDs, recorded trajectories and results remain unchanged.
The prior source, PDF and bibliography are archived under `paper/versions/2026-10-18-jbfc/`.
`make final` fills the experiment section through the main entry point and adapts its display labels.
Build with `make paper`; validate this preserved draft with `make verify` (ACM draft mode).
The original strict four-body-page check remains available directly with `scripts/verify_paper.py`.
The draft retains incomplete claims and ACM sample metadata; do not present it as submission-ready.
The native editor cannot load the local multi-file ACM project fully; keep it open and use the
installed LaTeX export path when native compilation fails for the supplied class/project files.

Keep paper/main.tex and the implementation consistent. The paper's final settings (user decisions
2026-10-18): repair limit r=3, Jev threshold 0.5 for both checks, and the first proposal when no proposal
passes within the limit (rule keep_first). The runs were made at threshold 0.8 returning the last proposal:
results/table2_gpt-5-nano_r3 raises the r=2 seven-method run (results/table2_gpt-5-nano_r2) to r=3,
results/table1_gpt-5.4-mini_r3 adds JBFC to the gpt-5.4-mini Direct run (make r3), and make repeat adds runs
2 and 3. The Jev methods then ran again at threshold 0.5 from the same first proposals (make threshold
THRESHOLD=0.5; make t05 for JBFC-F, JBFC-A, JBFC-L and the r=10 continuation), and scripts/final_tables.py
(make final) derives the paper's runs into results/final and fills Tables 1-3 and Figure 2 without new
requests. Do not claim more than the filled tables show.

- The user froze the current Abstract, Introduction, Related Work, and Method on 2026-10-17. Preserve their wording, equations, Figure 1, and Algorithm 1 unless the user explicitly asks to revise them. Experiment updates and automatic result-table generation must not modify these frozen sections. Layout changes must preserve their content. On 2026-10-18 the user asked to correct only the sentences the protocol change made false (Abstract: repair limit, example count and results; Introduction: repair limit; Method: r in experiments); everything else stays frozen. When adopting threshold 0.5 and the first-proposal rule (2026-10-18), the user approved the edits they require: the Abstract's run-1 numbers ("In one run with gpt-5-nano ...") plus one sentence on gpt-5.4-mini (mean of three runs), Figure 1's threshold note, the threshold sentence in 3.2, Algorithm 1's return line, and Section 3.3; this replaces the earlier choice to leave the Abstract's numbers unchanged.

- Call the proposed method **JBFC** after its first expanded definition. Use **JBFC-F** and **JBFC-A** for function-only and argument-only ablations, and **JBFC-L** for the latest-feedback comparison; Jev remains the verifier model name. Typeset every occurrence of these method acronyms in bold, including running text, figure captions, and generated tables. Keep paper and generated table labels consistent.
- The study centers on **JBFC** (`jev_joint`): an LLM proposes a function and arguments, Jev asks two Noul correctness questions about that same call, and failed checks guide the same LLM's next proposal. For no call (calls=[]), ask only the function question; mark arguments_skipped=True with argument verdict/probability null, and apply only the function threshold. A selected function with missing arguments still receives argument checks. The argument-only ablation has no semantic gate for no call. The seven methods are direct, schema_only, self_refine, jev_function, jev_arguments, jev_joint, and jev_joint_latest. Direct function selection with Jev Choice is outside the experiment; preserve accurate source notes about Jev's capabilities without presenting them as implemented comparison arms.
- Leave the original conversation unchanged for the initial proposal. On repairs, **JBFC** serializes all previous proposals and their actual feedback in chronological order into one added user-context message before the original conversation and user request. **JBFC-L** (`jev_joint_latest`) sends only the latest rejected proposal and its actual feedback, retaining its original round number. All other methods retain cumulative repair history. Preserve all original messages and their order, keep every round in the saved trace, and explicitly mark previous proposals as NOT executed. Never replace the original request or invent Jev rationales or gold-derived hints.
- **JBFC-L** uses the same models, applicable checks, thresholds, shared initial proposal, fixed repair limit (the paper's r=3), and final dispatch policy as **JBFC**. Prompt histories first differ at the second repair; r <= 1 cannot assess the contribution of accumulated feedback. Run separate actual LLM trajectories; do not synthesize latest-only results from cumulative records.
- The feedback loop may be compared to using observations in ReAct, but this experiment uses pre-execution Jev judgments and does not reproduce ReAct's full reasoning/action/environment-observation procedure.
- Typeset the model display name as \textit{Jev} throughout the manuscript, including headings and figures. Preserve the established formatting of the method name Jev-Backed Function Calling, cited paper titles such as JEV-as-a-Judge, and literal API model identifiers such as jev-1.13.0.
- Typeset the primitive name Noul consistently as \texttt{Noul} throughout the manuscript, including reference titles. Preserve literal URLs and citation keys.
- Never fabricate benchmark values or promote synthetic/demo outputs into a paper.
- Never send ground_truth, category labels, example IDs, offline scores or gold-derived feedback to models.
- Keep the pinned BFCL inputs and official evaluator unchanged; document annotation anomalies.
- The paper's tables use one repair limit for every repair method and run all 3,051 prepared examples (once per run; Table 1 averages three runs) (user decisions 2026-10-17 and 2026-10-18). The limit was r=2 until 2026-10-18; the user then chose r=3 from the threshold-0.8 repair-limit curve (the same examples) as the point after which each extra point costs about four times as much, so the paper must say that r was chosen on these examples. The threshold (0.5, after reruns at 0.7 and 0.5) and the first-proposal rule were also chosen after seeing results on these examples (user decision 2026-10-18); the paper says so in 4.1 and Study Limits. There is no development split. The library default (src/jbfc max_repairs) follows the paper's r. Direct and the repair methods use identical generator settings (gpt-5-nano, reasoning effort none, 4,096 output tokens). r repairs permits at most r+1 LLM proposals; re-evaluate all applicable **JBFC** checks on every new candidate rather than carrying forward a previous pass. The development-selection tools (jev_fc.tuning, scripts/run_suite.py, scripts/tune_repair_budget.py) are optional exploration and never fill paper tables.
- Figure 2 (formerly Table 3; its files keep the table3 name: make table3, scripts/table3.py, results/table3_*) varies JBFC's repair limit over 1, 2, 3, 5, 7 and 10 as a pgfplots plot of call and function accuracy (top) and cost (bottom). Build it by extending the Table 2 run's exhausted JBFC trajectories (scripts/extend_run.py: same source_sha256, models, interface and prices; the saved prefix is replayed through the pipeline and must match the record). The r=10 run continues the r=7 extension, so reported rows up to r=7 never change. scripts/table3.py cuts the final run at each limit and checks every run in the chain: the cut at each source's limit must equal that run's JBFC rows (r=2: the Table 2 run; r=7: the r=7 extension). Do not fill Figure 2 from separate runs per limit. The paper's Figure 2 is JBFC at threshold 0.5 (results/jbfc_t0.5_gpt-5-nano_run1) continued from r=3 to r=10 (make t05: results/table3_gpt-5-nano_r10_t0.5), cut at each limit and given the first-proposal rule by scripts/final_tables.py; its r=3 cut must equal Table 2's JBFC. The threshold-0.8 chain (results/table3_gpt-5-nano_r10) stays as the record of how r was chosen. Keep src/jev_fc unchanged between the Table 2 run and its extensions. An extension may route Jev differently only on request (JEV_GATEWAY / --jev-gateway; make table3 sends Jev directly with TYPESAFE_API_KEY while the generator stays on the gateway): the Jev model, thresholds and questions must not change, the manifest's jev_routes records the route from each attempt on, and the Figure 2 caption states it.
- Table 1 shows Direct and JBFC for gpt-5-nano and for the larger, costlier gpt-5.4-mini (user requests, 2026-10-18), with a USD/1k row. gpt-5.4-mini ($0.75/$0.075/$4.50 per million uncached/cached/output tokens) first ran Direct only (make compare: scripts/run_compare.sh, results/direct_gpt-5.4-mini); make r3 (scripts/run_r3.sh) adds JBFC from its saved first proposals. Both models use the same source_sha256, data, interface and settings; only the model and its prices differ (reasoning effort none; a fallback would be stated in the caption). scripts/raise_limit.py raises a run's limit by reusing every recorded attempt: direct rows stay; stop-at-first-pass methods continue only rows that reached the old limit; every self-refine row continues; --reuse METHOD=RUN takes rows from a longer run cut at the new limit (JBFC from the Figure 2 extension, so Table 2 agrees with Figure 2); --add METHOD starts a missing method from the shared first proposals. Replayed prefixes must match the record. scripts/paper_tables.py checks both runs and summarizes them (results/tables_r3.json); the paper's tables come from make final (make report is the same). Keep src/jev_fc unchanged so the hashes match. Jev may answer directly (TYPESAFE_API_KEY) when the gateway cannot; the manifests record the route. Report the comparison as measured, including rows where the larger model is better, and keep the 4.2/4.3 sentences naming the models and prices in sync with the runs.
- Table 1 over three runs (user request, 2026-10-18, after the runs looked inconsistent): run 1 of each generator is its r=3 run (make r3; for gpt-5-nano also the source of Table 2 and Figure 2); runs 2 and 3 (make repeat: scripts/run_repeat.sh) repeat Direct with new first proposals and add JBFC at r=3 with scripts/raise_limit.py --add jev_joint, with the same code (source_sha256), data, interface, settings and prices. scripts/table1_repeat.py checks every run (run 1 against the paper's protocol, later runs against run 1, and that each run generated its own first proposals), then writes Table 1 as the mean of the runs with the total's standard deviation across runs and the mean cost, and reports each run's values and paired differences that average each example over the runs before resampling examples. The paper's Table 1 is the threshold-0.5 rerun of each of these runs with the first-proposal rule (make final; scripts/table1_repeat.py still reports the threshold-0.8 runs and no longer writes the paper). Table 2 and Figure 2 stay single-run (run 1); the paper must say which numbers are means. Keep all runs: never drop, replace or rerun one after seeing its result.
- Threshold experiments (user request, 2026-10-18; make threshold: scripts/run_threshold.sh, scripts/rethreshold.py, scripts/threshold_compare.py) rerun only JBFC with another Jev threshold for both checks, from each Table 1 run's first proposals, so Direct and JBFC at 0.8 pair with the rerun example by example. Code, data, generator settings, prices, r and the exhaustion policy must equal the run's; Jev's route is recorded. The threshold also appears in the feedback the generator sees, so a different threshold needs new generations after the first rejection and cannot be replayed from records. The user adopted threshold 0.5 after these runs (2026-10-18), so the 0.5 reruns fill the paper's tables and the README through make final; other thresholds stay in docs/threshold.md. A threshold chosen after seeing these examples must be reported as such. scripts/threshold_table.py (make threshold-table; make threshold runs it at the end) writes docs/threshold.md from every rerun on disk; it is generated, so change the script, not the file.
- All seven methods share one initial LLM proposal per example (and run). Charge that proposal once to each method's standalone cost, along with its own later generation and verification work, including the feedback tokens actually sent. The paper grid is 3,051 examples x 1 run x 7 methods = 21,357 records (make table2); make table1 runs Direct and JBFC only (6,102 records). Paper tables accept only complete runs matching the declared protocol, models and prices with all 3,051 IDs and the dataset digest; a seven-method run fills Tables 1 and 2 together so their Direct and JBFC values agree. Generated tables bold the highest accuracy (Table 1: per row across its four columns; Table 2: per accuracy column).
- summary.json's history_ablation reports **JBFC** minus **JBFC-L** final-call accuracy. Match (example ID, run), average run differences per ID when there are several runs, then bootstrap IDs 2,000 times with seed 1729 for the 95% interval. Exclude live_relevance and retain no-call cases: the complete paper run has 3,035 paired IDs. Validate the shared initial proposal and score, repair limit, and dispatch policy.
- Generator responses cut off at the output limit, refused, or blocked by the provider's content filter (including HTTP 400 content_filter) are unreadable proposals: calls=None with the reason in proposal.incomplete, scored wrong, repaired with local feedback, and counted in summaries. Other API errors still stop the run and record a sanitized provider error code.
- Only the final dispatch boundary can invoke an application callback. Benchmark runs do not execute tools.
- The manuscript returns the first passing proposal or, when none passes within the limit, the first proposal (keep_first), and scores it without running functions; Table 3 compares returning the last proposal (execute_last), on the same trajectories. Runs record execute_last; the rule only changes which recorded proposal is final, so scripts/final_tables.py derives it exactly for every method that stops at a passing proposal (calls and score from attempt 0, final_attempt=0; trajectory, acceptance, repairs, time and cost unchanged); Direct and self-refine keep their last proposal. Keep Section 3.3 focused on this final proposal rule; omit storage inventories, manifests, callbacks, and optional abstention policies from the manuscript. Shared initial proposals and their costs belong in Experiments.
- src/jbfc is the public library (`import jbfc`, documented in the English README). It reuses jev_fc's repair loop (run_example), Jev client and generator prompt read-only, and its defaults are the paper's final settings (max_repairs=3, threshold 0.5, on_exhaustion="first"; "last" returns the last repair); tests/test_jbfc.py checks prompt parity with jev_fc. Library work must not change src/jev_fc while a run or its extension depends on the source hash. Keep README numbers equal to the filled paper tables (scripts/analyze_results.py prints the Results numbers); README charts and the hero image are generated from docs/assets/src (data.json must match Tables 1-2 and Figure 2). Do not headline performance gains in the README (user request, 2026-10-18): no result numbers in the hero image, badges, tagline or feature cells; results appear only in the Results section, after the API sections, opened by the study's scope (one generator, one run, BFCL subset) and stated without bold or superlatives.
- Never print, commit or package API keys or .env. Keep credential fields in .env.example blank. APIM uses only the project subscription key in api-key; do not forward provider credentials. Preserve API format/routing in run fingerprints and keep unknown pricing unavailable.
- Run appropriate offline tests after code changes. Use the native LaTeX editor for the standalone manuscript,
  and verify the exported PDF has four body pages with references beginning on page five (six pages maximum).
- Whenever paper/main.tex changes, compile it in the native editor and export the updated paper/main.pdf before replying. Keep distributed source/PDF copies in sync; updating only the editor preview is not enough.
