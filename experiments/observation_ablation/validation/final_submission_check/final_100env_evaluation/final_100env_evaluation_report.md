# Canonical 100-env evaluation validation

## Provenance-only compatibility patch

Four evaluator files changed only in a sibling-helper import and manifest Git metadata call: observation_ablation.py (historical evaluator; not executed), observation_ablation_budget.py, contact_stock_observation_ablation.py, stage2_observation_ablation.py. New helper: scripts/submission_provenance.py. Finalizers and source/ant are unchanged. See provenance_only_compatibility.patch.

Git HEAD is retained when available. Only evaluation mode falls back on the specific not-a-git-repository error. New manifests record original project HEAD, git_commit_source=preserved_submission_provenance, repository_snapshot=true. Preflight, syntax and AST comparison (excluding only these metadata changes) passed. Historical manifests remain unchanged.

## Results

| Condition | Episodes | Runtime observation | Return mean ± population std | Steps mean ± std | Displacement mean ± std (m) | Fall | Timeout | Exit |
|---|---:|---|---|---|---|---|---|---:|
| heightscan_stock | 100/100 | [100, 122] | 61.3354 ± 31.2305 | 731.6800 ± 335.5798 | 60.5140 ± 29.1880 | 41/100 | 59/100 | 0 |
| heightscan_contact_stock | 100/100 | [100, 126] | 63.5682 ± 28.5777 | 774.5300 ± 323.7330 | 61.8529 ± 27.8017 | 29/100 | 71/100 | 0 |
| heightscan_contact_modified | Not started | Not measured in this evaluation | — | — | — | — | — | 1 |

Both successful evaluations use seed 24, terrain seed 42, deterministic mean actions, first episodes only, 100 envs, 16 seconds/960 steps and dt=1/60. Their summaries and complete episode_metrics.csv are exactly identical to historical canonical outputs: **REPRODUCED**. This is observed output reproduction; it does not assert that every historical runtime package was independently identified.

Runtime widths are supported by the canonical assertions applied during evaluation, matching model Linear input widths and prior CPU checkpoint inspection. The passive diagnostic profiler emitted only its startup event after Kit initialization, so there is no independent profiler tensor dump or observed close-return marker. This instrumentation limitation is recorded rather than hidden.

Both Stock processes exited 0 without external kill or timeout. Normal close paths were reached by the evaluators, but return from SimulationApp.close was not independently observed; previous shutdown instability remains a limitation.

## Modified condition failure

The original dedicated evaluator fails before environment creation/checkpoint loading while reading logs/rsl_rl/ant/v3_depth/params/reward_weights.yaml (FileNotFoundError). The companion env.yaml is also absent. Its output directory exists but has no canonical result files. Exact traceback is preserved in heightscan_contact_modified/failure_traceback.txt. Exit 1 comes from the existing canonical exception handler; no new exit workaround was added. No physics/reward patch, skipped assertion, artifact fabrication, or semantic rerun occurred.

The required next step is to approve copying only these two real saved params files from the validated source repository after verifying their provenance, or approve another exact evidence-preserving input mapping. Full legacy checkpoint sequences are unnecessary. This has not been performed.

## Integrity and readiness

Post-patch before/after hashes: PASS. Source/ant, custom runtime helpers, checkpoints, historical results and evaluator files did not change during evaluation. Comparing against the unpatched snapshot shows only the four approved evaluator changes and new helper. All three original repositories retain identical Git state. JSON/CSV parsing and numeric finiteness checks pass.

Overall: **2 of 3 evaluations complete; final submission validation remains incomplete.** Both Stock evaluator commands are validated for lerobot-arena; the Modified evaluator still has a missing-input blocker. Root README runtime text remains an older diagnostic description and was not changed. Do not mark all README commands validated or the submission ready for Git initialization yet.

No Git initialization, commit, remote setup, push, training or DepthCam evaluation occurred.
