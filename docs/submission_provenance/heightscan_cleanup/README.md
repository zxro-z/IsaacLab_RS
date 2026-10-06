# HeightScan submission cleanup record

The submission now follows **Baseline → HeightScan → HeightScan + Contact → HeightScan + Contact + Modified Reward**. Cleanup removes policy depth perception, its training/evaluation artifacts and comparison narrative. It does not train, roll out policies, overwrite evaluation outputs, resave checkpoints, commit or push.

## Removed

The complete inventory of **186 deleted files**, including previously untracked recovered artifacts, is in [removed_files.json](removed_files.json). Deleted roots:

- `source/ant/depth_obs.py`
- `source/ant/depth_actor_critic.py`
- `scripts/evaluate_depthcam_ablation.py`
- `cross_eval_team1_policy/` — entirely devoted to the depth-image policy, including its supplementary cross-host evaluations, tables and scripts
- `experiments/observation_ablation/depthcam_2048x32x2000/`
- `experiments/observation_ablation/depthcam_modified_2048x32x2000/`
- `experiments/observation_ablation/validation/final_submission_check/final_100env_evaluation/depthcam_stock/` — pending depth-policy evaluation diagnostics only
- `logs/rsl_rl/ant/base_depth/`
- `logs/rsl_rl/ant/contact_depth/`
- `docs/submission_provenance/depthcam_artifact_recovery.json`

No separate depth-only figures remained outside those roots. The retained transfer-evaluation videos are in `validation_compare/`. Upstream Isaac Lab camera sensors, examples, Cartpole tasks and documentation belong to the framework snapshot and remain intact.

## Modified

The exact list is in [modified_files.txt](modified_files.txt):

| File | Change |
|---|---|
| `README.md` | Submission narrative, staged matrix, missing-baseline caveat, retained results and reproduction commands |
| `experiments/observation_ablation/README.md` | Staged research questions, tables, HeightScan implementation, retained Contact/reward analysis and caveats |
| `docs/submission_provenance/README.md` | Distinguish immutable historical evidence from current source |
| `source/ant/ant_env_cfg.py` | Remove policy camera and image observation group; preserve dynamics, proprioception, contacts and rewards |
| `source/ant/agents/rsl_rl_ppo_cfg.py` | Remove CNN configuration and depth observation groups; retain PPO and MLP settings |
| `source/ant/ablation_env_cfg.py` | Remove redundant disabling of deleted depth fields |
| `source/ant/stage2_env_cfg.py` | Describe the final HeightScan + Contact reward stage |
| `source/ant/__init__.py` | Register existing stock-reward baseline foundation |
| `source/ant/agents/ablation_budget_ppo_cfg.py` | Give the baseline a distinct run name |
| `scripts/reinforcement_learning/rsl_rl/train.py` | Remove depth-policy registration; enable cameras for video |
| `scripts/reinforcement_learning/rsl_rl/play.py` | Remove depth-policy registration/export branch; preserve video recording and MLP export |
| `scripts/evaluate_contact_teammate1.py` | Remove assignment to deleted camera field; preserve transfer evaluation |
| `scripts/finalize_observation_ablation.py` | Remove depth comparison columns and check against deleted depth-policy evaluation |
| `scripts/finalize_observation_ablation_budget.py` | Baseline/HeightScan narrative and tables |
| `scripts/finalize_stage2_observation_ablation.py` | Final reward stage narrative and tables |
| `experiments/observation_ablation/protocol.json` | Remove obsolete comparison metadata and reference to deleted evaluation |
| `experiments/observation_ablation/shared/stage2_contact_modified_protocol.json` | Final reward-stage comparison metadata; retain experimental settings |

The original protocol bytes are preserved as [historical_protocol.json](historical_protocol.json) and [historical_stage2_contact_modified_protocol.json](historical_stage2_contact_modified_protocol.json). These copies are evidence, not current protocols. The new smoke-check script and reports in this directory are independent cleanup validation.

## Preserved evidence and historical references

[before_sha256.json](before_sha256.json) inventories the pre-edit file contents, including the user's existing README edits and recovered artifacts. [initial_git_status.txt](initial_git_status.txt) records the initial dirty state. Existing historical copy/hash manifests, validation reports, command logs, original README snapshot and saved training configs are unchanged. Their recorded PASS statuses and hashes apply to the snapshot when they were created; removed files and modified source intentionally differ now. Do not use those historical inventories as a current-file existence checklist.

The preserved `logs/rsl_rl/ant/modified/params/env.yaml` includes the original policy-camera configuration. The final HeightScan evaluator reads its reward provenance and hash, not its historical camera/observation interface. Deleting or rewriting that YAML would break the reward-source evidence. Historical `v3_depth` names identify the origin of the reused reward implementation, not active depth sensing.

The [remaining reference review](remaining_reference_review.json) records every match and its classification for the terminology and symbol search outside this audit directory: historical snapshots/evidence, upstream framework functionality, or reward-source provenance. The audit inventories themselves necessarily name removed files. There is no active project depth observation, CNN, registration, preprocessing or evaluator. `validation_compare/postprocess_source.py` remains as the original transfer-result provenance generator; its description of replacing original observations is historical and it does not consume depth input.

HeightScan, HeightScan + Contact, Modified Reward, selected/final checkpoints, all non-depth training logs, canonical and final evaluations, transfer evaluation, bootstrap evidence, validation logs and integrity manifests remain intact. [validation.json](validation.json) confirms **473 protected files** are byte-for-byte unchanged. Across the full pre-edit inventory, 2310 files are unchanged, 17 files are intentionally modified and 186 files are intentionally removed.

## Validation and limits

- `git diff --check`: PASS.
- Python compilation: PASS for all 12 modified Python files and the new smoke check (13 total).
- Import/config smoke: PASS for all five project task registrations. No environment creation, physics steps, policy rollout or training.
- Actor/critic inputs: native legacy/baseline **59-D**, HeightScan **122-D**, both Contact variants **126-D**. The three retained selected checkpoints match these inputs, [400,200,100] MLP layers, 8 actions, iterations 804/972/788 and recorded SHA256 values. Model tensors are finite; no architecture conversion or checkpoint writes.
- HeightScan policy terms and full scanner configuration match the saved experiment configs exactly. Runtime tensor shapes were checked in the preserved earlier validation; the new check validates imports/configs/checkpoints and does not repeat environment initialization.
- Final [runtime_smoke.json](runtime_smoke.json): PASS. The two earlier attempts failed in the validation harness because Python slices needed conversion to their saved JSON representation; they remain as diagnostics. No sensor/policy behavior was changed to pass the check. All three processes exited normally; no separate application-close return marker was recorded.
- Rendering/video support remains; no new video or full simulation evaluation was run.

A matching-budget baseline checkpoint/evaluation is **unavailable**. Its config is registered for clarity, without inventing measurements. The baseline shared environment defaults to 2048 environments; use `--num_envs 4096` to match the canonical 4096×32×1000 budget. The earlier exploratory HeightScan run has a different budget and is not a matched baseline. The stock framework Ant has a different 60-D interface and is not substituted.

The retained runs use one training seed, fixed initial terrain assignment, disabled curriculum and the documented native reset behavior. No missing baseline seed, training duration or reset behavior is inferred. Map-boundary excursions limit interpretation of displacement. Stock/Modified returns use different objectives/scales, and the modified-reward condition has more falls in this seed; development intent does not establish a monotonic performance improvement.

Historical finalizers and supplementary evaluators retain original runtime paths and protected-source hashes and are not portable reproduction entry points. They were compiled but not executed. Use the root README's canonical reproduction commands and fresh output directories.

## Git summaries

Full outputs of `git status --short` and `git diff --stat` are in [git_status_short.txt](git_status_short.txt) and [git_diff_stat.txt](git_diff_stat.txt). The diff summary covers tracked changes; recovered untracked run/checkpoint removals appear in the deletion inventory rather than Git's diff. New cleanup evidence is untracked until the user stages it.
