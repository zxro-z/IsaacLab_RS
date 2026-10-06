# Final 100-env evaluation preflight — blocked

The three canonical evaluators were inspected but not launched. All require `git rev-parse HEAD` from the submission root while building the manifest before checkpoint loading and the evaluation loop. The submission intentionally has no Git repository. The exact read-only preflight command returns 128:

```text
fatal: not a git repository (or any of the parent directories): .git
```

No source or Git state was changed to bypass this dependency. No 100-env environment was created, no policy evaluated, and no new numerical results exist. There is no runtime traceback because the failure was detected before launching the evaluators.

| Condition | Selected iteration | Expected observation | Eval seed | Terrain seed | Envs | Status |
|---|---:|---:|---:|---:|---:|---|
| Ant-rl-Ablation-HeightScan-v0 | 804 | 122 | 24 | 42 | 100 | BLOCKED / NOT RUN |
| Ant-rl-Ablation-HeightScan-Contact-Stock-v0 | 972 | 126 | 24 | 42 | 100 | BLOCKED / NOT RUN |
| Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0 | 788 | 126 | 24 | 42 | 100 | BLOCKED / NOT RUN |

All historical recorded evaluations already used 100 environments, seed 24, terrain seed 42, deterministic mean actions, first episodes only and a 16-second / 960-control-step maximum. Selected checkpoint hashes match the recorded summaries. Per-condition preflight JSON preserves planned commands, checkpoint hashes, source locations and conditions.

Historical-result reproduction categories cannot be assigned without new results; no MATERIAL DISCREPANCY in policy metrics is implied by a missing-Git packaging dependency. Previous one-env interface checks passed but do not establish 100-env execution.

Integrity before/after: PASS. Existing submission files, source, checkpoints, historical results, helper modules and evaluators remain unchanged; all new files are under this output directory. Original repositories retain identical Git HEAD, branch, status and remotes.

## Required next decision

A minimal metadata-only compatibility change could read the preserved Tier B project HEAD from docs/submission_provenance/copy_manifest.json when no local Git HEAD exists, explicitly label it source provenance, and retain normal Git lookup when available. This would not change physics, observations, reward, checkpoint choice or evaluator accounting. It has not been applied because source changes are prohibited. Git initialization or using another repository as implicit Git metadata has also not been used. Approval of a concrete compatibility approach is required before rerunning.

No Git initialization, commit, remote configuration, push, training, or canonical-result overwrite occurred.
