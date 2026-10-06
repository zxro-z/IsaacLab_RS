# Final review before committing on main

This is a new verification record for the requested commit/push review. Earlier files in the parent cleanup directory, including `removed_files.json`, `modified_files.txt`, `git_status_short.txt`, `git_diff_stat.txt`, and the original runtime reports, remain unchanged historical records. Their statements about not committing/pushing describe the earlier cleanup operation. This verification does not claim a commit or successful push before those actions occur.

## Review

The staged/unstaged review started on `main`, tracking `origin/main`, with no staged changes. Origin is `https://github.com/zxro-z/IsaacLab_RS.git`. All 17 tracked modifications and 73 tracked deletions are intentional cleanup changes. The original cleanup inventory's 186 removed files comprise **73 Git-tracked files plus 113 untracked or ignored recovered artifacts**. Their counts describe different file sets; no files were changed to reconcile the numbers.

The RayCaster torso attachment, yaw frame, offset (0.8, 0, 20), 9×7 grid, 0.2 m spacing, 63 rays, observation construction, scale 1, clipping [-1, 1], and absence of empirical normalization remain unchanged. Contact uses the same ordered four feet, current world-frame net-force norm >1 N, history 0 and float32 binary encoding. Reward-side contact remains a separate definition. `ant.rewards.TotalReward`, contact helpers and shared observation functions remain byte-identical. Video rendering cameras and the canonical deterministic evaluation scripts are preserved.

The narrative is **Baseline → HeightScan → HeightScan + Contact → HeightScan + Contact + Modified Reward**. A matching-budget baseline result is unavailable and explicitly documented. Single-seed evidence, exploratory-budget differences, fixed initial terrain assignment, disabled curriculum, map-boundary excursions and reward-scale limitations remain visible. The modified-reward result is not claimed to prove improved robustness.

## Validation

- `git diff --check`: PASS; new files also pass whitespace checks.
- All 12 modified Python files and the new smoke utility compile (13 files).
- Import/config smoke rerun: all five project Ant tasks PASS; no environment creation, training, physics steps or evaluation rollout. Evidence: `runtime_smoke.json` and `runtime_smoke.log`. The process exited with code 0; a separate application-close return marker was not recorded.
- Policy/scanner config exactly matches the retained saved config; checkpoints are loaded read-only on CPU, with finite model tensors and matching SHA256 values.
- All **473 protected retained experiment/log/validation/transfer files** remain byte-identical. Full pre-edit inventory reconciliation: 2310 original files unchanged, 17 intentionally modified, 186 intentionally removed.
- **0 active submission-facing/project DepthCam policy references.** The 330 tracked terminology matches were matched against the original reviewed records and remain historical/provenance evidence or unrelated upstream functionality. New cleanup inventories and verification notes intentionally name removed files; they are evidence, not active workflows.
- No visible untracked caches, bytecode, editor files or OS junk would be staged. Python compile output was directed to an ignored cache.

| Retained policy | Task/config | Input | Selected iteration |
|---|---|---:|---:|
| HeightScan + Stock | `Ant-rl-Ablation-HeightScan-v0` / `AblationHeightScanCfg` | 59 + 63 = **122-D** | 804 |
| HeightScan + Contact + Stock | `Ant-rl-Ablation-HeightScan-Contact-Stock-v0` / `HeightScanContactStockCfg` | 59 + 63 + 4 = **126-D** | 972 |
| HeightScan + Contact + Modified Reward | `Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0` / `Stage2HeightScanContactCfg` | 59 + 63 + 4 = **126-D** | 788 |

Checkpoint paths remain `experiments/observation_ablation/<variant>_4096x32x1000/checkpoints/best_model.pt`, where variants are `heightscan`, `heightscan_contact_stock`, and `heightscan_contact_modified`. No architecture or checkpoint file was changed.

## Rerunnable smoke evidence

The smoke utility now accepts `--output-dir`, so future checks can write to a fresh directory without replacing immutable earlier reports. This is the sole additional code adjustment during final review; no observation or reward implementation was changed.

Activate the documented `lerobot-arena` environment, set the root README's source paths, then run:

```bash
python docs/submission_provenance/heightscan_cleanup/smoke_configs.py --headless \
  --output-dir docs/submission_provenance/heightscan_cleanup/new_verification
```

Use a directory without an existing `runtime_smoke.json`; that report is created exclusively and never overwritten. `integrity_and_compile.json` records the new integrity/compiler verification; `tracked_deletions.txt` lists Git's 73 deletions. Pre-commit status/stat snapshots below supplement the parent directory's earlier immutable snapshots.
