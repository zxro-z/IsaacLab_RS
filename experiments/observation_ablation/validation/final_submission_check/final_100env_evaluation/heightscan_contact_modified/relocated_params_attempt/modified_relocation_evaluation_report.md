# Modified evaluator config relocation and 100-env reproduction

## Provenance and layout

Original runtime path: `logs/rsl_rl/ant/v3_depth/params/`. Submission path: `logs/rsl_rl/ant/modified/params/`.

Only env.yaml (20,229 bytes) and reward_weights.yaml (217 bytes) were copied from the validated project repository. Both SHA256 values directly match canonical training/evaluation source_mapping.json records. They remain byte-identical to their source. No legacy run/checkpoint sequence was restored. Mapping is documented in docs/submission_provenance/modified_config_mapping.json.

Only four active path literals changed in scripts/stage2_observation_ablation.py: two YAML reads and two source-mapping entries. Reverse replacement reproduces the evaluator pre-change hash exactly. Source/ant, reward values, environment values, observation, task, checkpoint, seed, terrain seed, horizon, deterministic inference, rollout and metric calculations remain unchanged. Historical manifests/logs and the previous failed evaluation are preserved.

## Preflight

PASS: inert saved YAML parsing, canonical reward weights/TotalReward mapping, hashes, evaluator Python syntax, provenance fallback and selected checkpoint iteration 788 with actor width 126.

## Evaluation

Task: `Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0`. Selected best_model.pt: iteration 788. Checkpoint SHA256: `0471caaadd646feac85cb7bbed0d2c3668279d2d65db9507030a649895ea11b4`.

Actual runtime policy tensor: `[100, 126]`, float32, cuda:0; loaded actor first layer `[400, 126]`. Runtime/config/checkpoint interface agrees. Seed 24, terrain seed 42, 100 envs, deterministic mean action, first episode only, 16 seconds / 960 control steps, dt=1/60.

| Metric | Result |
|---|---:|
| Completed episodes | 100/100 |
| Return | 134.8413 ± 68.0187 |
| Episode steps | 727.1200 ± 340.8912 |
| Episode duration | 12.1187 ± 5.6815 s |
| Displacement | 56.5627 ± 28.3639 m |
| Mean vx | 4.0718 m/s |
| Fall | 43/100 (43%) |
| Timeout | 57/100 (57%) |
| Other | 0/100 |
| ≥2m / ≥5m / ≥10m | 86 / 86 / 84 |
| Out-of-terrain-X | 26/100 |

All standard deviations are population std. Metrics, decomposition and residual summary exactly match the preserved canonical evaluation_summary.json, and episode_metrics.csv is byte-identical. Classification: **REPRODUCED**.

Exit code 0, no external kill or timeout. env.close returned; app.close was invoked, but its return marker was not observed. Existing runtime shutdown limitation is not declared fixed.

## Integrity and current status

PASS: post-evaluation hashes unchanged. Relative to the pre-relocation snapshot, only the authorized evaluator path edit, two copied YAMLs and mapping document changed outside the validation output. Source, checkpoints, historical canonical results and previous Stock evaluation outputs remain unchanged. Original repositories retain identical Git state; project/framework recorded hashes and mtimes also remain unchanged. JSON/CSV/numeric finiteness checks pass.

All three canonical HeightScan evaluations have now completed and reproduced their canonical outputs, using the two previous Stock results and this Modified-only rerun. Root README runtime/command cleanup is a separate remaining documentation task; no broad README edit was made here.

No Git initialization, commit, remote setup, push, training or DepthCam evaluation occurred.
