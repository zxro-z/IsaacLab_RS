# IsaacLab_RS

Robotics Simulation Assignment 1 submission.

## 1. Project overview

This repository contains an Isaac Lab framework snapshot, custom Ant environments, RSL-RL training/play code, observation-ablation results, training logs, checkpoints, and validation evidence. The experiments study terrain perception, explicit foot-contact feedback, and reward shaping in PPO locomotion.

The framework base is **Isaac Lab v2.3.0**, commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`, exported from the official committed Git tree. Local framework modifications were excluded. Project files come from the validated `observation-ablation` branch at commit `4513d7cae5d4fdc6af9288349531e6b49422f7b1`.

See the [experiment analysis](experiments/observation_ablation/README.md) for numerical results and interpretation, and [submission provenance](docs/submission_provenance/README.md) for the four provenance tiers. The original framework README is [preserved separately](docs/submission_provenance/official_README.md).

## 2. Repository structure

| Course material | Location |
|---|---|
| Project environment, rewards, observations, PPO and task registration | [source/ant/](source/ant/) |
| Verified runtime helpers and stock Ant configuration | [Framework Ant namespace](source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/ant/) |
| RL training/play scripts | [scripts/reinforcement_learning/rsl_rl/](scripts/reinforcement_learning/rsl_rl/) |
| Canonical training/evaluation entry points | [scripts/](scripts/) |
| Canonical training logs and config snapshots | [logs/rsl_rl/observation_ablation/](logs/rsl_rl/observation_ablation/) |
| Experiment results, checkpoints and shared specifications | [experiments/observation_ablation/](experiments/observation_ablation/) |
| Supplementary validation | [cross_eval_team1_policy/](cross_eval_team1_policy/), [validation_compare/](validation_compare/) |
| Copy inventory and provenance | [docs/submission_provenance/](docs/submission_provenance/) |
| New submission checks | [validation/final_submission_check/](experiments/observation_ablation/validation/final_submission_check/) |

`source/ant/` intentionally remains separate from the stock Isaac Lab Ant namespace: project imports and saved configuration names use `ant.*`. The framework namespace provides stock functionality and the approved runtime helpers.

The supplementary validation directories are evidence archives, **not additional canonical training experiments**. Their original executable paths may require separate review before reuse.

## 3. Canonical experiments

| Experiment | Observation interface | Envs × steps/env × iterations | Transitions | Selected iteration | Available artifacts |
|---|---|---|---:|---:|---|
| HeightScan + Stock Reward | 122-D = 59 proprio + 63 HeightScan | 4096 × 32 × 1000 | 131,072,000 | 804 | Full; runtime/config/checkpoint verified; 100-env reproduced |
| HeightScan + Contact + Stock Reward | 126-D = 59 + 63 + 4 Contact | 4096 × 32 × 1000 | 131,072,000 | 972 | Full; runtime/config/checkpoint verified; 100-env reproduced |
| HeightScan + Contact + Modified Reward | 126-D = 59 + 63 + 4 Contact | 4096 × 32 × 1000 | 131,072,000 | 788 | Full; runtime/config/checkpoint verified; 100-env reproduced |
| DepthCam + Stock Reward | 123-D actor feature input; 48 × 64 depth image | 2048 × 32 × 2000 | 131,072,000 | 905 | Results/provenance only |
| DepthCam + Modified Reward | Same DepthCam interface; no separate 4-D Contact | 2048 × 32 × 2000 | 131,072,000 | 1974 | Results/provenance only |

The original canonical DepthCam checkpoint binaries and training-run directories are not present in the validated source checkout and have not been fabricated or replaced.

The comparison stages are:

1. **Terrain perception:** HeightScan + Stock vs DepthCam + Stock.
2. **Contact feedback:** HeightScan + Stock vs HeightScan + Contact + Stock.
3. **Reward shaping:** HeightScan + Contact + Stock vs HeightScan + Contact + Modified.

DepthCam + Modified is a reference experiment rather than a controlled comparison against HeightScan + Contact + Modified. Stock and Modified returns use different objectives/scales and are not a direct performance delta. Equal transitions do not make the different parallel environment counts and PPO update counts identical.

[heightscan/](experiments/observation_ablation/heightscan/) is a historical/exploratory reference, not a canonical experiment.

## 4. Registered task IDs

The copied [task registration](source/ant/__init__.py) declares:

| Task ID | Role |
|---|---|
| `Ant-rl-v0` | Original project environment |
| `Ant-rl-Ablation-HeightScan-v0` | HeightScan + Stock |
| `Ant-rl-Ablation-HeightScan-Contact-Stock-v0` | HeightScan + Contact + Stock |
| `Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0` | HeightScan + Contact + Modified |

DepthCam stock/modified conditions are selected internally by the dedicated evaluator; no separate DepthCam task IDs are claimed. The HeightScan task registrations and config/checkpoint interfaces have passed runtime validation after AppLauncher initialization.

## 5. Training and evaluation settings

| Setting | Recorded value |
|---|---|
| Training seed | 42 in all five training summaries |
| HeightScan training/evaluation terrain seed | 42 in preserved manifests and evaluation configs |
| Evaluation environment seed | 24 for all five recorded evaluations |
| Evaluation environments | 100 |
| Inference | Deterministic mean action |
| Episode accounting | First episode only; terminal reward included, post-reset reward excluded |
| Maximum episode | 16 s / 960 control steps |
| Evaluation reward | Stock for Stock conditions; Modified for Modified conditions |

DepthCam terrain setup is described by the current source and evaluation evidence; its original training config snapshots are unavailable. Do not treat that configuration as reverified from an absent saved config.

The environment does not terminate at map boundaries. Displacement can include movement beyond generated terrain bounds. Results use population standard deviation and a single training seed.

## 6. Checkpoints and logs

| Experiment | Checkpoint directory | Original training folder under `logs/rsl_rl/observation_ablation/` |
|---|---|---|
| HeightScan + Stock | [Checkpoints](experiments/observation_ablation/heightscan_4096x32x1000/checkpoints/) | `ablation_heightscan_stock_s42_e4096_n32_i1000/` |
| HeightScan + Contact + Stock | [Checkpoints](experiments/observation_ablation/heightscan_contact_stock_4096x32x1000/checkpoints/) | `ablation_heightscan_contact_stock_s42_e4096_n32_i1000/` |
| HeightScan + Contact + Modified | [Checkpoints](experiments/observation_ablation/heightscan_contact_modified_4096x32x1000/checkpoints/) | `ablation_heightscan_contact_modified_s42_e4096_n32_i1000/` |

`best_model.pt` uses the predeclared highest logged training completed-episode mean-return selection rule, without evaluation-based selection. `final_model.pt` in each experiment directory is the preserved copy of that run's `model_999.pt`. Original `model_*.pt` names remain unchanged in the training runs. TensorBoard events and JSON agent/config snapshots are retained; no missing YAML params were invented.

## 7. Verified runtime and setup

The final submission evaluations were reproduced with:

| Component | Verified version |
|---|---|
| Conda environment | `lerobot-arena` |
| Isaac Sim | 5.1.0 (installed package 5.1.0.0) |
| Isaac Lab | 2.3.0 |
| Python | 3.11.16 |
| PyTorch / CUDA build | 2.7.0+cu128 / 12.8 |

Isaac Sim and the configured Conda environment are external runtime prerequisites, not bundled binaries. The earlier `env_isaaclab_231` checks remain diagnostic history; they are not the final submission validation environment.

From the repository root, activate the verified environment and select this snapshot's framework and project sources:

```bash
conda activate lerobot-arena
unset LD_PRELOAD ISAAC_SIM_SITE_PACKAGES ISAAC_PATH CARB_APP_PATH EXP_PATH
export ISAACLAB_RS="$PWD"
export ISAACLAB_ROOT="$PWD"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD/source:$PWD/source/isaaclab:$PWD/source/isaaclab_assets:$PWD/source/isaaclab_tasks:$PWD/source/isaaclab_rl:$PWD/source/isaaclab_mimic"
```

This is the source-resolution mechanism used by validation. No editable installation was needed or performed. The root project packaging metadata has not been validated after overlaying framework packages; do not assume `pip install -e .` is a tested setup method. Isaac Lab task modules require AppLauncher/Kit initialization before runtime imports.

## 8. Canonical evaluation commands

Use these dedicated evaluators; general `play.py` changes evaluation rewards and must not replace them. Seeds, 100 environments, deterministic mean actions, first-episode accounting and the 16 s / 960-step horizon are fixed by each evaluator. Choose a **new output directory** for each run; existing directories are refused.

```bash
./isaaclab.sh -p scripts/observation_ablation_budget.py evaluate --headless \
  --checkpoint experiments/observation_ablation/heightscan_4096x32x1000/checkpoints/best_model.pt \
  --output experiments/observation_ablation/validation/reproduction_heightscan_stock

./isaaclab.sh -p scripts/contact_stock_observation_ablation.py evaluate --headless \
  --checkpoint experiments/observation_ablation/heightscan_contact_stock_4096x32x1000/checkpoints/best_model.pt \
  --output experiments/observation_ablation/validation/reproduction_contact_stock

./isaaclab.sh -p scripts/stage2_observation_ablation.py evaluate --headless \
  --checkpoint experiments/observation_ablation/heightscan_contact_modified_4096x32x1000/checkpoints/best_model.pt \
  --output experiments/observation_ablation/validation/reproduction_contact_modified
```

These evaluator entry points and arguments executed successfully in the final validation. Passive diagnostic launchers recorded runtime interfaces and lifecycle events without changing evaluator calculations. DepthCam's evaluator is preserved at [scripts/evaluate_depthcam_ablation.py](scripts/evaluate_depthcam_ablation.py), but final DepthCam execution remains unvalidated because canonical model binaries are unavailable.

The [provenance helper](scripts/submission_provenance.py) uses `git rev-parse HEAD` in a committed checkout. For evaluation in a snapshot without Git metadata, it reads the preserved original project HEAD and marks new manifests with `git_commit_source=preserved_submission_provenance` and `repository_snapshot=true`. Other Git errors are not hidden. Historical manifests remain unchanged.

### Modified config mapping

| Role | Path |
|---|---|
| Original runtime config location | `logs/rsl_rl/ant/v3_depth/params/` |
| Submission config location | [logs/rsl_rl/ant/modified/params/](logs/rsl_rl/ant/modified/params/) |

Only `env.yaml` and `reward_weights.yaml` were copied. Their contents and SHA256 values match canonical training/evaluation source records byte-for-byte. The active Modified evaluator uses the submission path. This is layout cleanup only: no environment values, reward values or evaluation behavior changed. Historical manifests/logs retain their original paths. See the [mapping and hashes](docs/submission_provenance/modified_config_mapping.json).

## 9. Reproduced 100-environment results

All three HeightScan conditions have **FULL ARTIFACTS**, **RUNTIME/CONFIG/CHECKPOINT INTERFACE VERIFIED**, and **100-ENV CANONICAL EVALUATION REPRODUCED** status. Both summary results and complete episode CSVs match their preserved canonical results exactly. Standard deviations below are population std.

| Metric | HeightScan + Stock | HeightScan + Contact + Stock | HeightScan + Contact + Modified |
|---|---:|---:|---:|
| Selected iteration | 804 | 972 | 788 |
| Policy observation | 122-D | 126-D | 126-D |
| Completed episodes | 100/100 | 100/100 | 100/100 |
| Return mean ± std | 61.3354 ± 31.2305 | 63.5682 ± 28.5777 | 134.8413 ± 68.0187 |
| Episode steps mean ± std | 731.6800 ± 335.5798 | 774.5300 ± 323.7330 | 727.1200 ± 340.8912 |
| Duration mean ± std (s) | 12.1947 ± 5.5930 | 12.9088 ± 5.3955 | 12.1187 ± 5.6815 |
| Displacement mean ± std (m) | 60.5140 ± 29.1880 | 61.8529 ± 27.8017 | 56.5627 ± 28.3639 |
| Mean forward velocity (m/s) | 4.4506 | 4.3692 | 4.0718 |
| Fall | 41/100 | 29/100 | 43/100 |
| Timeout | 59/100 | 71/100 | 57/100 |
| ≥5 m | 87/100 | 89/100 | 86/100 |
| Out-of-terrain-X | 28/100 | 27/100 | 26/100 |
| Reproduction | REPRODUCED | REPRODUCED | REPRODUCED |

Stock and Modified returns have different definitions/scales; their difference is not a direct performance improvement. The map-boundary and single-training-seed limitations described above still apply. Original experiment tables were not edited.

New evidence is separate from historical outputs:

- [HeightScan + Stock evaluation](experiments/observation_ablation/validation/final_submission_check/final_100env_evaluation/heightscan_stock/canonical_evaluator_output/)
- [HeightScan + Contact + Stock evaluation](experiments/observation_ablation/validation/final_submission_check/final_100env_evaluation/heightscan_contact_stock/canonical_evaluator_output/)
- [HeightScan + Contact + Modified evaluation](experiments/observation_ablation/validation/final_submission_check/final_100env_evaluation/heightscan_contact_modified/relocated_params_attempt/canonical_evaluator_output/)

## 10. Reproducibility notes

The evaluator runs completed and all three evaluation processes exited with code 0. `env.close()` returned in the instrumented environment checks and Modified rerun; the Stock final evaluations did not retain an independent close-return trace. No final `SimulationApp.close()` return marker was observed, so application-close completion is not claimed. Earlier bare AppLauncher, original-framework and pure Isaac Sim probes also showed shutdown instability outside this project's overlay. Detailed diagnostics are preserved in [final_submission_check/](experiments/observation_ablation/validation/final_submission_check/).

Historical commands, manifests, source mappings and diagnostic scripts retain original absolute paths as evidence. Archived supplementary evaluators and historical finalizers are not portable submission entry points and should not be executed without separate review. The root commands above select this repository through environment overrides; no file from the old unrelated repository was used.

See the [final integrity summary](docs/submission_provenance/final_submission_integrity.json) and [final path audit](experiments/observation_ablation/validation/final_submission_check/final_cleanup/path_audit.md) for the submission checks. Final checks do not retrain or overwrite canonical results.
