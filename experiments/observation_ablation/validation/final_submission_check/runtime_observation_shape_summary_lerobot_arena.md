# One-environment runtime observation validation — lerobot-arena

## Result

**RUNTIME / CONFIG / CHECKPOINT INTERFACE CONSISTENCY = PASS**

| Task | Runtime policy tensor | Config / preserved manifest | Checkpoint actor input | Selected iteration | Triple match |
|---|---|---:|---:|---:|---|
| `Ant-rl-Ablation-HeightScan-v0` | `[1, 122]` | 122 | 122 | 804 | PASS |
| `Ant-rl-Ablation-HeightScan-Contact-Stock-v0` | `[1, 126]` | 126 | 126 | 972 | PASS |
| `Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0` | `[1, 126]` | 126 | 126 | 788 | PASS |

## Runtime and method

- Python `/home/zxro/miniforge3/envs/lerobot-arena/bin/python`; Conda `lerobot-arena`; Isaac Sim package `5.1.0.0`; submission Isaac Lab `2.3.0`.
- Three independent serial headless AppLauncher processes; exactly one environment in each. Framework and ant import origins are verified inside the submission source tree. No LD_PRELOAD.
- Config seeds: environment 42, terrain 42. This is an initialization/interface check, not the canonical evaluation (seed 24, 100 environments).
- Selected checkpoints loaded on CPU with weights_only=True; iterations 804, 972, 788. Actor and critic input dimensions agree. Model tensors are finite and SHA256 values match the preserved checkpoint-selection records. No checkpoints were resaved.
- One env.reset per task obtains the policy tensor; all float32, cuda:0, finite, with zero NaN/Inf. Runtime observation term order agrees with each preserved manifest.

## Observation terms

| Term | Dimension |
|---|---:|
| base_lin_vel | 3 |
| base_ang_vel | 3 |
| base_yaw_roll | 2 |
| base_angle_to_target | 1 |
| base_up_proj | 1 |
| base_heading_proj | 1 |
| joint_pos_norm | 8 |
| joint_vel_rel | 8 |
| feet_body_forces | 24 |
| actions | 8 |
| height_scan | 63 |
| foot_contacts (Contact tasks only, after height_scan) | 4 |

## Initialization and physics activity

No env.step, policy inference, rollout, episode or training was performed. Environment control-step counters remain zero; Python SimulationContext.step calls are zero.

This does **not** mean physics was untouched: normal environment construction calls sim.reset, activates physics and runs the physics warm-start callback once per process. The runtime initialize_physics implementation performs start_simulation/update_simulation(dt,0)/fetch_results (simulation_manager.py:235-242). Reset also calls sim.forward and initialization performs rendering updates. These required initialization operations were not bypassed or patched. The exact native physics update count was not instrumented.

## Shutdown

| Task | env.close returned | App close return marker | Exit code | External termination |
|---|---|---|---:|---|
| Ant-rl-Ablation-HeightScan-v0 | Yes | Not observed | 0 | No |
| Ant-rl-Ablation-HeightScan-Contact-Stock-v0 | Yes | Not observed | 0 | No |
| Ant-rl-Ablation-HeightScan-Contact-ModifiedReward-v0 | Yes | Not observed | 0 | No |

Observation evidence was flushed and fsynced before shutdown. All three processes exited with code 0 without watchdog termination; normal env.close returned. SimulationApp.close was invoked, but no subsequent return marker was written. Thus this check establishes successful process exit, not a confirmed return from SimulationApp.close. Prior bare-runtime hangs/crashes remain preserved and are not considered resolved. No source or runtime shutdown workaround was applied.

## Evaluation readiness

The three HeightScan tasks are technically ready for a separately approved 100-environment evaluation **with respect to source/config/checkpoint observation interfaces**. This initialization check does not validate long-run stability, evaluator execution or performance. Preserve the canonical evaluation seed 24, terrain seed 42, deterministic mean actions, first-episode protocol and task-specific reward. Use separate output paths and retain the known runtime shutdown limitation. No 100-environment evaluation has been run.

## Preservation

All three original repositories retain identical HEAD, branch, status and remotes. Original project/framework file hashes and mtimes match the recorded inventories. Submission files outside final_submission_check retain identical hashes, including source, checkpoints and canonical results; no files were added outside that validation directory. See limited_observation_preservation_after.json.

No Git initialization, commit, remote configuration, push, training or canonical-result overwrite occurred.
