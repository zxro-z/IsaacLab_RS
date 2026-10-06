# Team 1 v3 depth policy — three-environment stock-reward evaluation

Repository에는 제출용 checkpoint가 명시적으로 지정되어 있지 않았다. 평가 command가 best_model.pt를 사용하고 v3_depth가 후속 depth-based policy이므로 v3_depth/best_model.pt를 대표 checkpoint로 사전 선택했다.

Selected checkpoint: `logs/rsl_rl/ant/v3_depth/best_model.pt`; iteration 6632; SHA256 `29a9474cc6e6d488b50ca33b9bc7ec99b516fdfc4313c3663d3843535990c2c5`. Selection was fixed before evaluation and unchanged.

## Protocol

Environment seed 24, 100 parallel environments per host, deterministic `act_inference`, first episode only. Terminal reward and terminal step are included; later reset episodes are excluded. Population standard deviation (`ddof=0`). Native host geometry, asset, physics, action mapping, initialization and termination are unchanged. The original 59-D proprioception and 48×64×1 depth observation, native normalization, 15 Hz camera and original depth-CNN are used on every host. No Contact 127-D adapter is used.

Primary total return uses official float32 accumulation; float64 trace and per-component accumulation are retained separately. Weighted components are read from RewardManager (`_step_reward × control_dt`); reward functions are not recomputed. Contributions are `raw × weight × dt`.

## Mean accumulated weighted contributions

| Component | FinalUnseen | Teammate 1 | Teammate 2 |
|---|---:|---:|---:|
| progress | 30.147 | 43.255 | 36.276 |
| alive | 5.425 | 6.002 | 7.574 |
| upright | 0.966 | 1.171 | 1.498 |
| move_to_target | 4.996 | 5.488 | 7.320 |
| action_l2 | -0.066 | -0.089 | -0.107 |
| energy | -4.438 | -5.066 | -5.461 |
| joint_pos_limits | -1.097 | -1.216 | -1.755 |
| **total** | **35.933** | **49.544** | **45.344** |

## Main metrics

| Environment | Return mean ± std | Displacement mean ± std (m) | Fall | Timeout | Other |
|---|---:|---:|---:|---:|---:|
| FinalUnseen | 35.933 ± 20.067 | 30.149 ± 17.050 | 61% | 39% | 0% |
| Teammate 1 | 49.544 ± 26.753 | 43.298 ± 23.933 | 43% | 57% | 0% |
| Teammate 2 | 45.344 ± 14.939 | 36.277 ± 15.193 | 5% | 87% | 8% |

| Environment | Steps mean ± std | Mean duration (s) | Mean forward velocity (m/s) | Return min / max | ≥5m reach count |
|---|---:|---:|---:|---:|---:|
| FinalUnseen | 651.62 ± 321.28 | 10.860 | 2.548 | 0.377 / 67.411 | 91 |
| Teammate 1 | 720.67 ± 352.45 | 12.011 | 3.113 | -0.263 / 79.043 | 85 |
| Teammate 2 | 908.93 ± 146.83 | 15.149 | 2.408 | 11.780 / 66.326 | 92 |

## Reward identity

| Host | Step max residual | Episode max residual (float64 trace) | Episode mean residual | Float32 vs float64 max |
|---|---:|---:|---:|---:|
| FinalUnseen | 1.694e-08 | 2.216e-06 | 1.170e-06 | 6.851e-05 |
| Teammate 1 | 2.081e-08 | 2.377e-06 | 1.394e-06 | 8.496e-05 |
| Teammate 2 | 1.580e-08 | 2.395e-06 | 1.508e-06 | 7.754e-05 |

## Smoke and runtime checks

All three hosts passed n=4 smoke before their n=100 evaluation. Runtime proprioception59 + depth48×64×1 → CNN64 → actor123; action8. Normalized inputs/actions/rewards and components are finite. Depth values change during rollout, and camera updates were observed in every environment. Sky/background handling is the original `normalized_depth` implementation.

Two initial teammate1 diagnostics stopped before policy rollout because the new script initially referenced `base.num_actions` instead of the wrapper’s `env.num_actions`; error shutdown was delayed by renderer teardown. This evaluation-only assertion was corrected. No host/camera/reward/policy settings were changed. The failed initialization attempts and successful smoke are preserved separately.

## Caveats

- Teammate 1 is the current native training-distribution host, not an independent unseen holdout. FinalUnseen and Teammate 2 are additional evaluation hosts.
- Fall differs by host: FinalUnseen uses terrain-relative torso clearance <0.31 m (including missing terrain hit); Teammate 1/2 use orientation termination. Timeout is 16 s. Teammate 2 additionally truncates at map boundary; it is reported as Other rather than timeout. Termination counts are native term outcomes.
- Terrain seeds remain 2404 / 42 / 71. Different terrain/reset/material/asset configurations and termination definitions prevent treating cross-environment fall rates or returns as a robustness ranking.
- Signed world-X displacement includes terminal position captured immediately before original auto-reset. The legacy preterminal displacement is retained in per-episode CSV. This differs slightly from preterminal displacement in older native Final reports. Mean forward velocity is the mean of each episode’s stepwise world-X velocity average.
- `energy` is the stock action/joint-velocity proxy, not literal measured physical energy.
- Original depth camera optics/extrinsics/preprocessing/timing are preserved across hosts; added sensing supplies the required policy interface and does not alter collision geometry. Simulator rendering/runtime version may differ from original training. Current runtime uses IsaacSim5.0.0-rc.45 and RSL-RL3.0.1 with the RS IsaacLab source checkout; exact original runtime equivalence is not asserted.
- Evaluation uses one seed and one realization per host. No post-evaluation checkpoint selection, training, video, commit or push was performed.

Full precision: [reward table](comparison.csv), [main metrics](main_metrics.csv). Per-host `main/` contains config, source hashes, first-episode CSVs, step contributions and evaluation JSON. `commands.log` records execution commands.
